use std::path::Path;
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

// ── Typaliaser ────────────────────────────────────────────────────────────────

pub type NerModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// ── Match ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub source: String,
}

// ── Etikettmappning ───────────────────────────────────────────────────────────

fn id_to_ui(id: usize, requested: &[String]) -> Option<&'static str> {
    let label = match id {
        9        => "Person",
        3 | 5    => "Person",        // ORG/PRS, PRS/WRK
        10       => "Person",        // LOC/PRS
        8 | 4    => "Organisation",  // ORG, OBJ/ORG
        11       => "Organisation",  // LOC/ORG
        7        => "Plats",
        2        => "Tid",           // TME
        13       => "Händelse",      // EVN
        12       => "Övrigt",        // MSR
        _        => return None,     // O, OBJ, WRK
    };
    if requested.iter().any(|r| r == label) { Some(label) } else { None }
}

// ── Modell-init ───────────────────────────────────────────────────────────────

pub fn load_model(model_path: &Path) -> TractResult<NerModel> {
    tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()
}

pub fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer, String> {
    let mut tok = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Tokenizer-fel: {e}"))?;
    tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length: 512,
        ..Default::default()
    })).map_err(|e| format!("Truncation-fel: {e}"))?;
    Ok(tok)
}

// ── Chunking ──────────────────────────────────────────────────────────────────

const CHUNK_CHARS: usize = 1200;  // Ökad från 800 → 33% färre chunks
const OVERLAP: usize = 80;

fn chunk_text(text: &str) -> Vec<(String, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut chunks = Vec::new();
    let mut pos = 0usize;

    while pos < n {
        let end_raw = (pos + CHUNK_CHARS).min(n);
        let end = if end_raw >= n {
            n
        } else {
            let slice = &chars[pos..end_raw];
            let sentence_break = slice.windows(2)
                .rposition(|w| w[0] == '.' && w[1] == ' ')
                .map(|i| i + 1);
            let word_break = slice.iter().rposition(|&c| c == ' ');
            let cut = sentence_break.or(word_break).unwrap_or(end_raw - pos);
            pos + cut
        };

        let chunk_str: String = chars[pos..end].iter().collect();
        let byte_start: usize = chars[..pos].iter().collect::<String>().len();
        chunks.push((chunk_str, byte_start));

        if end >= n { break; }
        let next_pos = if end > OVERLAP { end - OVERLAP } else { pos + 1 };
        pos = next_pos.max(pos + 1);
    }
    chunks
}

// ── Inferens för ett chunk ────────────────────────────────────────────────────

fn run_chunk(
    chunk: &str,
    chunk_byte_start: usize,
    model: &NerModel,
    tokenizer: &Tokenizer,
    requested: &[String],
    threshold: f32,
) -> Vec<Match> {
    let encoding = match tokenizer.encode(chunk, false) {
        Ok(e) => e,
        Err(_) => return vec![],
    };

    let ids: Vec<i64>   = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mask: Vec<i64>  = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let types: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    let offsets         = encoding.get_offsets();
    let word_ids        = encoding.get_word_ids(); // word_id per token (None = special token)

    let seq_len = ids.len();
    if seq_len == 0 { return vec![]; }

    let mk = |v: Vec<i64>| -> TractResult<TValue> {
        let arr = tract_ndarray::Array2::from_shape_vec((1, seq_len), v)?;
        Ok(arr.into_tensor().into())
    };

    let input_ids_t   = match mk(ids)   { Ok(t) => t, Err(_) => return vec![] };
    let attention_t   = match mk(mask)  { Ok(t) => t, Err(_) => return vec![] };
    let token_types_t = match mk(types) { Ok(t) => t, Err(_) => return vec![] };

    let outputs = match model.run(tvec!(input_ids_t, attention_t, token_types_t)) {
        Ok(o) => o,
        Err(_) => return vec![],
    };

    // logits: [1, seq_len, 14]
    let logits = match outputs[0].to_array_view::<f32>() {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let num_labels = logits.shape()[2];

    // ── Fas 1: aggregera token-prediktioner till ord-nivå via word_ids ────────
    //
    // word_id-strategin löser "Cosmi" ur "Cosmic":
    // Tokenizern delar "Cosmic" → ["Cosmi"(word_id=5), "##c"(word_id=5)].
    // Om "Cosmi" → Person men "##c" → O, ska hela ordet "Cosmic" ändå få
    // etiketten Person (så att det är ett komplett ord som markeras).
    // BTreeMap håller orden i textordning (stigande word_id).
    //
    // (usize, usize, Option<&'static str>) = (byte_start, byte_end, label)
    let mut word_map: BTreeMap<u32, (usize, usize, Option<&'static str>)> = BTreeMap::new();

    // Tröskeln skickas in som parameter från frontend (inställningar-panel).

    for (tok_i, offset) in offsets.iter().enumerate() {
        let Some(word_id) = word_ids[tok_i] else { continue }; // hoppa [CLS]/[SEP]

        // Argmax + softmax-sannolikhet för vinnande klass
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for l in 0..num_labels {
            let v = logits[[0, tok_i, l]];
            if v > best_val { best_val = v; best_id = l; }
        }
        // Numeriskt stabilt softmax (subtrahera max för att undvika overflow)
        let exp_vals: Vec<f32> = (0..num_labels)
            .map(|l| (logits[[0, tok_i, l]] - best_val).exp())
            .collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let best_prob = exp_vals[best_id] / sum_exp;

        // Klassificera bara som entitet om modellen är tillräckligt säker
        let ui_label = if best_id == 0 || best_prob < threshold {
            None
        } else {
            id_to_ui(best_id, requested)
        };

        let entry = word_map.entry(word_id).or_insert((offset.0, offset.1, None));
        if offset.0 < entry.0 { entry.0 = offset.0; }
        if offset.1 > entry.1 { entry.1 = offset.1; }
        // Första entity-klassificerade token i ordet vinner etiketten
        if ui_label.is_some() && entry.2.is_none() {
            entry.2 = ui_label;
        }
    }

    // ── Fas 2: aggregera intilliggande ord med samma entitetsetikett ──────────

    let mut result_matches: Vec<Match> = Vec::new();
    let mut current: Option<(usize, usize, &'static str)> = None;

    let flush = |cur: Option<(usize, usize, &'static str)>,
                 chunk_text: &str,
                 out: &mut Vec<Match>| {
        if let Some((s, e, lbl)) = cur {
            let raw = &chunk_text[s..e.min(chunk_text.len())];
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                let trim_start = s + (raw.len() - raw.trim_start().len());
                out.push(Match {
                    text: trimmed.to_string(),
                    label: lbl.to_string(),
                    start: chunk_byte_start + trim_start,
                    end: chunk_byte_start + trim_start + trimmed.len(),
                    source: "kblab".to_string(),
                });
            }
        }
    };

    for &(tok_start, tok_end, ui_label) in word_map.values() {
        match (ui_label, &current) {
            (Some(lbl), Some(cur)) if lbl == cur.2 => {
                let gap = tok_start.saturating_sub(cur.1);
                if gap <= 3 {
                    let (s, _, l) = *cur;
                    current = Some((s, tok_end, l));
                } else {
                    let taken = current.take();
                    flush(taken, chunk, &mut result_matches);
                    current = Some((tok_start, tok_end, lbl));
                }
            }
            (Some(lbl), _) => {
                let taken = current.take();
                flush(taken, chunk, &mut result_matches);
                current = Some((tok_start, tok_end, lbl));
            }
            (None, _) => {
                let taken = current.take();
                flush(taken, chunk, &mut result_matches);
            }
        }
    }
    flush(current, chunk, &mut result_matches);

    result_matches
}

// ── Deduplicering ─────────────────────────────────────────────────────────────

fn deduplicate(mut matches: Vec<Match>) -> Vec<Match> {
    matches.sort_by_key(|m| m.start);
    let mut result: Vec<Match> = Vec::new();
    let mut last_end = 0usize;
    for m in matches {
        if m.start >= last_end {
            last_end = m.end;
            result.push(m);
        }
    }
    result
}

// ── Publik ingångspunkt ───────────────────────────────────────────────────────

pub fn run_ner(
    text: &str,
    requested: &[String],
    threshold: f32,
    model: &NerModel,
    tokenizer: &Tokenizer,
    progress_cb: impl Fn(usize, usize),
) -> Vec<Match> {
    if requested.is_empty() { return vec![]; }

    let chunks = chunk_text(text);
    let total = chunks.len();
    let mut all_matches: Vec<Match> = Vec::new();

    for (i, (chunk, byte_start)) in chunks.iter().enumerate() {
        if i % 5 == 0 {
            progress_cb(i, total);
        }
        let mut ms = run_chunk(chunk, *byte_start, model, tokenizer, requested, threshold);
        all_matches.append(&mut ms);
    }
    // Sista progress
    progress_cb(total, total);

    deduplicate(all_matches)
}
