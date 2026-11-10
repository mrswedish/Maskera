use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ndarray::Array2;
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use tokenizers::{Encoding, Tokenizer};

// ── Global ORT-miljö ──────────────────────────────────────────────────────────

static ORT_ENV: OnceLock<Result<Environment, String>> = OnceLock::new();

fn get_ort_env() -> Result<&'static Environment, String> {
    ORT_ENV
        .get_or_init(|| {
            ort::init()
                .with_telemetry(false)
                .build()
                .map_err(|e| format!("Kunde inte initiera ONNX Runtime: {e}"))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

// ── Modell-wrapper ────────────────────────────────────────────────────────────

pub struct NerModel {
    session: Mutex<Session>,
    has_token_type_ids: bool,
}

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

#[inline]
fn id_to_ui(id: usize, requested_mask: u8) -> Option<&'static str> {
    // Bitmask för requested-kategorier eliminerar upprepade String-jämförelser i inre loopen
    let (bit, label) = match id {
        9 | 3 | 5 | 10 => (1 << 0, "Person"),       // PER, ORG/PRS, PRS/WRK, LOC/PRS
        8 | 4 | 11     => (1 << 1, "Organisation"), // ORG, OBJ/ORG, LOC/ORG
        7              => (1 << 2, "Plats"),        // LOC
        12             => (1 << 3, "Övrigt"),       // MSR
        2              => (1 << 4, "Tid"),          // TME
        13             => (1 << 5, "Händelse"),     // EVN
        _              => return None,              // O, OBJ, WRK
    };
    if (requested_mask & bit) != 0 {
        Some(label)
    } else {
        None
    }
}

fn build_requested_mask(requested: &[String]) -> u8 {
    let mut mask = 0u8;
    for r in requested {
        match r.as_str() {
            "Person"       => mask |= 1 << 0,
            "Organisation" => mask |= 1 << 1,
            "Plats"        => mask |= 1 << 2,
            "Övrigt"       => mask |= 1 << 3,
            "Tid"          => mask |= 1 << 4,
            "Händelse"     => mask |= 1 << 5,
            _ => {}
        }
    }
    mask
}

// ── Modell-init ───────────────────────────────────────────────────────────────

pub fn load_model(model_path: &Path) -> Result<NerModel, String> {
    let env = get_ort_env()?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 16))
        .unwrap_or(4);

    let session = Session::builder(env)
        .map_err(|e| format!("SessionBuilder-fel: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap_or_else(|e| e.recover())
        .with_intra_threads(threads)
        .unwrap_or_else(|e| e.recover())
        .commit_from_file(model_path)
        .map_err(|e| format!("Kunde inte ladda ONNX-modell: {e}"))?;

    let has_token_type_ids = session
        .inputs()
        .iter()
        .any(|inp| inp.name() == "token_type_ids");

    Ok(NerModel {
        session: Mutex::new(session),
        has_token_type_ids,
    })
}

pub fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer, String> {
    let mut tok = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Tokenizer-fel: {e}"))?;
    tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length: 512,
        ..Default::default()
    }))
    .map_err(|e| format!("Truncation-fel: {e}"))?;
    Ok(tok)
}

// ── Smart zero-copy chunking ──────────────────────────────────────────────────

const CHUNK_CHARS: usize = 1600;
const MIN_BREAK_SEARCH: usize = 700;
const MID_SENTENCE_OVERLAP: usize = 120;
const DENSE_FALLBACK_CHARS: usize = 500;
const BATCH_SIZE: usize = 8;

/// Delar texten i naturliga stycke-/meningsblock utan onödig kopiering (`O(N)`).
/// Lägger endast överlapp när vi tvingas bryta mitt i en mening — vid stycke-
/// eller meningsslut behövs inget överlapp alls (sparar ~30–40 % beräkning).
fn chunk_text_with_limit<'a>(text: &'a str, max_chars: usize) -> Vec<(&'a str, usize)> {
    if text.is_empty() {
        return vec![];
    }

    let char_offsets: Vec<(usize, char)> = text.char_indices().collect();
    let n = char_offsets.len();
    let mut chunks = Vec::new();
    let mut pos = 0usize;

    let min_search = (max_chars / 2).min(MIN_BREAK_SEARCH);

    while pos < n {
        let end_raw = (pos + max_chars).min(n);
        if end_raw >= n {
            let byte_start = char_offsets[pos].0;
            let slice = &text[byte_start..];
            if !slice.trim().is_empty() {
                chunks.push((slice, byte_start));
            }
            break;
        }

        let search_from = (pos + min_search).min(end_raw);
        let window = &char_offsets[search_from..end_raw];

        // 1. Styckebrytning (\n\n) eller radbrytning (\n) → ren brytpunkt (0 överlapp)
        let para_break = window
            .windows(2)
            .rposition(|w| w[0].1 == '\n' && w[1].1 == '\n')
            .map(|i| search_from + i + 2)
            .or_else(|| {
                window
                    .iter()
                    .rposition(|&(_, c)| c == '\n')
                    .map(|i| search_from + i + 1)
            });

        // 2. Meningsslut (. / ! / ? följt av mellanslag) → ren brytpunkt (0 överlapp)
        let sentence_break = if para_break.is_none() {
            window
                .windows(2)
                .rposition(|w| matches!(w[0].1, '.' | '!' | '?') && w[1].1.is_whitespace())
                .map(|i| search_from + i + 2)
        } else {
            None
        };

        let (end, clean_break) = if let Some(cut) = para_break.or(sentence_break) {
            (cut, true)
        } else if let Some(space_rel) = char_offsets[pos..end_raw]
            .iter()
            .rposition(|&(_, c)| c.is_whitespace())
        {
            let cut = pos + space_rel + 1;
            if cut > pos {
                (cut, false)
            } else {
                (end_raw, false)
            }
        } else {
            (end_raw, false)
        };

        let byte_start = char_offsets[pos].0;
        let byte_end = char_offsets[end].0;
        let slice = &text[byte_start..byte_end];
        if !slice.trim().is_empty() {
            chunks.push((slice, byte_start));
        }

        if clean_break {
            pos = end;
        } else {
            let overlap = MID_SENTENCE_OVERLAP.min(max_chars / 3);
            let mut next_pos = end.saturating_sub(overlap).max(pos + 1);
            // Backa/framåt till närmaste ordgräns så vi inte klipper mitt i ett ord
            while next_pos < end && !char_offsets[next_pos - 1].1.is_whitespace() {
                next_pos += 1;
            }
            if next_pos >= end {
                next_pos = end;
            }
            pos = next_pos;
        }
    }

    chunks
}

// ── Post-processing av ett chunks logits ──────────────────────────────────────

fn extract_matches_from_logits(
    chunk: &str,
    chunk_byte_start: usize,
    encoding: &Encoding,
    chunk_logits: &[f32],
    num_labels: usize,
    requested_mask: u8,
    threshold: f32,
) -> Vec<Match> {
    let offsets = encoding.get_offsets();
    let word_ids = encoding.get_word_ids();
    let seq_len = encoding.get_ids().len();

    // word_id -> (byte_start, byte_end, best_label, best_prob)
    let mut word_map: BTreeMap<u32, (usize, usize, Option<&'static str>, f32)> = BTreeMap::new();

    for tok_i in 0..seq_len {
        let Some(word_id) = word_ids[tok_i] else {
            continue; // hoppa över [CLS] och [SEP]
        };
        let offset = offsets[tok_i];
        let row = &chunk_logits[tok_i * num_labels..(tok_i + 1) * num_labels];

        let mut best_id = 0usize;
        let mut best_val = row[0];
        for (l, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_id = l;
            }
        }

        // Snabbväg: 90–95 % av alla tokens är "O" (id 0) eller en icke-efterfrågad kategori.
        // Då hoppar vi över alla exp()-beräkningar helt.
        let mut chosen_label = None;
        let mut best_prob = 0.0f32;
        if best_id != 0 {
            if let Some(lbl) = id_to_ui(best_id, requested_mask) {
                let mut sum_exp = 0.0f32;
                for &v in row {
                    sum_exp += (v - best_val).exp();
                }
                // exp(best_val - best_val) == 1.0
                best_prob = 1.0 / sum_exp;
                if best_prob >= threshold {
                    chosen_label = Some(lbl);
                }
            }
        }

        let entry = word_map
            .entry(word_id)
            .or_insert((offset.0, offset.1, None, 0.0));
        if offset.0 < entry.0 {
            entry.0 = offset.0;
        }
        if offset.1 > entry.1 {
            entry.1 = offset.1;
        }
        if let Some(lbl) = chosen_label {
            if best_prob > entry.3 {
                entry.2 = Some(lbl);
                entry.3 = best_prob;
            }
        }
    }

    // Aggregera intilliggande ord med samma entitetsetikett
    let mut result_matches: Vec<Match> = Vec::new();
    let mut current: Option<(usize, usize, &'static str)> = None;

    let flush = |cur: Option<(usize, usize, &'static str)>,
                 chunk_text: &str,
                 out: &mut Vec<Match>| {
        if let Some((s, e, lbl)) = cur {
            let safe_e = e.min(chunk_text.len());
            if s >= safe_e {
                return;
            }
            let raw = &chunk_text[s..safe_e];
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

    for &(tok_start, tok_end, ui_label, _prob) in word_map.values() {
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
    let mut seen = HashSet::with_capacity(matches.len());
    matches.retain(|m| seen.insert((m.start, m.end, m.label.clone())));

    matches.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
    let mut result: Vec<Match> = Vec::with_capacity(matches.len());
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
    let requested_mask = build_requested_mask(requested);
    if requested_mask == 0 || text.trim().is_empty() {
        return vec![];
    }

    let initial_chunks = chunk_text_with_limit(text, CHUNK_CHARS);
    if initial_chunks.is_empty() {
        return vec![];
    }

    // 1. Parallell batch-tokenisering (använder rayon internt i tokenizers)
    let raw_slices: Vec<&str> = initial_chunks.iter().map(|(s, _)| *s).collect();
    let initial_encodings = match tokenizer.encode_batch(raw_slices, true) {
        Ok(enc) => enc,
        Err(_) => return vec![],
    };

    // Om något chunk slår i taket (>= 510 tokens, t.ex. tabeller med många symboler),
    // dela upp just det chunket i mindre bitar så att inga tokens någonsin trunkeras bort.
    let mut prepared: Vec<(&str, usize, Encoding)> = Vec::with_capacity(initial_chunks.len());
    for ((slice, byte_start), enc) in initial_chunks.into_iter().zip(initial_encodings) {
        if enc.get_ids().len() >= 510 && slice.chars().count() > DENSE_FALLBACK_CHARS {
            let sub_chunks = chunk_text_with_limit(slice, DENSE_FALLBACK_CHARS);
            let sub_slices: Vec<&str> = sub_chunks.iter().map(|(s, _)| *s).collect();
            if let Ok(sub_encs) = tokenizer.encode_batch(sub_slices, true) {
                for ((sub_s, rel_start), sub_e) in sub_chunks.into_iter().zip(sub_encs) {
                    if !sub_e.get_ids().is_empty() {
                        prepared.push((sub_s, byte_start + rel_start, sub_e));
                    }
                }
                continue;
            }
        }
        if !enc.get_ids().is_empty() {
            prepared.push((slice, byte_start, enc));
        }
    }

    let total = prepared.len();
    if total == 0 {
        return vec![];
    }
    progress_cb(0, total);

    // 2. Sortera index efter sekvenslängd för minimal padding i varje mini-batch
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by_key(|&idx| prepared[idx].2.get_ids().len());

    let mut session = match model.session.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    let mut per_chunk_matches: Vec<Vec<Match>> = vec![Vec::new(); total];
    let mut completed = 0usize;

    for batch_indices in order.chunks(BATCH_SIZE) {
        let b_size = batch_indices.len();
        let max_seq_len = batch_indices
            .iter()
            .map(|&idx| prepared[idx].2.get_ids().len())
            .max()
            .unwrap_or(0);

        if max_seq_len == 0 {
            completed += b_size;
            progress_cb(completed, total);
            continue;
        }

        let mut ids_flat = vec![0i64; b_size * max_seq_len];
        let mut mask_flat = vec![0i64; b_size * max_seq_len];

        for (row_i, &chunk_idx) in batch_indices.iter().enumerate() {
            let enc = &prepared[chunk_idx].2;
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let row_start = row_i * max_seq_len;
            for (j, &id) in ids.iter().enumerate() {
                ids_flat[row_start + j] = id as i64;
            }
            for (j, &m) in mask.iter().enumerate() {
                mask_flat[row_start + j] = m as i64;
            }
        }

        let Ok(ids_arr) = Array2::from_shape_vec((b_size, max_seq_len), ids_flat) else {
            continue;
        };
        let Ok(mask_arr) = Array2::from_shape_vec((b_size, max_seq_len), mask_flat) else {
            continue;
        };

        let Ok(ids_ref) = TensorRef::from_array_view(&ids_arr) else {
            continue;
        };
        let Ok(mask_ref) = TensorRef::from_array_view(&mask_arr) else {
            continue;
        };

        let run_res = if model.has_token_type_ids {
            let types_arr = Array2::<i64>::zeros((b_size, max_seq_len));
            match TensorRef::from_array_view(&types_arr) {
                Ok(types_ref) => session.run(ort::inputs![
                    "input_ids" => ids_ref,
                    "attention_mask" => mask_ref,
                    "token_type_ids" => types_ref,
                ]),
                Err(_) => continue,
            }
        } else {
            session.run(ort::inputs![
                "input_ids" => ids_ref,
                "attention_mask" => mask_ref,
            ])
        };

        if let Ok(outputs) = run_res {
            if let Ok((shape, logits_slice)) = outputs[0].try_extract_tensor::<f32>() {
                if shape.len() == 3 {
                    let num_labels = shape[2] as usize;
                    let chunk_stride = max_seq_len * num_labels;

                    for (row_i, &chunk_idx) in batch_indices.iter().enumerate() {
                        let (chunk_str, byte_start, ref enc) = prepared[chunk_idx];
                        let slice_start = row_i * chunk_stride;
                        let slice_end = slice_start + chunk_stride;
                        if slice_end <= logits_slice.len() {
                            per_chunk_matches[chunk_idx] = extract_matches_from_logits(
                                chunk_str,
                                byte_start,
                                enc,
                                &logits_slice[slice_start..slice_end],
                                num_labels,
                                requested_mask,
                                threshold,
                            );
                        }
                    }
                }
            }
        }

        completed += b_size;
        progress_cb(completed, total);
    }

    let all_matches: Vec<Match> = per_chunk_matches.into_iter().flatten().collect();
    deduplicate(all_matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_preserve_exact_utf8_byte_offsets() {
        let sentence = "Handläggare Åsa Öberg i Ängelholm granskade ärendet noggrant.\n";
        let full = sentence.repeat(80);
        let chunks = chunk_text_with_limit(&full, CHUNK_CHARS);
        assert!(chunks.len() >= 3);
        for (slice, byte_start) in &chunks {
            assert_eq!(*slice, &full[*byte_start..*byte_start + slice.len()]);
        }
    }
}
