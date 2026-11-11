use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::ner::get_ort_env;

const EMBED_DIM: usize = 768;
const MAX_PARAGRAPH_CHARS: usize = 1200;
const MIN_PARAGRAPH_CHARS: usize = 25;
/// Kortare rader (t.ex. i mejl med dubbla radbrytningar) slås ihop upp till ~320 tecken
/// så att modellen får sammanhängande kontext och färre anrop.
const MERGE_TARGET_CHARS: usize = 320;
const BATCH_SIZE: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParagraphFinding {
    pub paragraph_index: usize,
    pub preview: String,
    pub start: usize,
    pub end: usize,
    pub category: String,
    pub axis: String,     // "integritet" | "affär" | "säkerhet"
    pub severity: String, // "high" | "medium"
    pub confidence: f32,  // 0.0..1.0 kalibrerad indikator
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// "safe" | "mask_required" | "caution_after_mask" | "do_not_share"
    pub verdict_before_mask: String,
    /// Bedömning om användaren kopierar den MASKERADE versionen till extern AI
    pub verdict_after_mask: String,
    pub summary_before: String,
    pub summary_after: String,
    pub active_pii_count: usize,
    pub unmasked_pii_count: usize,
    pub secret_count: usize,
    pub semantic_available: bool,
    pub paragraphs_scanned: usize,
    pub scan_duration_ms: u64,
    pub findings: Vec<ParagraphFinding>,
}

struct PrototypeGroup {
    category: &'static str,
    axis: &'static str,
    severity: &'static str,
    reason: &'static str,
    prompts: &'static [&'static str],
}

const PROTOTYPE_GROUPS: &[PrototypeGroup] = &[
    PrototypeGroup {
        category: "Hälsa & Vård (GDPR Art. 9)",
        axis: "integritet",
        severity: "high",
        reason: "Beskriver medicinskt tillstånd, diagnos, behandling eller vårdkontakt som kan röja känsliga personuppgifter även utan namn.",
        prompts: &[
            "Patientjournal, sjukdomshistoria, medicinsk diagnos, läkemedelsbehandling, psykiatrisk bedömning, sjukskrivning eller rehabiliteringsärende.",
            "Beskrivning av en persons fysiska eller psykiska hälsotillstånd, symtom, operation, vårdplan, provsvar, missbruk eller funktionsnedsättning.",
            "Patienten har varit sjukskriven på grund av utmattning, depression, ångest, smärta eller genomgår utredning och medicinering inom vården.",
        ],
    },
    PrototypeGroup {
        category: "Känslig personuppgift / Lagöverträdelse (GDPR Art. 9–10)",
        axis: "integritet",
        severity: "high",
        reason: "Innehåller uppgifter om brottsmisstanke, rättsprocess, socialtjänst, politisk åsikt, fackförening eller religion.",
        prompts: &[
            "Uppgifter om politiska åsikter, partimedlemskap, facklig tillhörighet, religiös övertygelse eller sexuell läggning.",
            "Brottsmisstanke, polisanmälan, förundersökning, åtal, dom i brottmål, disciplinärende, tvångsvård eller utredning hos socialtjänsten.",
            "Orosanmälan, ekonomiskt bistånd, skuldsanering hos Kronofogden, omhändertagande enligt LVU eller LVM eller pågående rättstvist.",
        ],
    },
    PrototypeGroup {
        category: "Affärskritiskt & Företagshemligheter",
        axis: "affär",
        severity: "high",
        reason: "Liknar intern affärsstrategi, opublicerade siffror, prissättning, förvärvsbud eller konfidentiella avtalsvillkor.",
        prompts: &[
            "Konfidentiell affärsstrategi, företagshemlighet, opublicerat bokslut, intern budget, vinstmarginal, företagsförvärv eller styrelseprotokoll.",
            "Icke-offentlig kundoffert, konfidentiellt avtal, timpriser, rabattstruktur, inköpspriser, upphandlingsanbud eller förhandlingsstrategi.",
            "Inför styrelsemötet föreslår vi ett konfidentiellt bud, opublicerad omsättningsprognos, bruttomarginal eller exklusivt samarbetsavtal innan rapporten offentliggörs.",
        ],
    },
    PrototypeGroup {
        category: "Säkerhet, IT-arkitektur & Intern kod",
        axis: "säkerhet",
        severity: "high",
        reason: "Berör intern IT-säkerhet, sårbarheter, nätverks-/systemarkitektur, behörigheter eller proprietär systemlogik.",
        prompts: &[
            "Intern IT-säkerhet, sårbarhetsrapport, penetrationstest, brandväggsregler, intern nätverksarkitektur, säkerhetsincident eller åtkomstkontroller.",
            "Proprietär intern källkod, produktionskonfiguration, databasstruktur med affärslogik, API-nycklar eller säkerhetsklassad verksamhet.",
            "Vid säkerhetsgranskningen upptäcktes en sårbarhet i autentiseringstjänsten bakom brandväggen som ger obehörig åtkomst till interna system eller databaser.",
        ],
    },
    PrototypeGroup {
        category: "Tjänstekort, eID, Behörighet & Intranät",
        axis: "säkerhet",
        severity: "high",
        reason: "Innehåller uppgifter om SITHS-kort, tjänstekort, e-legitimation/certifikat, korthandläggare eller interna intranätlänkar och säkerhetsrutiner.",
        prompts: &[
            "Ditt SITHS eID certifikat på ditt SITHS-kort, tjänstekort, reservkort eller konsultkort kommer att upphöra och behöver förnyas hos korthandläggare eller ID-fotokontor.",
            "Intern information om e-legitimation, behörighetskort, HSA-ID, aktivering av tjänstekort, inloggningscertifikat, passerkort eller länk till internt intranät inom region, myndighet eller företag.",
        ],
    },
    PrototypeGroup {
        category: "HR & Personalärende",
        axis: "affär",
        severity: "medium",
        reason: "Innehåller bedömning av medarbetare/kandidat, löneuppgifter, arbetsrättsligt ärende eller intern personalfråga.",
        prompts: &[
            "Medarbetarsamtal, individuell lönesättning, löneanspråk, omplacering, skriftlig varning, arbetsrättslig förhandling, uppsägning eller kandidatbedömning.",
        ],
    },
];

const NEUTRAL_PROMPTS: &[&str] = &[
    "Allmän informationstext, offentlig webbtext, pressmeddelande, produktbeskrivning eller utbildningsmaterial för allmänheten.",
    "Vardagligt meddelande om kaffe, öppen inbjudan till webbinarium, kurs eller publik konferens nästa vecka.",
    "Sammanfattning av offentliga riktlinjer, publicerad myndighetsrapport, öppen presentation eller generella skrivtips.",
    "Allmän fråga om språkgranskning, sammanfattning av offentlig artikel, recept, öppen källkod eller generella programmeringskoncept.",
    "Kort allmän fråga om öppettider, publik leveransinformation eller allmän produktfunktion utan interna system eller personuppgifter.",
];

struct EmbeddedPrototype {
    group_idx: usize,
    vec: Vec<f32>,
}

pub struct EmbeddingClassifier {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    prototypes: Vec<EmbeddedPrototype>,
    neutral_vecs: Vec<Vec<f32>>,
}

fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 1e-12 {
        let inv = 1.0 / norm_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

impl EmbeddingClassifier {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let env = get_ort_env()?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 16))
            .unwrap_or(4);

        let model_path = dir.join("model_q4.onnx");
        let tok_path = dir.join("tokenizer.json");

        let session = Session::builder(env)
            .map_err(|e| format!("SessionBuilder-fel (EmbeddingGemma 2): {e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(threads)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(&model_path)
            .map_err(|e| format!("Kunde inte ladda EmbeddingGemma 2: {e}"))?;

        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("Tokenizer-fel (EmbeddingGemma 2): {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 1024,
                ..Default::default()
            }))
            .map_err(|e| format!("Truncation-fel: {e}"))?;

        let mut instance = Self {
            session: Mutex::new(session),
            tokenizer,
            prototypes: Vec::new(),
            neutral_vecs: Vec::new(),
        };

        // Förberäkna alla ankar-vektorer en gång vid laddning
        let mut all_anchor_texts: Vec<String> = Vec::new();
        let mut anchor_meta: Vec<Option<usize>> = Vec::new();

        for (g_idx, grp) in PROTOTYPE_GROUPS.iter().enumerate() {
            for &p in grp.prompts {
                all_anchor_texts.push(format!("task: classification | query: {p}"));
                anchor_meta.push(Some(g_idx));
            }
        }
        for &n in NEUTRAL_PROMPTS {
            all_anchor_texts.push(format!("task: classification | query: {n}"));
            anchor_meta.push(None);
        }

        let refs: Vec<&str> = all_anchor_texts.iter().map(|s| s.as_str()).collect();
        let vecs = instance.embed_texts_with_progress(&refs, |_, _| {})?;
        for (meta, vec) in anchor_meta.into_iter().zip(vecs) {
            if let Some(g_idx) = meta {
                instance
                    .prototypes
                    .push(EmbeddedPrototype { group_idx: g_idx, vec });
            } else {
                instance.neutral_vecs.push(vec);
            }
        }

        Ok(instance)
    }

    pub fn embed_texts_with_progress<F>(
        &self,
        texts: &[&str],
        mut on_progress: F,
    ) -> Result<Vec<Vec<f32>>, String>
    where
        F: FnMut(usize, usize),
    {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total = texts.len();
        on_progress(0, total);

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("Tokeniseringsfel: {e}"))?;

        let empty_mm = Array2::<f32>::zeros((0, 512));

        let mut session = match self.session.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };

        let mut results = vec![vec![0.0f32; EMBED_DIM]; texts.len()];
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| encodings[i].get_ids().len());

        let mut completed = 0usize;

        for batch_indices in order.chunks(BATCH_SIZE) {
            let b_size = batch_indices.len();
            let max_len = batch_indices
                .iter()
                .map(|&i| encodings[i].get_ids().len())
                .max()
                .unwrap_or(0);
            if max_len == 0 {
                completed += b_size;
                on_progress(completed, total);
                continue;
            }

            let mut ids_flat = vec![0i64; b_size * max_len];
            let mut mask_flat = vec![0i64; b_size * max_len];

            for (row_i, &orig_idx) in batch_indices.iter().enumerate() {
                let enc = &encodings[orig_idx];
                let row_start = row_i * max_len;
                for (j, &id) in enc.get_ids().iter().enumerate() {
                    ids_flat[row_start + j] = id as i64;
                }
                for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                    mask_flat[row_start + j] = m as i64;
                }
            }

            let ids_arr = Array2::from_shape_vec((b_size, max_len), ids_flat)
                .map_err(|e| e.to_string())?;
            let mask_arr = Array2::from_shape_vec((b_size, max_len), mask_flat)
                .map_err(|e| e.to_string())?;

            // Viktigt: skapa zero-copy TensorRef per batch utan `.clone()`
            let ids_ref = TensorRef::from_array_view(&ids_arr).map_err(|e| e.to_string())?;
            let mask_ref = TensorRef::from_array_view(&mask_arr).map_err(|e| e.to_string())?;
            let empty_img = TensorRef::from_array_view(&empty_mm).map_err(|e| e.to_string())?;
            let empty_vid = TensorRef::from_array_view(&empty_mm).map_err(|e| e.to_string())?;
            let empty_aud = TensorRef::from_array_view(&empty_mm).map_err(|e| e.to_string())?;

            let outputs = session
                .run(ort::inputs![
                    "input_ids" => ids_ref,
                    "attention_mask" => mask_ref,
                    "image_features" => empty_img,
                    "video_features" => empty_vid,
                    "audio_features" => empty_aud,
                ])
                .map_err(|e| format!("EmbeddingGemma 2 körfel: {e}"))?;

            let emb_val = outputs
                .get("sentence_embedding")
                .ok_or_else(|| "Saknar sentence_embedding-utgång".to_string())?;
            let (_shape, emb_slice) = emb_val
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Kunde inte läsa sentence_embedding: {e}"))?;

            for (row_i, &orig_idx) in batch_indices.iter().enumerate() {
                let start = row_i * EMBED_DIM;
                let end = start + EMBED_DIM;
                if end <= emb_slice.len() {
                    let mut v = emb_slice[start..end].to_vec();
                    l2_normalize(&mut v);
                    results[orig_idx] = v;
                }
            }

            completed += b_size;
            on_progress(completed, total);
        }

        Ok(results)
    }

    pub fn scan_paragraphs_with_progress<F>(
        &self,
        original_text: &str,
        masked_text: &str,
        on_progress: F,
    ) -> (Vec<ParagraphFinding>, usize)
    where
        F: FnMut(usize, usize),
    {
        let paragraphs = split_paragraphs_paired(original_text, masked_text);
        if paragraphs.is_empty() {
            return (vec![], 0);
        }
        let total_paras = paragraphs.len();

        let formatted: Vec<String> = paragraphs
            .iter()
            .map(|(_, _, slice)| format!("task: classification | query: {slice}"))
            .collect();
        let refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();
        let Ok(vecs) = self.embed_texts_with_progress(&refs, on_progress) else {
            return (vec![], total_paras);
        };

        let mut findings = Vec::new();
        for ((p_idx, (byte_start, byte_end, slice)), vec) in
            paragraphs.into_iter().enumerate().zip(vecs)
        {
            let neutral_best = self
                .neutral_vecs
                .iter()
                .map(|nv| dot(&vec, nv))
                .fold(f32::NEG_INFINITY, f32::max);

            let mut best_group: Option<(usize, f32)> = None;
            for proto in &self.prototypes {
                let sim = dot(&vec, &proto.vec);
                if best_group.is_none_or(|(_, b)| sim > b) {
                    best_group = Some((proto.group_idx, sim));
                }
            }

            if let Some((g_idx, best_sim)) = best_group {
                let margin = best_sim - neutral_best;
                // Kalibrerat mot EmbeddingGemma 2 (768d, `task: classification | query:`):
                // vardaglig/publik text har margin <= 0.00, medan känsliga stycken ligger på ~0.032..0.19+
                if best_sim >= 0.76 && margin >= 0.030 {
                    let grp = &PROTOTYPE_GROUPS[g_idx];
                    let clean_line = slice.replace('\n', " ");
                    let preview: String = clean_line.chars().take(135).collect();
                    let preview = if clean_line.chars().count() > 135 {
                        format!("{}…", preview.trim_end())
                    } else {
                        preview
                    };
                    let conf = (0.62 + ((margin - 0.030) / 0.090) * 0.35).clamp(0.62, 0.98);
                    let severity = if grp.severity == "high" && (margin >= 0.050 || best_sim >= 0.83)
                    {
                        "high"
                    } else {
                        "medium"
                    };

                    findings.push(ParagraphFinding {
                        paragraph_index: p_idx + 1,
                        preview,
                        start: byte_start,
                        end: byte_end,
                        category: grp.category.to_string(),
                        axis: grp.axis.to_string(),
                        severity: severity.to_string(),
                        confidence: conf,
                        reason: grp.reason.to_string(),
                    });
                }
            }
        }

        (findings, total_paras)
    }
}

fn raw_double_newline_blocks(text: &str) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let is_break = bytes[i] == b'\n' && (i + 1 < bytes.len() && bytes[i + 1] == b'\n');
        if is_break {
            if let Some(span) = trimmed_span(text, start, i) {
                blocks.push(span);
            }
            while i < bytes.len() && bytes[i] == b'\n' {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    if start < bytes.len() {
        if let Some(span) = trimmed_span(text, start, bytes.len()) {
            blocks.push(span);
        }
    }
    blocks
}

fn trimmed_span(full: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let raw = &full[start..end];
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let trim_offset = raw.len() - raw.trim_start().len();
    let p_start = start + trim_offset;
    let p_end = p_start + trimmed.len();
    Some((p_start, p_end))
}

/// Delar upp `masked_text` i avsnitt för EmbeddingGemma 2 samtidigt som varje avsnitts
/// exakta byte-intervall `(orig_start, orig_end)` i `original_text` bevaras för UI-markering.
fn split_paragraphs_paired<'a>(
    original_text: &'a str,
    masked_text: &'a str,
) -> Vec<(usize, usize, &'a str)> {
    let orig_blocks = raw_double_newline_blocks(original_text);
    let mask_blocks = raw_double_newline_blocks(masked_text);

    if original_text.is_empty() || orig_blocks.len() != mask_blocks.len() {
        return split_paragraphs(masked_text);
    }

    let mut raw_paired: Vec<(usize, usize, usize, usize)> = Vec::new();
    for ((o_start, o_end), (m_start, m_end)) in orig_blocks.into_iter().zip(mask_blocks) {
        let m_trimmed = masked_text[m_start..m_end].trim();
        let o_trimmed = original_text[o_start..o_end].trim();
        if m_trimmed.chars().count() < MIN_PARAGRAPH_CHARS
            && o_trimmed.chars().count() < MIN_PARAGRAPH_CHARS
        {
            continue;
        }
        if m_trimmed.chars().count() <= MAX_PARAGRAPH_CHARS {
            raw_paired.push((o_start, o_end, m_start, m_end));
        } else {
            // Vid mycket långa stycken delas de upp längs meningsgränser
            let m_subs = split_long_span(masked_text, m_start, m_end);
            let o_subs = split_long_span(original_text, o_start, o_end);
            if m_subs.len() == o_subs.len() {
                for ((os, oe), (ms, me)) in o_subs.into_iter().zip(m_subs) {
                    raw_paired.push((os, oe, ms, me));
                }
            } else {
                raw_paired.push((o_start, o_end, m_start, m_end));
            }
        }
    }

    let mut merged: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(raw_paired.len());
    for (o_start, o_end, m_start, m_end) in raw_paired {
        if let Some((_, prev_o_end, prev_m_start, prev_m_end)) = merged.last_mut() {
            let prev_chars = masked_text[*prev_m_start..*prev_m_end].chars().count();
            let curr_chars = masked_text[m_start..m_end].chars().count();
            if (prev_chars < 140 || curr_chars < 95)
                && (prev_chars + curr_chars <= MERGE_TARGET_CHARS)
            {
                *prev_o_end = o_end;
                *prev_m_end = m_end;
                continue;
            }
        }
        merged.push((o_start, o_end, m_start, m_end));
    }

    merged
        .into_iter()
        .map(|(o_start, o_end, m_start, m_end)| {
            (o_start, o_end, masked_text[m_start..m_end].trim())
        })
        .collect()
}

fn split_long_span(full: &str, p_start: usize, p_end: usize) -> Vec<(usize, usize)> {
    let trimmed = &full[p_start..p_end];
    let mut out = Vec::new();
    let mut sub_start_byte = 0usize;
    let mut count = 0usize;
    for (b_idx, ch) in trimmed.char_indices() {
        count += 1;
        if count >= MAX_PARAGRAPH_CHARS && (ch == '.' || ch == '\n') {
            let next_b = b_idx + ch.len_utf8();
            let sub = trimmed[sub_start_byte..next_b].trim();
            if sub.chars().count() >= MIN_PARAGRAPH_CHARS {
                out.push((p_start + sub_start_byte, p_start + next_b));
            }
            sub_start_byte = next_b;
            count = 0;
        }
    }
    if sub_start_byte < trimmed.len() {
        let sub = trimmed[sub_start_byte..].trim();
        if sub.chars().count() >= MIN_PARAGRAPH_CHARS {
            out.push((p_start + sub_start_byte, p_end));
        }
    }
    out
}

fn split_paragraphs(text: &str) -> Vec<(usize, usize, &str)> {
    let mut raw_paras = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let is_break = bytes[i] == b'\n' && (i + 1 < bytes.len() && bytes[i + 1] == b'\n');
        if is_break {
            push_paragraph(text, start, i, &mut raw_paras);
            while i < bytes.len() && bytes[i] == b'\n' {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    if start < bytes.len() {
        push_paragraph(text, start, bytes.len(), &mut raw_paras);
    }

    // Om texten består av många korta enradiga stycken (t.ex. ett e-postmeddelande med
    // dubbla radbrytningar mellan varje mening), slå ihop korta grannstycken (< 140 tecken)
    // upp till MERGE_TARGET_CHARS så att modellen får rikare kontext och färre anrop.
    let mut merged: Vec<(usize, usize, &str)> = Vec::with_capacity(raw_paras.len());
    for (p_start, p_end, p_slice) in raw_paras {
        if let Some((prev_start, prev_end, _)) = merged.last_mut() {
            let prev_chars = text[*prev_start..*prev_end].chars().count();
            let curr_chars = p_slice.chars().count();
            if (prev_chars < 140 || curr_chars < 95)
                && (prev_chars + curr_chars <= MERGE_TARGET_CHARS)
            {
                *prev_end = p_end;
                continue;
            }
        }
        merged.push((p_start, p_end, p_slice));
    }
    for item in &mut merged {
        item.2 = text[item.0..item.1].trim();
    }
    merged
}

fn push_paragraph<'a>(
    full: &'a str,
    start: usize,
    end: usize,
    out: &mut Vec<(usize, usize, &'a str)>,
) {
    let raw = &full[start..end];
    let trimmed = raw.trim();
    if trimmed.chars().count() < MIN_PARAGRAPH_CHARS {
        return;
    }
    let trim_offset = raw.len() - raw.trim_start().len();
    let p_start = start + trim_offset;
    let p_end = p_start + trimmed.len();

    if trimmed.chars().count() <= MAX_PARAGRAPH_CHARS {
        out.push((p_start, p_end, trimmed));
    } else {
        let mut sub_start_byte = 0usize;
        let mut count = 0usize;
        for (b_idx, ch) in trimmed.char_indices() {
            count += 1;
            if count >= MAX_PARAGRAPH_CHARS && (ch == '.' || ch == '\n') {
                let next_b = b_idx + ch.len_utf8();
                let sub = trimmed[sub_start_byte..next_b].trim();
                if sub.chars().count() >= MIN_PARAGRAPH_CHARS {
                    out.push((p_start + sub_start_byte, p_start + next_b, sub));
                }
                sub_start_byte = next_b;
                count = 0;
            }
        }
        if sub_start_byte < trimmed.len() {
            let sub = trimmed[sub_start_byte..].trim();
            if sub.chars().count() >= MIN_PARAGRAPH_CHARS {
                out.push((p_start + sub_start_byte, p_end, sub));
            }
        }
    }
}

pub fn build_report(
    active_pii_count: usize,
    unmasked_pii_count: usize,
    secret_count: usize,
    unmasked_secret_count: usize,
    semantic_available: bool,
    paragraphs_scanned: usize,
    scan_duration_ms: u64,
    findings: Vec<ParagraphFinding>,
) -> ClassificationReport {
    let has_high_semantic = findings.iter().any(|f| f.severity == "high");
    let has_any_semantic = !findings.is_empty();

    // 1. Bedömning av ORIGINALTEXTEN (före maskering)
    let (verdict_before_mask, summary_before) = if secret_count > 0 || has_high_semantic {
        (
            "do_not_share".to_string(),
            "Stopp: Originaltexten innehåller känsligt/konfidentiellt innehåll eller tekniska nycklar och får inte klistras in i en öppen AI-tjänst.".to_string(),
        )
    } else if active_pii_count > 0 || unmasked_pii_count > 0 {
        (
            "mask_required".to_string(),
            format!(
                "Maskering krävs: Originaltexten innehåller {} identifierade personuppgifter.",
                active_pii_count + unmasked_pii_count
            ),
        )
    } else if has_any_semantic {
        (
            "caution_after_mask".to_string(),
            "Granska innehållet: Stycken i texten berör intern eller känslig kontext.".to_string(),
        )
    } else {
        (
            "safe".to_string(),
            if semantic_available {
                "Grön: Inga personuppgifter, hemligheter eller känslig kontext upptäcktes.".to_string()
            } else {
                "Inga direkta personuppgifter eller nycklar hittades (aktivera EmbeddingGemma 2 för djup kontextanalys).".to_string()
            },
        )
    };

    // 2. Bedömning av den MASKERADE texten (om användaren kopierar maskerad text till extern AI)
    let (verdict_after_mask, summary_after) = if unmasked_secret_count > 0 {
        (
            "do_not_share".to_string(),
            "Dela ej: Du har avmarkerat en teknisk hemlighet/nyckel eller intern länk — den maskerade texten är inte säker att dela!".to_string(),
        )
    } else if has_high_semantic {
        (
            "do_not_share".to_string(),
            format!(
                "Dela ej i öppen AI: Trots att identifierare har maskerats innehåller {} textavsnitt känslig kontext (t.ex. vård/hälsa, affärshemligheter, eID/tjänstekort eller intern IT-säkerhet).",
                findings.len()
            ),
        )
    } else if unmasked_pii_count > 0 {
        (
            "caution_after_mask".to_string(),
            format!(
                "Granska: {} träff(ar) har lämnats omaskerade i texten.",
                unmasked_pii_count
            ),
        )
    } else if has_any_semantic {
        (
            "caution_after_mask".to_string(),
            format!(
                "Granska: Personuppgifter är maskerade, men {} textavsnitt har intern karaktär — kontrollera innan delning i extern AI.",
                findings.len()
            ),
        )
    } else {
        (
            "safe".to_string(),
            if active_pii_count > 0 || secret_count > 0 {
                format!(
                    "Redo för AI: {} uppgift(er) maskerade och ingen känslig kontext flaggades.",
                    active_pii_count + secret_count
                )
            } else {
                "Redo för AI: Texten bedöms fri från personuppgifter och känslig kontext.".to_string()
            },
        )
    };

    ClassificationReport {
        verdict_before_mask,
        verdict_after_mask,
        summary_before,
        summary_after,
        active_pii_count,
        unmasked_pii_count,
        secret_count,
        semantic_available,
        paragraphs_scanned,
        scan_duration_ms,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrate_embeddinggemma2() {
        let Ok(home) = std::env::var("HOME") else {
            return;
        };
        let dir = Path::new(&home)
            .join("Library/Application Support/com.patriksvensk.maskera/models/embeddinggemma2");
        if !dir.join("model_q4.onnx_data").exists() {
            return;
        }

        let clf = EmbeddingClassifier::load(&dir).expect("load EmbeddingGemma 2");

        let sample = "\
Hej! Kan du hjälpa mig att formulera om den här inbjudan till vår öppna webbinariedag om hållbar stadsplanering nästa vecka? Vi bjuder på kaffe från kl 09:00 och alla intresserade medborgare är varmt välkomna att anmäla sig.\n\n\
[PERSON A] har varit sjukskriven på heltid sedan mars på grund av svår utmattningsdepression och panikångest, och behandlas för närvarande med sertralin vid psykiatriska öppenvårdsmottagningen.\n\n\
Inför styrelsemötet föreslår vi att lägga ett konfidentiellt bud om 145 MSEK på konkurrenten med en beräknad bruttomarginal på 42 procent innan kvartalsrapporten offentliggörs.\n\n\
Vid gårdagens säkerhetsgranskning upptäcktes en kritisk SQL-injektion i den interna autentiseringstjänsten bakom brandväggen som ger obehörig läsbehörighet till kunddatabasen.\n\n\
Tack för mötet igår [PERSON B]! Här är en kort sammanfattning av de publika riktlinjerna från Boverket som vi kan ha som underlag till den öppna presentationen nästa månad.";

        let (findings, scanned) = clf.scan_paragraphs_with_progress(sample, sample, |_, _| {});
        assert_eq!(scanned, 5);
        assert_eq!(findings.len(), 3, "Exakt de 3 känsliga styckena ska flaggas");
        assert_eq!(findings[0].paragraph_index, 2);
        assert_eq!(findings[0].axis, "integritet");
        assert_eq!(findings[0].severity, "high");
        assert_eq!(findings[1].paragraph_index, 3);
        assert_eq!(findings[1].axis, "affär");
        assert_eq!(findings[1].severity, "high");
        assert_eq!(findings[2].paragraph_index, 4);
        assert_eq!(findings[2].axis, "säkerhet");
        assert_eq!(findings[2].severity, "high");

        let siths_orig = "\
Hej!\n\n\
Ditt SITHS eID (certifikat) på ditt SITHS-kort nr.  9752267235710527661 med utgångsdatum 2027-01-04 kommer att upphöra om 60 dagar.\n\n\
Kortets giltighetstid står angiven på framsidan av kortet men SITHS eID (certifikatet) kan ha ett annat datum, se ovan.\n\n\
Har du SITHS reservkort eller SITHS konsultkort?\n\n\
Om du ska förnya/förlänga SITHS reservkort, konsultkort eller SITHS eID (certifikat) på någon av dessa två korttyper, kontakta din korthandläggare på din arbetsplats.\n\n\
Har du ett ordinarie SITHS-kort (med foto)?\n\n\
Då ska du förnya ditt ordinarie SITHS-kort (med foto) genom att boka tid till ett av regionens tre ID-fotokontor. Mer information och bokning hittar du på regionens intranät: https://intranat.rjl.se/service-och-stod-i-arbetet/lokaler-flytt-och-id/lokaler-flytt-och-id/bestalla-fota-och-hamta-ut-siths-kort/?pageId=49904&blockId=58111\n\n\
Observera att det tar cirka två veckor från att kortet har beställts av ID-fotokontoret tills du kan aktivera det.\n\n\
Boka tid så snart som möjligt för fotografering.\n\n\
Detta e-postmeddelande kan inte besvaras.\n\n\
Vänliga hälsningar\n\
SITHS-förvaltningen\n\
Region Jönköpings län";

        let siths = "\
Hej!\n\n\
Ditt SITHS eID (certifikat) på ditt SITHS-kort nr. [KORTNUMMER 1] med utgångsdatum 2027-01-04 kommer att upphöra om 60 dagar.\n\n\
Kortets giltighetstid står angiven på framsidan av kortet men SITHS eID (certifikatet) kan ha ett annat datum, se ovan.\n\n\
Har du SITHS reservkort eller SITHS konsultkort?\n\n\
Om du ska förnya/förlänga SITHS reservkort, konsultkort eller SITHS eID (certifikat) på någon av dessa två korttyper, kontakta din korthandläggare på din arbetsplats.\n\n\
Har du ett ordinarie SITHS-kort (med foto)?\n\n\
Då ska du förnya ditt ordinarie SITHS-kort (med foto) genom att boka tid till ett av regionens tre ID-fotokontor. Mer information och bokning hittar du på regionens intranät: [HEMLIGHET/NYCKEL 1]\n\n\
Observera att det tar cirka två veckor från att kortet har beställts av ID-fotokontoret tills du kan aktivera det.\n\n\
Boka tid så snart som möjligt för fotografering.\n\n\
Detta e-postmeddelande kan inte besvaras.\n\n\
Vänliga hälsningar\n\
[ORGANISATION A]\n\
[ORGANISATION B]";

        let (siths_findings, siths_scanned) =
            clf.scan_paragraphs_with_progress(siths_orig, siths, |_, _| {});
        println!("SITHS ihopslagna block: {siths_scanned}, flaggade: {}", siths_findings.len());
        for f in &siths_findings {
            let orig_slice = &siths_orig[f.start..f.end];
            assert!(!orig_slice.is_empty());
            println!(
                "  Block {}: {} ({}, {:.0}%) [{}..{}] => {}",
                f.paragraph_index,
                f.category,
                f.severity,
                f.confidence * 100.0,
                f.start,
                f.end,
                f.preview
            );
        }
        assert!(
            siths_findings
                .iter()
                .any(|f| siths_orig[f.start..f.end].contains("9752267235710527661")),
            "Original-offset ska peka på stycket med SITHS-kortnumret"
        );
        assert!(
            siths_findings
                .iter()
                .any(|f| siths_orig[f.start..f.end].contains("https://intranat.rjl.se/")),
            "Original-offset ska peka på stycket med intranätlänken"
        );
        assert!(
            !siths_findings.is_empty() && siths_findings.iter().any(|f| f.severity == "high"),
            "SITHS-meddelandet ska flaggas som hög känslighet under säkerhet/eID"
        );
    }
}
