use std::borrow::Cow;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::info_classification::{build_handling_guidance, InfoClassificationReport, InfoSignal};

pub const JUDGE_FILENAME: &str = "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
pub const JUDGE_HF_URL: &str =
    "https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
pub const JUDGE_EXPECTED_BYTES: u64 = 2_620_400_000;
pub const JUDGE_MODEL_LABEL: &str = "Gemma 4 E2B (Q4_K_XL · 2,62 GB)";

const JUDGE_N_CTX: u32 = 2048;
const JUDGE_N_BATCH: usize = 1024;
const JUDGE_N_UBATCH: u32 = 512;
const MAX_DOC_TOKENS: usize = 1350;
const MAX_GEN_TOKENS: usize = 96;
/// Förifylld start på JSON-svaret så att modellen direkt skriver dokumenttypens värde utan GBNF-overhead.
const PREFILL_JSON_PREFIX: &str = "{\"t\":\"";

static LLAMA_BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
static RE_INLINE_DATA_URI: OnceLock<Regex> = OnceLock::new();

fn ensure_llama_backend() -> Result<&'static LlamaBackend, String> {
    let res = LLAMA_BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init()
            .map_err(|e| format!("Kunde inte initiera lokal LLM-motor (llama.cpp): {e}"))?;
        backend.void_logs();
        Ok(backend)
    });
    match res {
        Ok(b) => Ok(b),
        Err(e) => Err(e.clone()),
    }
}

/// Beräknar trådantal separat för minnesbunden ordgenerering (`n_threads`, fysiska kärnor utan hyperthread-trängsel)
/// respektive beräkningsbunden prompt-inläsning (`n_threads_batch`).
pub fn optimal_cpu_threads() -> (i32, i32) {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let gen_threads = if logical <= 4 {
        logical.max(1)
    } else {
        (logical / 2).clamp(4, 6)
    };
    let batch_threads = if logical <= 4 {
        logical.max(1)
    } else {
        ((logical * 3) / 4).clamp(4, 8)
    };
    (gen_threads as i32, batch_threads as i32)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JudgeModelStatus {
    pub installed: bool,
    pub loaded: bool,
    pub file_size_bytes: u64,
    pub expected_bytes: u64,
    pub model_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JudgeAssessment {
    pub document_type: String,
    /// Nivån som Lager 4 (Judge) föreslog utifrån helheten: "green" | "yellow" | "red"
    pub suggested_level: String,
    /// Slutlig nivå efter eventuell säkerhetsvetorätt från Lager 1: "green" | "yellow" | "red"
    pub final_level: String,
    /// Nivån som Lager 1–3 gav före helhetsbedömningen
    pub layer3_level: String,
    /// Sant om slutnivån ändrades jämfört med Lager 3
    pub level_adjusted: bool,
    /// Sant om skarpa fynd i Lager 1 (t.ex. personnummer, bankkonto, kort eller lösenord) blockerade sänkning från Röd
    pub hard_veto_applied: bool,
    /// Kortfattad svensk motivering utifrån dokumentets helhet och syfte
    pub rationale: String,
    pub duration_ms: u64,
}

/// Kompakt intern JSON-struktur (`t`, `l`, `d`, `r`) som halverar antalet genererade strukturtokens,
/// men även accepterar de fullständiga fältnamnen bakåtkompatibelt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawJudgeJson {
    #[serde(rename = "t", alias = "documentType", alias = "type", default)]
    pub document_type: String,
    #[serde(rename = "l", alias = "suggestedLevel", alias = "level", default)]
    pub suggested_level: String,
    #[serde(rename = "d", alias = "downgradeRedSections", alias = "downgrade", default)]
    pub downgrade_red_sections: bool,
    #[serde(rename = "r", alias = "rationale", alias = "reason", default)]
    pub rationale: String,
}

pub struct JudgeEngine {
    /// Viktigt: `ctx` ligger i `Option` och töms alltid explicit i `Drop` INNAN `model` frigörs.
    ctx: Option<LlamaContext<'static>>,
    /// Heap-allokerad modell på stabil minnesadress under hela `JudgeEngine`s livstid.
    model: Box<LlamaModel>,
    /// Förinläst systemprompt i KV-cachen `[0 .. cached_prefix_len)` så att den aldrig räknas om mellan dokument.
    system_tokens: Vec<LlamaToken>,
    cached_prefix_len: usize,
}

impl Drop for JudgeEngine {
    fn drop(&mut self) {
        self.ctx = None;
    }
}

impl JudgeEngine {
    pub fn load(model_path: &Path) -> Result<Self, String> {
        if !model_path.exists() {
            return Err(format!(
                "Lager 4-modellen hittades inte på disk: {}",
                model_path.display()
            ));
        }

        let backend = ensure_llama_backend()?;
        let mut model_params = LlamaModelParams::default().with_use_mmap(true);

        // Använd Apple Silicon Metal automatiskt på macOS om tillgängligt; på Windows körs 100 % portabel CPU.
        #[cfg(target_os = "macos")]
        if backend.supports_gpu_offload() {
            model_params = model_params.with_n_gpu_layers(999);
        }

        let model = Box::new(
            LlamaModel::load_from_file(backend, model_path, &model_params)
                .map_err(|e| format!("Kunde inte ladda Lager 4-modellen (Gemma 4 E2B): {e}"))?,
        );

        let (gen_threads, batch_threads) = optimal_cpu_threads();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(JUDGE_N_CTX))
            .with_n_batch(JUDGE_N_BATCH as u32)
            .with_n_ubatch(JUDGE_N_UBATCH)
            .with_swa_full(true)
            .with_no_perf(true)
            .with_n_threads(gen_threads)
            .with_n_threads_batch(batch_threads);

        // SAFETY: `model` ägs i en `Box<LlamaModel>` på stabil heap-adress inne i `JudgeEngine`,
        // exponeras aldrig utåt som muterbar referens och `self.ctx` töms alltid explicit före `model` i `Drop`.
        let model_static_ref: &'static LlamaModel = unsafe { &*(&*model as *const LlamaModel) };
        let mut ctx = model_static_ref
            .new_context(backend, ctx_params)
            .map_err(|e| format!("Kunde inte skapa inferenskontext för Lager 4: {e}"))?;

        let system_prefix = build_judge_system_prefix();
        let system_tokens = model
            .vocab()
            .tokenize(system_prefix.as_bytes(), false, true);

        let mut cached_prefix_len = 0usize;
        if !system_tokens.is_empty() {
            decode_token_slice(&mut ctx, &system_tokens, 0, false)?;
            cached_prefix_len = system_tokens.len();
        }

        Ok(Self {
            ctx: Some(ctx),
            model,
            system_tokens,
            cached_prefix_len,
        })
    }

    pub fn assess_document(
        &mut self,
        text: &str,
        report: &InfoClassificationReport,
    ) -> Result<(RawJudgeJson, u64), String> {
        let t0 = Instant::now();
        let vocab = self.model.vocab();

        let user_turn = build_judge_user_turn(text, report);
        // `add_special = false` eftersom `<bos>` redan ligger i den cachade systemprompten,
        // `parse_special = true` så att `<|turn>`, `<turn|>`, `<|channel>`, `<channel|>` tolkas som kontrolltokens.
        let mut doc_tokens = vocab.tokenize(user_turn.as_bytes(), false, true);
        if doc_tokens.is_empty() {
            return Err("Kunde inte tokenisera underlaget för Lager 4.".to_string());
        }
        if doc_tokens.len() > MAX_DOC_TOKENS {
            doc_tokens.truncate(MAX_DOC_TOKENS);
        }

        let ctx = self
            .ctx
            .as_mut()
            .ok_or_else(|| "Inferenskontexten för Lager 4 är inte initierad.".to_string())?;

        // Rensa enbart dokumentdelen `[cached_prefix_len ..)` ur KV-cachen så att systemprompten återanvänds på 0 ms!
        let prefix_reused = self.cached_prefix_len > 0
            && matches!(
                ctx.clear_kv_cache_seq(Some(0), Some(self.cached_prefix_len as u32), None),
                Ok(true)
            );

        if !prefix_reused {
            ctx.clear_kv_cache();
            if !self.system_tokens.is_empty() {
                decode_token_slice(ctx, &self.system_tokens, 0, false)?;
                self.cached_prefix_len = self.system_tokens.len();
            } else {
                self.cached_prefix_len = 0;
            }
        }

        let last_batch_tokens =
            decode_token_slice(ctx, &doc_tokens, self.cached_prefix_len, true)?;

        // Ren `greedy()` utan GBNF eliminerar 262 144-vokabulärens strängallokeringar vid varje genererat ord.
        let mut sampler = LlamaSampler::greedy();
        let mut batch = LlamaBatch::new(1, 1);
        let mut gen_tokens = Vec::with_capacity(MAX_GEN_TOKENS);
        let mut cur_pos = (self.cached_prefix_len + doc_tokens.len()) as i32;
        let mut logits_idx = (last_batch_tokens as i32) - 1;

        for _ in 0..MAX_GEN_TOKENS {
            let next_token = sampler.sample(ctx, logits_idx);
            if vocab.is_eog(next_token) {
                break;
            }
            gen_tokens.push(next_token);

            let bytes = vocab.detokenize(&gen_tokens, true, false);
            let partial = String::from_utf8_lossy(&bytes);
            let combined = format!("{}{}", PREFILL_JSON_PREFIX, partial.trim_end());
            if combined.ends_with('}') {
                if let Ok(parsed) = parse_raw_judge_json(&combined) {
                    let elapsed = t0.elapsed().as_millis() as u64;
                    return Ok((parsed, elapsed));
                }
            }

            batch.clear();
            batch
                .add(next_token, cur_pos, &[0], true)
                .map_err(|e| format!("Batch-fel under generering i Lager 4: {e}"))?;
            cur_pos += 1;
            ctx.decode(&mut batch)
                .map_err(|e| format!("Inferensfel under generering i Lager 4: {e}"))?;
            logits_idx = 0;
        }

        let bytes = vocab.detokenize(&gen_tokens, true, false);
        let partial = String::from_utf8_lossy(&bytes);
        let full_raw = format!("{}{}", PREFILL_JSON_PREFIX, partial.trim());
        let parsed = parse_raw_judge_json(&full_raw)?;
        let elapsed = t0.elapsed().as_millis() as u64;
        Ok((parsed, elapsed))
    }
}

/// Läser in en sekvens av tokens i kontextens KV-cache i bitar om `JUDGE_N_BATCH`.
/// Returnerar antalet tokens i den sista batchen (för logit-indexering när `compute_last_logit = true`).
fn decode_token_slice(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start_pos: usize,
    compute_last_logit: bool,
) -> Result<usize, String> {
    let mut batch = LlamaBatch::new(JUDGE_N_BATCH, 1);
    let total = tokens.len();
    let mut last_chunk_len = 0usize;

    for chunk_start in (0..total).step_by(JUDGE_N_BATCH) {
        let chunk_end = (chunk_start + JUDGE_N_BATCH).min(total);
        last_chunk_len = chunk_end - chunk_start;
        batch.clear();
        for (offset, &tok) in tokens[chunk_start..chunk_end].iter().enumerate() {
            let rel = chunk_start + offset;
            let pos = (start_pos + rel) as i32;
            let want_logit = compute_last_logit && (rel + 1 == total);
            batch
                .add(tok, pos, &[0], want_logit)
                .map_err(|e| format!("Batch-fel vid inläsning i Lager 4: {e}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("Kunde inte tolka underlaget i Lager 4: {e}"))?;
    }

    Ok(last_chunk_len)
}

/// Säker teckenbaserad trunkering som aldrig klipper mitt i ett svenskt UTF-8-tecken (å/ä/ö/emoji).
pub fn truncate_chars(input: &str, max_chars: usize) -> String {
    let mut chars = input.chars();
    let taken: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}…", taken.trim_end())
    } else {
        taken
    }
}

/// Städar bort eventuella långa inbäddade base64-bilder (`data:image/...;base64,...`) ur Judge-utdraget
/// så att rå bildkod aldrig slösar kontexttokens om användaren klistrat in Markdown/HTML.
pub fn strip_inline_data_uris(input: &str) -> Cow<'_, str> {
    if !input.contains(";base64,") {
        return Cow::Borrowed(input);
    }
    let re = RE_INLINE_DATA_URI.get_or_init(|| {
        Regex::new(r"data:[a-zA-Z0-9/+.-]+;base64,[A-Za-z0-9+/=\s]{64,}")
            .expect("valid data URI regex")
    });
    re.replace_all(input, "[bild]")
}

/// Extraherar korta rubrikrader ur dokumentet så att Lager 4 ser hela dokumentets disposition.
fn extract_document_headings(text: &str, max_headings: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.contains(";base64,") {
            continue;
        }
        let char_len = trimmed.chars().count();
        let word_count = trimmed.split_whitespace().count();
        let ends_with_sentence_punct =
            trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?');
        let looks_like_heading = trimmed.starts_with('#')
            || (word_count <= 8
                && char_len >= 3
                && char_len <= 68
                && !ends_with_sentence_punct
                && trimmed
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase() || c.is_ascii_digit())
                    .unwrap_or(false));

        if looks_like_heading {
            let clean = trimmed.trim_start_matches('#').trim().to_string();
            if !clean.is_empty() && !out.contains(&clean) {
                out.push(clean);
                if out.len() >= max_headings {
                    break;
                }
            }
        }
    }
    out
}

/// Invariant systemprefix som förberäknas i KV-cachen vid modellstart (slutar vid kontrolltoken `<turn|>`
/// så att BPE-tokeniseringen mot efterföljande användartur alltid är exakt på tokengränsen).
pub fn build_judge_system_prefix() -> String {
    let system_prompt = "\
Du är Lager 4 (Judge) för svensk informationsklassning (green, yellow, red) i offentlig verksamhet.\n\
Lager 3 granskar bara lösryckta stycken och kan feltolka kursreferat, utbildning, metoder eller allmänna rutiner/utkast som RED pga enstaka ord (t.ex. sekretess, samtal, upphandling, patientdatalagen).\n\
Bedöm HELA DOKUMENTETS SYFTE:\n\
- \"green\": Offentlig information för öppen spridning (broschyr, pressmeddelande, öppen info utan interna noteringar/personkoppling).\n\
- \"yellow\": Intern arbetsinformation (utkast, intern rutin/lathund, kurs-/seminariereferat, APT-/mötesanteckning, schema, interna arbetskontakter/intranät).\n\
- \"red\": Verkligt känsligt individärende (enskild patients/elevs vård/hälsa/utredning, enskild medarbetares rehab/sjukfrånvaro/disciplin/lön), skarp anbuds-/besluts-/säkerhetssekretess eller personnummer/konto/lösenord.\n\
 Om Lager 3 gav RED på generell rutin/utbildning/referat utan verkligt individärende: sätt \"l\":\"yellow\" (eller \"green\" om helt publikt) och \"d\":true.\n\
Svara ENDAST på en rad JSON utan citattecken inne i textvärden: {\"t\":\"<dokumenttyp 2-4 ord på svenska>\",\"l\":\"green|yellow|red\",\"d\":true|false,\"r\":\"<1 kort mening på svenska, max 22 ord>\"}";

    format!("<bos><|turn>system\n{system_prompt}<turn|>")
}

/// Bygger den kompakta dokumentturen som läggs direkt efter det cachade systemprefixet.
/// Prioriterar alltid eventuella RÖDA avsnitt från Lager 3 främst så att Judge alltid ser vad som triggade Röd.
pub fn build_judge_user_turn(text: &str, report: &InfoClassificationReport) -> String {
    let cleaned = strip_inline_data_uris(text);
    let trimmed_doc = cleaned.trim();
    let total_chars = trimmed_doc.chars().count();
    let opening = truncate_chars(trimmed_doc, 460);
    let closing = if total_chars > 750 {
        let tail_chars: Vec<char> = trimmed_doc.chars().collect();
        let start_idx = tail_chars.len().saturating_sub(220);
        let tail: String = tail_chars[start_idx..].iter().collect();
        format!("…{}", tail.trim_start())
    } else {
        String::new()
    };

    let headings = extract_document_headings(trimmed_doc, 8);
    let headings_line = if headings.is_empty() {
        "(inga tydliga rubriker)".to_string()
    } else {
        headings.join(" | ")
    };

    let structural_signals: Vec<String> = report
        .signals
        .iter()
        .filter(|s| s.source == "struktur")
        .map(|s| format!("[{}] {}", s.level.to_uppercase(), s.title))
        .collect();
    let structural_summary = if structural_signals.is_empty() {
        "Inga (inga pnr/konton/nycklar/personkontakter)".to_string()
    } else {
        structural_signals.join("; ")
    };

    // Sortera så att RÖDA stycken alltid visas före GULA och högst konfidens kommer först (max 4 utdrag).
    let mut prioritized_findings: Vec<_> = report.section_findings.iter().collect();
    prioritized_findings.sort_by(|a, b| {
        let rank_a = if a.level == "red" { 0u8 } else { 1u8 };
        let rank_b = if b.level == "red" { 0u8 } else { 1u8 };
        rank_a.cmp(&rank_b).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut flagged_lines = Vec::new();
    for f in prioritized_findings.into_iter().take(4) {
        let raw_snippet = if f.end <= text.len()
            && f.start <= f.end
            && text.is_char_boundary(f.start)
            && text.is_char_boundary(f.end)
        {
            strip_inline_data_uris(text[f.start..f.end].trim()).into_owned()
        } else {
            strip_inline_data_uris(&f.preview).into_owned()
        };
        let snippet = truncate_chars(&raw_snippet, 180);
        flagged_lines.push(format!(
            "- #{p} [{lvl} | {cat}]: {snip}",
            p = f.paragraph_index,
            lvl = f.level.to_uppercase(),
            cat = truncate_chars(&f.category, 42),
            snip = snippet.replace('\n', " ")
        ));
    }
    let flagged_summary = if flagged_lines.is_empty() {
        "- Inga flaggade avsnitt.".to_string()
    } else {
        flagged_lines.join("\n")
    };

    let mut user_prompt = format!(
        "Längd: {} tecken ({} stycken) | Lager 1–3: {}\n\
Lager 1–2: {}\n\
Rubriker: {}\n\
Inledning: {}\n",
        total_chars,
        report.paragraphs_scanned,
        report.level.to_uppercase(),
        structural_summary,
        headings_line,
        opening
    );

    if !closing.is_empty() {
        user_prompt.push_str(&format!("Slut: {}\n", closing));
    }

    user_prompt.push_str(&format!("Flaggade avsnitt:\n{}", flagged_summary));

    format!(
        "\n<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{PREFILL_JSON_PREFIX}"
    )
}

/// Bygger hela kaskadprompten (systemprefix + dokumenttur) för inspektion och enhetstester.
#[cfg(test)]
pub fn build_judge_prompt(text: &str, report: &InfoClassificationReport) -> String {
    format!(
        "{}{}",
        build_judge_system_prefix(),
        build_judge_user_turn(text, report)
    )
}

/// Tolkar JSON-svaret robust — stöder både kompakta nycklar (`t`,`l`,`d`,`r`), långa nycklar
/// och fallback-utvinning om modellen råkat skriva oescapade citattecken inne i den svenska motiveringen.
pub fn parse_raw_judge_json(raw: &str) -> Result<RawJudgeJson, String> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<RawJudgeJson>(trimmed) {
        if !parsed.suggested_level.is_empty() {
            return Ok(sanitize_raw_judge(parsed));
        }
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            let slice = &trimmed[start..=end];
            if let Ok(parsed) = serde_json::from_str::<RawJudgeJson>(slice) {
                if !parsed.suggested_level.is_empty() {
                    return Ok(sanitize_raw_judge(parsed));
                }
            }
            if let Some(fallback) = salvage_loose_judge_json(slice) {
                return Ok(sanitize_raw_judge(fallback));
            }
        }
    }

    Err(format!(
        "Kunde inte tolka JSON-svaret från Lager 4: {}",
        truncate_chars(trimmed, 180)
    ))
}

/// Räddar ett JSON-svar om den svenska fritexten i `r`/`rationale` råkat innehålla oescapade `"`-tecken.
fn salvage_loose_judge_json(slice: &str) -> Option<RawJudgeJson> {
    let lower = slice.to_ascii_lowercase();
    let suggested_level = if lower.contains("\"l\":\"green\"")
        || lower.contains("\"l\": \"green\"")
        || lower.contains("\"suggestedlevel\":\"green\"")
    {
        "green"
    } else if lower.contains("\"l\":\"red\"")
        || lower.contains("\"l\": \"red\"")
        || lower.contains("\"suggestedlevel\":\"red\"")
    {
        "red"
    } else if lower.contains("\"l\":\"yellow\"")
        || lower.contains("\"l\": \"yellow\"")
        || lower.contains("\"suggestedlevel\":\"yellow\"")
    {
        "yellow"
    } else {
        return None;
    };

    let downgrade_red_sections =
        lower.contains("\"d\":true") || lower.contains("\"d\": true") || lower.contains("\"downgraderedsections\":true");

    let document_type = extract_between_markers(slice, "\"t\":\"", "\",\"")
        .or_else(|| extract_between_markers(slice, "\"documentType\":\"", "\",\""))
        .unwrap_or_else(|| "Granskat dokument".to_string())
        .replace('"', "'");

    let rationale = extract_rationale_tail(slice, "\"r\":\"")
        .or_else(|| extract_rationale_tail(slice, "\"rationale\":\""))
        .unwrap_or_default()
        .replace('"', "'");

    Some(RawJudgeJson {
        document_type,
        suggested_level: suggested_level.to_string(),
        downgrade_red_sections,
        rationale,
    })
}

fn extract_between_markers(s: &str, start_pat: &str, end_pat: &str) -> Option<String> {
    let start = s.find(start_pat)? + start_pat.len();
    let rest = &s[start..];
    let end = rest.find(end_pat)?;
    Some(rest[..end].trim().to_string())
}

fn extract_rationale_tail(s: &str, key_pat: &str) -> Option<String> {
    let start = s.find(key_pat)? + key_pat.len();
    let rest = s[start..].trim_end();
    let stripped = rest
        .strip_suffix("\"}")
        .or_else(|| rest.strip_suffix('}'))?;
    Some(stripped.trim_end_matches('"').trim().to_string())
}

fn sanitize_raw_judge(mut parsed: RawJudgeJson) -> RawJudgeJson {
    let norm = parsed.suggested_level.trim().to_ascii_lowercase();
    parsed.suggested_level = match norm.as_str() {
        "green" | "grön" | "gron" => "green".to_string(),
        "yellow" | "gul" => "yellow".to_string(),
        "red" | "röd" | "rod" => "red".to_string(),
        _ => "yellow".to_string(),
    };
    parsed.document_type = truncate_chars(parsed.document_type.trim(), 70);
    if parsed.document_type.is_empty() {
        parsed.document_type = "Granskat dokument".to_string();
    }
    parsed.rationale = truncate_chars(parsed.rationale.trim(), 420);
    if parsed.rationale.is_empty() {
        parsed.rationale =
            "Helhetsbedömning genomförd utifrån dokumentets övergripande struktur och sammanhang."
                .to_string();
    }
    parsed
}

fn level_swedish_label(level: &str) -> &'static str {
    match level {
        "red" => "RÖD",
        "yellow" => "GUL",
        _ => "GRÖN",
    }
}

/// Applicerar Lager 4:s helhetsbedömning på rapporten från Lager 1–3, inklusive säkerhetsvetorätt för Lager 1.
pub fn apply_judge_to_report(
    report: &mut InfoClassificationReport,
    raw: RawJudgeJson,
    duration_ms: u64,
) {
    let layer3_level = report.level.clone();

    // Skarpa Lager 1-ankare (personnummer, privat bankkonto/IBAN, kortnummer, lösenord/hemlig nyckel) har ALLTID vetorätt för RÖD!
    let has_hard_red_anchor = report
        .signals
        .iter()
        .any(|s| s.level == "red" && s.source == "struktur");

    // Intranätlänkar är per definition aldrig Grön/offentlig information — lägst Gul.
    let has_internal_url_anchor = report
        .signals
        .iter()
        .any(|s| s.source == "struktur" && s.title.contains("intranät"));

    let mut hard_veto_applied = false;
    let final_level = if has_hard_red_anchor && raw.suggested_level != "red" {
        hard_veto_applied = true;
        "red".to_string()
    } else if has_internal_url_anchor && raw.suggested_level == "green" {
        "yellow".to_string()
    } else {
        raw.suggested_level.clone()
    };

    let level_adjusted = final_level != layer3_level;

    // Om Lager 4 sänker RÖD -> GUL/GRÖN (och inga skarpa Lager 1-ankare finns), nedgradera falska röda styckesträffar
    if final_level != "red" && (layer3_level == "red" || raw.downgrade_red_sections) {
        let downgraded_target = if final_level == "green" {
            "yellow"
        } else {
            "yellow"
        };

        if final_level == "green" {
            // Vid helt grön helhetsbedömning tas falska semantiska utslag bort så att rapporten inte motsäger sig själv
            report.section_findings.clear();
            report.signals.retain(|s| s.source == "struktur");
        } else {
            for f in &mut report.section_findings {
                if f.level == "red" {
                    f.level = downgraded_target.to_string();
                    if !f.category.starts_with("Helhetsbedömd som Gul") {
                        f.category = format!("Helhetsbedömd som Gul — {}", f.category);
                    }
                    f.reason = format!(
                        "Lager 3 flaggade ordvalet i detta stycke som rött, men Lager 4 (Judge) bedömde utifrån dokumentets helhet ({}) att avsnittet är intern/generell information: {}",
                        raw.document_type, raw.rationale
                    );
                }
            }
            // Uppdatera eventuella röda kontextsignaler till gula
            for s in &mut report.signals {
                if s.level == "red" && s.source == "kontext" {
                    s.level = "yellow".to_string();
                    s.title = format!(
                        "Överprövad av Lager 4 (Judge): styckena bedöms som GUL ({})",
                        raw.document_type
                    );
                    s.detail = raw.rationale.clone();
                }
            }
        }
    }

    // Lägg till en tydlig sammanfattande signal från Lager 4 i signallistan om den inte redan skrevs över ovan
    let already_has_judge_signal = report
        .signals
        .iter()
        .any(|s| s.title.contains("Lager 4 (Judge)"));

    if !already_has_judge_signal {
        // Ta bort eventuell platshållarsignal för "Inga röda eller gula markörer" om Lager 4 justerade upp nivån
        if final_level != "green" {
            report.signals.retain(|s| s.level != "green");
        }
        let judge_sig_title = if hard_veto_applied {
            format!(
                "Lager 4 (Judge · {}): Behåller RÖD p.g.a. skarpa strukturfynd i Lager 1",
                raw.document_type
            )
        } else if level_adjusted {
            format!(
                "Lager 4 (Judge · {}): Justerade {} → {} utifrån helheten",
                raw.document_type,
                level_swedish_label(&layer3_level),
                level_swedish_label(&final_level)
            )
        } else {
            format!(
                "Lager 4 (Judge · {}): Bekräftar {} nivå vid helhetsgranskning",
                raw.document_type,
                level_swedish_label(&final_level)
            )
        };

        let judge_sig_detail = if hard_veto_applied {
            format!(
                "Helhetskontexten liknar '{}' ({}), men dokumentet innehåller skarpa identifierare i Lager 1 (personnummer/bankkonto/kort/nyckel) som alltid kräver RÖD hantering.",
                raw.document_type, raw.rationale
            )
        } else {
            raw.rationale.clone()
        };

        report.signals.insert(
            0,
            InfoSignal {
                level: final_level.clone(),
                source: "kontext".to_string(),
                title: judge_sig_title,
                detail: judge_sig_detail,
            },
        );
    }

    // Uppdatera toppnivå, rubriker och hanteringsråd så att kort 1 och kort 2 matchar slutnivån
    report.level = final_level.clone();
    match final_level.as_str() {
        "red" => {
            report.badge_title = "Förslag: RÖD — Kräver särskilt skydd".to_string();
            report.content_headline = if level_adjusted {
                "Helhetsbedömning (Lager 4): Indikerar RÖD information".to_string()
            } else {
                "Maskinell bedömning: Indikerar RÖD information".to_string()
            };
            report.content_explanation = if hard_veto_applied {
                "Dokumentet innehåller skarpa identifierare från Lager 1 (t.ex. personnummer, bankkonto, kortnummer eller nyckel) och behålls därför som RÖD oavsett övergripande dokumenttyp.".to_string()
            } else {
                format!(
                    "Vid granskning av både enskilda avsnitt och dokumentets helhet ({}) indikeras sekretessbelagd eller särskilt skyddsvärd information.",
                    raw.document_type
                )
            };
        }
        "yellow" => {
            report.badge_title = "Förslag: GUL — Intern försiktighet".to_string();
            report.content_headline = if level_adjusted {
                "Helhetsbedömning (Lager 4): Justerad till GUL (intern) information".to_string()
            } else {
                "Maskinell bedömning: Indikerar GUL (intern) information".to_string()
            };
            report.content_explanation = if level_adjusted && layer3_level == "red" {
                format!(
                    "Lager 3 reagerade på enskilda ord i styckena, men Lager 4 (Judge) bedömde dokumentet i sin helhet som '{}' utan enskilda sekretessärenden eller skarpa personidentifierare.",
                    raw.document_type
                )
            } else {
                format!(
                    "Helhetsanalysen ({}) tyder på internt arbetsmaterial, mötes-/kursanteckningar, utkast eller opublicerade kontaktuppgifter för behörig krets.",
                    raw.document_type
                )
            };
        }
        _ => {
            report.badge_title = "Förslag: GRÖN — Kan delas öppet".to_string();
            report.content_headline = if level_adjusted {
                "Helhetsbedömning (Lager 4): Justerad till GRÖN information".to_string()
            } else {
                "Maskinell bedömning: Indikerar GRÖN information".to_string()
            };
            report.content_explanation = format!(
                "Helhetsanalysen ({}) tyder på offentlig eller allmän information utan identifierade interna eller särskilt skyddsvärda uppgifter (förutsatt att materialet är klargjort för publicering).",
                raw.document_type
            );
        }
    }

    let mut handling = build_handling_guidance(&final_level);
    handling.disclaimer = "Detta är ett lokalt maskinellt beslutsstöd (Lager 1–3 + helhetsbedömning med Lager 4 / Judge) — inte en fastställd sanning. Även en språkmodell som läser helheten kan missförstå lokal kontext eller verksamhetsspecifika regler. Det är alltid du som granskar och fattar det slutliga beslutet.".to_string();
    report.handling = handling;

    let rationale_for_ui = if hard_veto_applied {
        format!(
            "{} (Obs: Slutnivån hålls kvar på RÖD eftersom Lager 1 hittade skarpa identifierare i texten.)",
            raw.rationale
        )
    } else {
        raw.rationale
    };

    report.judge = Some(JudgeAssessment {
        document_type: raw.document_type,
        suggested_level: raw.suggested_level,
        final_level,
        layer3_level,
        level_adjusted,
        hard_veto_applied,
        rationale: rationale_for_ui,
        duration_ms,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info_classification::{InfoSectionFinding, InfoSignal};

    fn sample_red_layer3_report(with_hard_pnr: bool) -> InfoClassificationReport {
        let mut signals = Vec::new();
        if with_hard_pnr {
            signals.push(InfoSignal {
                level: "red".to_string(),
                source: "struktur".to_string(),
                title: "1 personnummer / samordningsnummer".to_string(),
                detail: "Särskilt skyddsvärd personidentitet.".to_string(),
            });
        }
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "kontext".to_string(),
            title: "2 avsnitt med indikerat rött innehåll (Upphandlingssekretess)".to_string(),
            detail: "Berör pågående upphandling.".to_string(),
        });

        InfoClassificationReport {
            level: "red".to_string(),
            badge_title: "Förslag: RÖD — Kräver särskilt skydd".to_string(),
            content_headline: "Maskinell bedömning: Indikerar RÖD information".to_string(),
            content_explanation: "Preliminär text.".to_string(),
            signals,
            handling: build_handling_guidance("red"),
            section_findings: vec![InfoSectionFinding {
                paragraph_index: 6,
                preview: "Större hybridmöten typ 100 medverkande där man anlitar extern kompetens…".to_string(),
                start: 0,
                end: 72,
                category: "Upphandlingssekretess, Affärshemligheter & Ej offentliggjord politik".to_string(),
                level: "red".to_string(),
                confidence: 0.73,
                reason: "Berör pågående upphandling.".to_string(),
            }],
            paragraphs_scanned: 25,
            scan_duration_ms: 420,
            semantic_available: true,
            judge: None,
        }
    }

    #[test]
    fn judge_downgrades_false_red_seminar_summary_when_no_layer1_anchor() {
        let mut report = sample_red_layer3_report(false);
        let raw = parse_raw_judge_json(
            r#"{"documentType":"Internt seminariereferat","suggestedLevel":"yellow","downgradeRedSections":true,"rationale":"Dokumentet är i sin helhet en kollegas sammanfattning från ett seminarium om hybridmöten och innehåller inga verkliga personal- eller upphandlingsärenden."}"#,
        )
        .expect("valid JSON");

        apply_judge_to_report(&mut report, raw, 3450);

        assert_eq!(report.level, "yellow");
        assert!(report.section_findings.iter().all(|f| f.level == "yellow"));
        assert!(report.signals.iter().all(|s| s.level != "red"));
        let judge = report.judge.as_ref().expect("judge metadata attached");
        assert_eq!(judge.layer3_level, "red");
        assert_eq!(judge.final_level, "yellow");
        assert!(judge.level_adjusted);
        assert!(!judge.hard_veto_applied);
    }

    #[test]
    fn layer1_hard_anchor_vetoes_judge_downgrade() {
        let mut report = sample_red_layer3_report(true);
        let raw = parse_raw_judge_json(
            r#"{"documentType":"Mötesanteckning","suggestedLevel":"yellow","downgradeRedSections":true,"rationale":"Ser ut som allmänna mötesanteckningar."}"#,
        )
        .expect("valid JSON");

        apply_judge_to_report(&mut report, raw, 2100);

        assert_eq!(report.level, "red", "Personnummer i Lager 1 ska alltid behålla RÖD nivå");
        let judge = report.judge.as_ref().expect("judge metadata attached");
        assert_eq!(judge.suggested_level, "yellow");
        assert_eq!(judge.final_level, "red");
        assert!(!judge.level_adjusted);
        assert!(judge.hard_veto_applied);
    }

    #[test]
    fn truncate_chars_handles_swedish_multibyte_safely() {
        let s = "Årsredovisning för Östgötatrafiken — ärendehantering och mötesmetodik";
        let t = truncate_chars(s, 22);
        assert!(t.ends_with('…'));
        assert!(t.starts_with("Årsredovisning för Öst"));
    }

    #[test]
    fn parses_compact_prefilled_json_and_unescaped_quotes_in_rationale() {
        let model_tail = r#"Intern samordningsrutin","l":"yellow","d":true,"r":"Utkast till rutin som nämner "sekretess" generellt utan enskilda patientuppgifter."}"#;
        let combined = format!("{}{}", PREFILL_JSON_PREFIX, model_tail);
        let parsed = parse_raw_judge_json(&combined).expect("salvages unescaped Swedish quotes");
        assert_eq!(parsed.document_type, "Intern samordningsrutin");
        assert_eq!(parsed.suggested_level, "yellow");
        assert!(parsed.downgrade_red_sections);
        assert!(parsed.rationale.contains("'sekretess'"));
    }

    #[test]
    fn strips_inline_base64_images_and_prioritizes_red_sections_first() {
        let mut report = sample_red_layer3_report(false);
        report.section_findings.clear();
        for idx in 1..=5 {
            report.section_findings.push(InfoSectionFinding {
                paragraph_index: idx,
                preview: format!("Gult arbetsavsnitt nummer {idx}"),
                start: 0,
                end: 0,
                category: "APT- & Mötesanteckningar".to_string(),
                level: "yellow".to_string(),
                confidence: 0.80,
                reason: "Intern notering.".to_string(),
            });
        }
        report.section_findings.push(InfoSectionFinding {
            paragraph_index: 9,
            preview: "Om sekretessbelagd information lämnas dokumenteras den enligt patientdatalagen.".to_string(),
            start: 0,
            end: 0,
            category: "Vård, Hälsa & Elevhälsa".to_string(),
            level: "red".to_string(),
            confidence: 0.71,
            reason: "Lager 3 reagerade på patientdatalagen.".to_string(),
        });

        let fake_b64 = "A".repeat(200);
        let doc = format!(
            "Rutin Digital mötesform\n\n![](data:image/png;base64,{fake_b64})\n\nGemensam rutin."
        );
        let turn = build_judge_user_turn(&doc, &report);
        let full = build_judge_prompt(&doc, &report);

        assert!(full.starts_with("<bos><|turn>system\n"));
        assert!(!turn.contains(&fake_b64), "base64-bilddata ska rensas bort ur Judge-prompten");
        assert!(turn.contains("[bild]"));
        assert!(
            turn.contains("#9 [RED"),
            "det röda avsnittet (#9) måste alltid prioriteras före gula avsnitt"
        );
        assert!(turn.ends_with(PREFILL_JSON_PREFIX));
    }
}
