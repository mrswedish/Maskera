use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};

use crate::info_classification::{build_handling_guidance, InfoClassificationReport, InfoSignal};

pub const JUDGE_FILENAME: &str = "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
pub const JUDGE_HF_URL: &str =
    "https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
pub const JUDGE_EXPECTED_BYTES: u64 = 2_620_400_000;
pub const JUDGE_MODEL_LABEL: &str = "Gemma 4 E2B (Q4_K_XL · 2,62 GB)";

static LLAMA_BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

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
    /// Kortfattad svensk motivering (1–3 meningar) utifrån dokumentets helhet och syfte
    pub rationale: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawJudgeJson {
    pub document_type: String,
    pub suggested_level: String,
    #[serde(default)]
    pub downgrade_red_sections: bool,
    pub rationale: String,
}

/// Strikt GBNF-grammatik som tvingar Gemma 4 E2B att direkt generera ett giltigt, kompakt JSON-objekt.
const JUDGE_JSON_GBNF: &str = r#"root ::= "{" "\"documentType\":" string ",\"suggestedLevel\":" level ",\"downgradeRedSections\":" boolean ",\"rationale\":" string "}"
level ::= "\"green\"" | "\"yellow\"" | "\"red\""
boolean ::= "true" | "false"
string ::= "\"" char+ "\""
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
"#;

pub struct JudgeEngine {
    model: LlamaModel,
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

        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| format!("Kunde inte ladda Lager 4-modellen (Gemma 4 E2B): {e}"))?;

        Ok(Self { model })
    }

    pub fn assess_document(
        &mut self,
        text: &str,
        report: &InfoClassificationReport,
    ) -> Result<(RawJudgeJson, u64), String> {
        let t0 = Instant::now();
        let backend = ensure_llama_backend()?;

        let prompt = build_judge_prompt(text, report);
        let vocab = self.model.vocab();
        // `add_special = false` eftersom `<bos>` ingår explicit i Gemma 4-mallen,
        // `parse_special = true` så att `<bos>`, `<|turn>`, `<turn|>`, `<|channel>`, `<channel|>` tolkas som kontrolltokens.
        let mut prompt_tokens = vocab.tokenize(prompt.as_bytes(), false, true);
        if prompt_tokens.is_empty() {
            return Err("Kunde inte tokenisera underlaget för Lager 4.".to_string());
        }

        // Håll kontextfönstret kompakt (max ~3 500 prompttokens + ~256 svarsmarginal = 4 096) för låg RAM-förbrukning och snabb CPU-körning.
        const MAX_PROMPT_TOKENS: usize = 3500;
        if prompt_tokens.len() > MAX_PROMPT_TOKENS {
            prompt_tokens.truncate(MAX_PROMPT_TOKENS);
        }

        let n_ctx: u32 = 4096;
        let n_batch: usize = 1024;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 8) as i32;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch as u32)
            .with_n_ubatch(512)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);

        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| format!("Kunde inte skapa inferenskontext för Lager 4: {e}"))?;

        let mut batch = LlamaBatch::new(n_batch, 1);
        let total_prompt = prompt_tokens.len();

        for chunk_start in (0..total_prompt).step_by(n_batch) {
            let chunk_end = (chunk_start + n_batch).min(total_prompt);
            batch.clear();
            for (offset, &tok) in prompt_tokens[chunk_start..chunk_end].iter().enumerate() {
                let pos = chunk_start + offset;
                let is_last = pos + 1 == total_prompt;
                batch
                    .add(tok, pos as i32, &[0], is_last)
                    .map_err(|e| format!("Batch-fel vid inläsning av prompt i Lager 4: {e}"))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| format!("Kunde inte tolka dokumentunderlaget i Lager 4: {e}"))?;
        }

        let mut sampler = match LlamaSampler::grammar(&self.model, JUDGE_JSON_GBNF, "root") {
            Ok(grammar) => LlamaSampler::chain_simple([grammar, LlamaSampler::greedy()]),
            Err(_) => LlamaSampler::greedy(),
        };

        let mut gen_tokens = Vec::with_capacity(180);
        let mut cur_pos = total_prompt as i32;
        const MAX_GEN_TOKENS: usize = 180;

        for _ in 0..MAX_GEN_TOKENS {
            // Obs: `sampler.sample` anropar redan `llama_sampler_accept` internt i llama.cpp.
            let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if vocab.is_eog(next_token) {
                break;
            }
            gen_tokens.push(next_token);

            let bytes = vocab.detokenize(&gen_tokens, true, false);
            let partial = String::from_utf8_lossy(&bytes);
            let trimmed = partial.trim();
            if trimmed.ends_with('}') {
                if let Ok(parsed) = parse_raw_judge_json(trimmed) {
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
        }

        let bytes = vocab.detokenize(&gen_tokens, true, false);
        let raw_output = String::from_utf8_lossy(&bytes);
        let parsed = parse_raw_judge_json(&raw_output)?;
        let elapsed = t0.elapsed().as_millis() as u64;
        Ok((parsed, elapsed))
    }
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

/// Extraherar korta rubrikrader ur dokumentet så att Lager 4 ser hela dokumentets disposition.
fn extract_document_headings(text: &str, max_headings: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
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

/// Bygger kaskadprompten (Gemma 4 non-thinking format med förifylld tom `<|channel>thought\n<channel|>`).
pub fn build_judge_prompt(text: &str, report: &InfoClassificationReport) -> String {
    let total_chars = text.chars().count();
    let opening = truncate_chars(text.trim(), 750);
    let closing = if total_chars > 1100 {
        let tail_chars: Vec<char> = text.trim().chars().collect();
        let start_idx = tail_chars.len().saturating_sub(420);
        let tail: String = tail_chars[start_idx..].iter().collect();
        format!("…{}", tail.trim_start())
    } else {
        String::new()
    };

    let headings = extract_document_headings(text, 10);
    let headings_line = if headings.is_empty() {
        "(inga tydliga mellanrubriker)".to_string()
    } else {
        headings.join(" | ")
    };

    let structural_signals: Vec<String> = report
        .signals
        .iter()
        .filter(|s| s.source == "struktur")
        .map(|s| format!("- [{}] {}", s.level.to_uppercase(), s.title))
        .collect();
    let structural_summary = if structural_signals.is_empty() {
        "- Inga personnummer, bankkonton, kortnummer, hemliga nycklar, intranätlänkar eller personkontaktuppgifter hittades av Lager 1–2.".to_string()
    } else {
        structural_signals.join("\n")
    };

    let mut flagged_lines = Vec::new();
    for f in report.section_findings.iter().take(6) {
        let snippet = if f.end <= text.len()
            && f.start <= f.end
            && text.is_char_boundary(f.start)
            && text.is_char_boundary(f.end)
        {
            truncate_chars(text[f.start..f.end].trim(), 240)
        } else {
            truncate_chars(&f.preview, 200)
        };
        flagged_lines.push(format!(
            "- Avsnitt {} [Lager 3: {} | {}]: \"{}\"",
            f.paragraph_index,
            f.level.to_uppercase(),
            f.category,
            snippet.replace('\n', " ")
        ));
    }
    let flagged_summary = if flagged_lines.is_empty() {
        "- Inga enskilda sakavsnitt flaggades av Lager 3.".to_string()
    } else {
        flagged_lines.join("\n")
    };

    let system_prompt = "\
Du är Lager 4 (Judge) i Maskera — ett lokalt svenskt beslutsstöd för informationsklassificering enligt trafikljusmodellen (green, yellow, red) i offentlig verksamhet.\n\
Lager 3 bedömer enskilda textstycken med vektorlikhet och kan därför feltolka lösryckta ord (t.ex. klassa ett kursreferat, utbildningsmaterial eller en metodguide som RED bara för att ord som samtal, ledare, extern kompetens, upphandling eller säkerhet nämns generellt).\n\
Din uppgift är att bedöma DOKUMENTETS HELHET OCH SYFTE:\n\
- \"green\": Offentlig information avsedd för öppen spridning (t.ex. publicerad broschyr, tidtabell, pressmeddelande, öppen inbjudan eller allmän metodtext utan interna anteckningar/personkopplingar).\n\
- \"yellow\": Intern arbetsinformation inom verksamheten/arbetsgruppen (t.ex. kurs- eller seminariereferat från kollega, APT-/mötesanteckningar, utkast, interna rutiner/lathundar, schema/bemanning, vanligt medarbetarsamtal om mål/trivsel eller dokument med interna arbetskontakter/intranätlänkar).\n\
- \"red\": Särskilt skyddsvärd eller sekretessbelagd information om verkliga fall/hemligheter (t.ex. enskild patients/elevs hälsa eller utredning, faktiskt personal-/rehab-/disciplinärende eller individuell lön, pågående anbudssekretess före tilldelning, ej offentliggjort politiskt beslutsunderlag, krisberedskap, lås/larmkoder/lösenord, personnummer eller privat bankkonto).\n\
Regler:\n\
1. Om Lager 3 flaggade RED men helheten visar att texten är ett kursreferat, seminarium, utbildningsmaterial, metodtips eller allmänt internt arbetsutkast utan verkligt individärende eller skarp sekretess: sätt suggestedLevel till \"yellow\" (eller \"green\" om helt publikt) och downgradeRedSections till true.\n\
2. Om dokumentet trots mjuk ton faktiskt beskriver en enskild individs sjukdom/rehab/misskötsamhet/lön eller skarp verksamhetssekretess: sätt suggestedLevel till \"red\" och downgradeRedSections till false.\n\
3. Skriv documentType kort på svenska (2–5 ord, t.ex. \"Internt seminariereferat\", \"Offentlig informationsbroschyr\", \"Känsligt personal-/rehabärende\") och rationale på klar svenska i 1–2 korta meningar.";

    let mut user_prompt = format!(
        "GRANSKA DOKUMENTETS HELHET:\n\
Längd: {} tecken ({} granskade stycken)\n\
Lager 1–3 preliminär nivå: {}\n\n\
STRUKTURELLA FYND (Lager 1–2):\n{}\n\n\
RUBRIKER / DISPOSITION:\n{}\n\n\
DOKUMENTETS INLEDNING:\n{}\n",
        total_chars,
        report.paragraphs_scanned,
        report.level.to_uppercase(),
        structural_summary,
        headings_line,
        opening
    );

    if !closing.is_empty() {
        user_prompt.push_str(&format!("\nDOKUMENTETS AVSLUTNING:\n{}\n", closing));
    }

    user_prompt.push_str(&format!(
        "\nAVSNITT SOM LAGER 3 FLAGGADE:\n{}\n\n\
Svara ENDAST med JSON enligt schemat.",
        flagged_summary
    ));

    format!(
        "<bos><|turn>system\n{system_prompt}<turn|>\n<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )
}

/// Tolkar JSON-svaret robust även om modellen skulle ha lagt till blanksteg eller om grammatik-fallback användes.
pub fn parse_raw_judge_json(raw: &str) -> Result<RawJudgeJson, String> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<RawJudgeJson>(trimmed) {
        return Ok(sanitize_raw_judge(parsed));
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            let slice = &trimmed[start..=end];
            if let Ok(parsed) = serde_json::from_str::<RawJudgeJson>(slice) {
                return Ok(sanitize_raw_judge(parsed));
            }
        }
    }

    Err(format!(
        "Kunde inte tolka JSON-svaret från Lager 4: {}",
        truncate_chars(trimmed, 180)
    ))
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
}
