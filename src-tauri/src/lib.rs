mod classifier;
mod documents;
mod ner;

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};

use classifier::{build_report, ClassificationReport, EmbeddingClassifier, ParagraphFinding};
use ner::{Match, NerModel};
use regex::Regex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tokenizers::Tokenizer;

// ── Globala singletons ────────────────────────────────────────────────────────

static SESSION: OnceLock<NerModel> = OnceLock::new();
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
static CLASSIFIER: OnceLock<EmbeddingClassifier> = OnceLock::new();

/// Filer för KBLab BERT NER
const MODEL_FILES: &[(&str, &str)] = &[
    ("onnx/model_quantized.onnx", "model_quantized.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("config.json", "config.json"),
];

/// Filer för EmbeddingGemma 2 (Q4 text-backbone)
const GEMMA_FILES: &[(&str, &str)] = &[
    ("onnx/model_q4.onnx", "model_q4.onnx"),
    ("onnx/model_q4.onnx_data", "model_q4.onnx_data"),
    ("tokenizer.json", "tokenizer.json"),
];

const HF_BASE_NER: &str =
    "https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx/resolve/main";
const HF_BASE_GEMMA: &str =
    "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main";

// ── Förkompilerade reguljära uttryck (kompileras en enda gång) ────────────────

static RE_PNR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\d{2})?\d{6}[-+]?\d{4}\b").unwrap());

static RE_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap()
});

static RE_PHONE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        (?:(?:\+46)|\b(?:0046|0))
        [\s\-]?
        [1-9]\d{0,2}
        (?:[\s\-]?\d{2,4}){2,4}\b
        ",
    )
    .unwrap()
});

static RE_ORGNR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\d{2})?\d{6}-\d{4}\b").unwrap());

static RE_IBAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bSE\d{2}[0-9A-Z]{20}\b").unwrap());

static RE_BG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{3,4}[\s\-]\d{4}\b").unwrap());

static RE_PG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{1,7}-\d\b").unwrap());

static RE_IPV4: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b")
        .unwrap()
});

static RE_CARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d(?:[\s\-]?\d){12,18}\b").unwrap());

/// Tjänstekortsnummer / SITHS-kortnummer / certifikatserienummer (16–22 sammanhängande siffror)
static RE_SERVICE_CARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{16,22}\b").unwrap());

// ── Tekniska hemligheter, nycklar, HSA-ID och interna intranätlänkar ─────────
static RE_SECRETS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i:-----BEGIN\s+(?:[A-Z\s]+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:[A-Z\s]+)?PRIVATE\s+KEY-----)",
        r"|\b(?:sk-[a-zA-Z0-9_-]{20,64})\b",
        r"|\b(?:ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{40,90})\b",
        r"|\b(?:xox[baprs]-[0-9a-zA-Z]{10,48})\b",
        r"|\b(?:AKIA[0-9A-Z]{16})\b",
        r"|\b(?:AIza[0-9A-Za-z-_]{35})\b",
        r"|(?:(?i:bearer)\s+eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})",
        r"|\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b",
        r"|(?:(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis):\/\/[^\s:@/]+:[^\s:@/]+@[^\s/]+)",
        r#"|(?:https?:\/\/(?:[a-zA-Z0-9-]+\.)*(?:intranat|intranet|internal|corp|lan|local)(?:\.[a-zA-Z0-9-]+)+(?::\d+)?(?:\/[^\s"'<>)]*)?)"#,
        r"|\bSE\d{10}-[A-Za-z0-9]{3,15}\b",
        r#"|(?:(?i:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|lösenord)\s*[:=]\s*['"]?[A-Za-z0-9_\-.~!@#$%^&*()+=]{8,64}['"]?)"#
    ))
    .unwrap()
});

// ── Event-payloads ────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct NerProgress {
    chunk: usize,
    total: usize,
}

#[derive(Clone, Serialize)]
struct ModelStatus {
    message: String,
}

#[derive(Clone, Serialize)]
struct DownloadProgress {
    file: String,
    downloaded: u64,
    total: u64,
}

// ── Hjälpfunktioner för kataloger ─────────────────────────────────────────────

fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Kan inte hitta app-datakatalog: {e}"))?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Kan inte skapa modellkatalog: {e}"))?;
    Ok(dir)
}

fn gemma_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = models_dir(app)?.join("embeddinggemma2");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Kan inte skapa gemma-katalog: {e}"))?;
    Ok(dir)
}

fn find_source_model_file(src_dir: &Path, remote_rel: &str, local_name: &str) -> Option<PathBuf> {
    let flat = src_dir.join(local_name);
    if flat.exists() {
        return Some(flat);
    }
    let nested = src_dir.join(remote_rel);
    if nested.exists() {
        return Some(nested);
    }
    None
}

// ── KBLab BERT NER Kommandon ──────────────────────────────────────────────────

#[tauri::command]
fn models_exist(app: AppHandle) -> Result<bool, String> {
    let data = models_dir(&app)?;
    let resource = app.path().resource_dir().ok().map(|d| d.join("models"));

    for &(remote_rel, local_name) in MODEL_FILES {
        let in_data = data.join(local_name).exists();
        let in_res = resource
            .as_ref()
            .is_some_and(|d| find_source_model_file(d, remote_rel, local_name).is_some());
        if !in_data && !in_res {
            return Ok(false);
        }
    }
    Ok(true)
}

#[tauri::command]
async fn download_model(app: AppHandle) -> Result<(), String> {
    use futures_util::StreamExt;
    use std::io::Write;

    let dir = models_dir(&app)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("HTTP-klient-fel: {e}"))?;

    for &(remote_rel, local_name) in MODEL_FILES {
        let dest = dir.join(local_name);
        if dest.exists() {
            continue;
        }

        let _ = app.emit(
            "model_status",
            ModelStatus {
                message: format!("Laddar ner {local_name}…"),
            },
        );

        let resp = client
            .get(format!("{HF_BASE_NER}/{remote_rel}"))
            .send()
            .await
            .map_err(|e| format!("Nätverksfel för {local_name}: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("HTTP {} för {local_name}", resp.status()));
        }

        let total = resp.content_length().unwrap_or(0);
        let tmp = dest.with_extension("part");
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("Kan inte skapa tempfil för {local_name}: {e}"))?;
        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Läsfel för {local_name}: {e}"))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Skrivfel för {local_name}: {e}"))?;
            downloaded += chunk.len() as u64;
            let _ = app.emit(
                "download_progress",
                DownloadProgress {
                    file: local_name.to_string(),
                    downloaded,
                    total,
                },
            );
        }

        drop(file);
        if std::fs::rename(&tmp, &dest).is_err() {
            std::fs::copy(&tmp, &dest)
                .map_err(|e| format!("Kunde inte flytta {local_name}: {e}"))?;
            let _ = std::fs::remove_file(&tmp);
        }
    }

    Ok(())
}

#[tauri::command]
async fn load_model_from_dir(src_dir: String, app: AppHandle) -> Result<(), String> {
    let src = PathBuf::from(&src_dir);
    let dest = models_dir(&app)?;

    let mut resolved: Vec<(PathBuf, &'static str)> = Vec::with_capacity(MODEL_FILES.len());
    let mut missing: Vec<&str> = Vec::new();
    for &(remote_rel, local_name) in MODEL_FILES {
        if let Some(found) = find_source_model_file(&src, remote_rel, local_name) {
            resolved.push((found, local_name));
        } else {
            missing.push(local_name);
        }
    }
    if !missing.is_empty() {
        return Err(format!("Saknade filer i vald mapp: {}", missing.join(", ")));
    }

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        for (src_path, local_name) in resolved {
            let _ = app_c.emit(
                "model_status",
                ModelStatus {
                    message: format!("Kopierar {local_name}…"),
                },
            );
            std::fs::copy(&src_path, dest.join(local_name))
                .map_err(|e| format!("Kunde inte kopiera {local_name}: {e}"))?;
        }
        load_and_store(&dest, &app_c)
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    app.emit("model_ready", ()).map_err(|e| e.to_string())
}

#[tauri::command]
async fn load_model(app: AppHandle) -> Result<(), String> {
    if SESSION.get().is_some() {
        app.emit("model_ready", ()).map_err(|e| e.to_string())?;
        return Ok(());
    }

    let _ = app.emit(
        "model_status",
        ModelStatus {
            message: "Kontrollerar modellfiler…".into(),
        },
    );

    let data_dir = models_dir(&app)?;
    let res_dir = app.path().resource_dir().ok().map(|d| d.join("models"));

    for &(remote_rel, local_name) in MODEL_FILES {
        let dst = data_dir.join(local_name);
        if dst.exists() {
            continue;
        }
        if let Some(ref rd) = res_dir {
            if let Some(src) = find_source_model_file(rd, remote_rel, local_name) {
                std::fs::copy(&src, &dst)
                    .map_err(|e| format!("Kunde inte kopiera {local_name}: {e}"))?;
                continue;
            }
        }
        return Err(format!(
            "Modellfil saknas: {local_name}. Starta om appen för att ladda ner."
        ));
    }

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || load_and_store(&data_dir, &app_c))
        .await
        .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    app.emit("model_ready", ()).map_err(|e| e.to_string())
}

fn load_and_store(dir: &Path, app: &AppHandle) -> Result<(), String> {
    let _ = app.emit(
        "model_status",
        ModelStatus {
            message: "Initierar ONNX Runtime…".into(),
        },
    );
    let model = ner::load_model(&dir.join("model_quantized.onnx"))?;
    let tokenizer = ner::load_tokenizer(&dir.join("tokenizer.json"))?;
    let _ = SESSION.set(model);
    let _ = TOKENIZER.set(tokenizer);
    Ok(())
}

// ── EmbeddingGemma 2 Kommandon (Lager 3 AI-Gatekeeper) ────────────────────────

#[tauri::command]
fn gemma_exists(app: AppHandle) -> Result<bool, String> {
    let dir = gemma_dir(&app)?;
    for &(_, local_name) in GEMMA_FILES {
        if !dir.join(local_name).exists() {
            return Ok(false);
        }
    }
    Ok(true)
}

#[tauri::command]
async fn download_gemma(app: AppHandle) -> Result<(), String> {
    use futures_util::StreamExt;
    use std::io::Write;

    let dir = gemma_dir(&app)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1200))
        .build()
        .map_err(|e| format!("HTTP-klient-fel: {e}"))?;

    for &(remote_rel, local_name) in GEMMA_FILES {
        let dest = dir.join(local_name);
        if dest.exists() {
            continue;
        }

        let _ = app.emit(
            "gemma_status",
            ModelStatus {
                message: format!("Laddar ner {local_name} (EmbeddingGemma 2)…"),
            },
        );

        let resp = client
            .get(format!("{HF_BASE_GEMMA}/{remote_rel}"))
            .send()
            .await
            .map_err(|e| format!("Nätverksfel för {local_name}: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("HTTP {} för {local_name}", resp.status()));
        }

        let total = resp.content_length().unwrap_or(0);
        let tmp = dest.with_extension("part");
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("Kan inte skapa tempfil för {local_name}: {e}"))?;
        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Läsfel för {local_name}: {e}"))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Skrivfel för {local_name}: {e}"))?;
            downloaded += chunk.len() as u64;
            let _ = app.emit(
                "gemma_download_progress",
                DownloadProgress {
                    file: local_name.to_string(),
                    downloaded,
                    total,
                },
            );
        }

        drop(file);
        if std::fs::rename(&tmp, &dest).is_err() {
            std::fs::copy(&tmp, &dest)
                .map_err(|e| format!("Kunde inte flytta {local_name}: {e}"))?;
            let _ = std::fs::remove_file(&tmp);
        }
    }

    Ok(())
}

#[tauri::command]
async fn load_gemma(app: AppHandle) -> Result<(), String> {
    if CLASSIFIER.get().is_some() {
        app.emit("gemma_ready", ()).map_err(|e| e.to_string())?;
        return Ok(());
    }

    let dir = gemma_dir(&app)?;
    for &(_, local_name) in GEMMA_FILES {
        if !dir.join(local_name).exists() {
            return Err(format!(
                "EmbeddingGemma 2-fil saknas: {local_name}. Ladda ner modellen först."
            ));
        }
    }

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let _ = app_c.emit(
            "gemma_status",
            ModelStatus {
                message: "Initierar EmbeddingGemma 2 & beräknar klassificeringsankare…".into(),
            },
        );
        let clf = EmbeddingClassifier::load(&dir)?;
        let _ = CLASSIFIER.set(clf);
        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    app.emit("gemma_ready", ()).map_err(|e| e.to_string())
}

#[tauri::command]
async fn classify_document(
    original_text: String,
    masked_text: String,
    active_pii_count: usize,
    unmasked_pii_count: usize,
    secret_count: usize,
    unmasked_secret_count: usize,
    cached_findings: Option<Vec<ParagraphFinding>>,
    cached_paragraphs_scanned: Option<usize>,
    cached_duration_ms: Option<u64>,
    app: AppHandle,
) -> Result<ClassificationReport, String> {
    tauri::async_runtime::spawn_blocking(move || -> Result<ClassificationReport, String> {
        let semantic_available = CLASSIFIER.get().is_some();

        let (findings, paragraphs_scanned, scan_duration_ms) =
            if let Some(existing) = cached_findings {
                (
                    existing,
                    cached_paragraphs_scanned.unwrap_or(0),
                    cached_duration_ms.unwrap_or(0),
                )
            } else if let Some(clf) = CLASSIFIER.get() {
                let t0 = std::time::Instant::now();
                let (found, total_paras) = clf.scan_paragraphs_with_progress(
                    &original_text,
                    &masked_text,
                    |chunk, total| {
                        let _ = app.emit("gemma_scan_progress", NerProgress { chunk, total });
                    },
                );
                let ms = t0.elapsed().as_millis() as u64;
                (found, total_paras, ms)
            } else {
                (Vec::new(), 0, 0)
            };

        Ok(build_report(
            active_pii_count,
            unmasked_pii_count,
            secret_count,
            unmasked_secret_count,
            semantic_available,
            paragraphs_scanned,
            scan_duration_ms,
            findings,
        ))
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel (classify): {e}"))?
}

// ── Textanalys ────────────────────────────────────────────────────────────────

#[tauri::command]
async fn analyze_text(
    text: String,
    entities: Vec<String>,
    threshold: f32,
    app: AppHandle,
) -> Result<Vec<Match>, String> {
    const REGEX_ONLY: &[&str] = &[
        "Personnummer",
        "E-post",
        "Telefonnummer",
        "Org.nummer",
        "Konto/Giro",
        "IP-adress",
        "Kortnummer",
        "Hemlighet/Nyckel",
    ];

    let ner_entities: Vec<String> = entities
        .iter()
        .filter(|e| !REGEX_ONLY.contains(&e.as_str()))
        .cloned()
        .collect();

    if !ner_entities.is_empty() && SESSION.get().is_none() {
        return Err("AI-modellen är inte laddad ännu".into());
    }

    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<Match>, String> {
        let regex_matches = run_regex(&text, &entities);
        if ner_entities.is_empty() {
            return Ok(regex_matches);
        }

        let model = SESSION.get().ok_or("Modell ej laddad")?;
        let tokenizer = TOKENIZER.get().ok_or("Tokenizer ej laddad")?;
        let ner_matches = ner::run_ner(
            &text,
            &ner_entities,
            threshold,
            model,
            tokenizer,
            |chunk, total| {
                let _ = app.emit("ner_progress", NerProgress { chunk, total });
            },
        );

        Ok(merge_regex_and_ner(regex_matches, ner_matches))
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))?
}

fn merge_regex_and_ner(regex_matches: Vec<Match>, ner_matches: Vec<Match>) -> Vec<Match> {
    let mut combined = Vec::with_capacity(regex_matches.len() + ner_matches.len());
    for nm in ner_matches {
        let overlaps_regex = regex_matches
            .iter()
            .any(|rx| !(nm.end <= rx.start || nm.start >= rx.end));
        if !overlaps_regex {
            combined.push(nm);
        }
    }
    combined.extend(regex_matches);
    combined.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));

    let mut deduped = Vec::with_capacity(combined.len());
    let mut last_end = 0usize;
    for m in combined {
        if m.start >= last_end {
            last_end = m.end;
            deduped.push(m);
        }
    }
    deduped
}

fn valid_swedish_pnr(s: &str) -> bool {
    let digits: Vec<u8> = s
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();

    let offset = match digits.len() {
        10 => 0,
        12 => {
            let century = digits[0] * 10 + digits[1];
            if !matches!(century, 18..=20) {
                return false;
            }
            2
        }
        _ => return false,
    };

    let month = digits[offset + 2] * 10 + digits[offset + 3];
    let day = digits[offset + 4] * 10 + digits[offset + 5];

    (1..=12).contains(&month) && ((1..=31).contains(&day) || (61..=91).contains(&day))
}

fn valid_swedish_orgnr(s: &str) -> bool {
    let digits: Vec<u8> = s
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();

    let offset = match digits.len() {
        10 => 0,
        12 => {
            let prefix = digits[0] * 10 + digits[1];
            if prefix != 16 {
                return false;
            }
            2
        }
        _ => return false,
    };

    let middle = digits[offset + 2] * 10 + digits[offset + 3];
    middle >= 20
}

fn luhn_valid(s: &str) -> bool {
    let nums: Vec<u32> = s
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c.to_digit(10).unwrap())
        .collect();
    if !(13..=19).contains(&nums.len()) {
        return false;
    }
    let sum: u32 = nums
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                let x = d * 2;
                if x > 9 {
                    x - 9
                } else {
                    x
                }
            } else {
                d
            }
        })
        .sum();
    sum % 10 == 0
}

fn overlaps_any(matches: &[Match], start: usize, end: usize) -> bool {
    matches.iter().any(|x| !(end <= x.start || start >= x.end))
}

fn run_regex(text: &str, requested: &[String]) -> Vec<Match> {
    let has = |name: &str| requested.iter().any(|e| e == name);
    let mut matches = Vec::new();

    // ── Tekniska hemligheter och API-nycklar (högsta prioritet) ──
    if has("Hemlighet/Nyckel") {
        for m in RE_SECRETS.find_iter(text) {
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Hemlighet/Nyckel".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Personnummer / Samordningsnummer ──
    if has("Personnummer") {
        for m in RE_PNR.find_iter(text) {
            let s = m.as_str();
            if !valid_swedish_pnr(s) || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Personnummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Organisationsnummer (mittenpar >= 20) ──
    if has("Org.nummer") {
        for m in RE_ORGNR.find_iter(text) {
            let s = m.as_str();
            if !valid_swedish_orgnr(s) || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Org.nummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── E-post ──
    if has("E-post") {
        for m in RE_EMAIL.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "E-post".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── IBAN (före telefon/kort så SE-IBAN inte klipps) ──
    if has("Konto/Giro") {
        for m in RE_IBAN.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Telefonnummer ──
    if has("Telefonnummer") {
        for m in RE_PHONE.find_iter(text) {
            let s = m.as_str().trim_end();
            let end = m.start() + s.len();
            if overlaps_any(&matches, m.start(), end) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Telefonnummer".to_string(),
                start: m.start(),
                end,
                source: "regex".to_string(),
            });
        }
    }

    // ── Bankgiro & Plusgiro ──
    if has("Konto/Giro") {
        for m in RE_BG.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
        for m in RE_PG.find_iter(text) {
            if m.end() - m.start() < 3 || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── IPv4-adresser (publika) ──
    if has("IP-adress") {
        for m in RE_IPV4.find_iter(text) {
            let s = m.as_str();
            let is_private = s.starts_with("10.")
                || s.starts_with("127.")
                || s.starts_with("192.168.")
                || s.strip_prefix("172.").is_some_and(|rest| {
                    rest.split('.')
                        .next()
                        .and_then(|n| n.parse::<u32>().ok())
                        .is_some_and(|n| (16..=31).contains(&n))
                });
            if is_private || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "IP-adress".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Kreditkortsnummer (med Luhn-validering) samt tjänstekort/SITHS-kort (16–22 siffror) ──
    if has("Kortnummer") {
        for m in RE_SERVICE_CARD.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Kortnummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
        for m in RE_CARD.find_iter(text) {
            let s = m.as_str();
            if !luhn_valid(s) || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Kortnummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    matches.sort_by_key(|m| m.start);
    matches
}

#[tauri::command]
fn write_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content.as_bytes()).map_err(|e| e.to_string())
}

#[tauri::command]
async fn import_document_bytes(
    file_name: String,
    bytes: Vec<u8>,
) -> Result<documents::ImportedDocument, String> {
    tauri::async_runtime::spawn_blocking(move || {
        documents::import_document_bytes(&file_name, &bytes)
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel (import): {e}"))?
}

#[tauri::command]
async fn export_masked_docx(
    path: String,
    original_bytes: Vec<u8>,
    replacements: Vec<documents::ReplacementPair>,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let masked_bytes = documents::export_masked_docx_bytes(&original_bytes, &replacements)?;
        std::fs::write(&path, &masked_bytes).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel (export_docx): {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            models_exist,
            download_model,
            load_model,
            load_model_from_dir,
            gemma_exists,
            download_gemma,
            load_gemma,
            classify_document,
            analyze_text,
            write_file,
            import_document_bytes,
            export_masked_docx
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_regex_labels() -> Vec<String> {
        vec![
            "Personnummer",
            "E-post",
            "Telefonnummer",
            "Org.nummer",
            "Konto/Giro",
            "IP-adress",
            "Kortnummer",
            "Hemlighet/Nyckel",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn separates_personnummer_and_orgnummer() {
        let text = "Sökande 850515-1234 (samordning 701063-2391) vid bolaget 556614-5180.";
        let res = run_regex(text, &all_regex_labels());
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].label, "Personnummer");
        assert_eq!(res[0].text, "850515-1234");
        assert_eq!(res[1].label, "Personnummer");
        assert_eq!(res[1].text, "701063-2391");
        assert_eq!(res[2].label, "Org.nummer");
        assert_eq!(res[2].text, "556614-5180");
    }

    #[test]
    fn detects_api_keys_and_tokens() {
        let text = "OpenAI key: sk-abcdef1234567890abcdef123456, GitHub: ghp_123456789012345678901234567890123456";
        let res = run_regex(text, &all_regex_labels());
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].label, "Hemlighet/Nyckel");
        assert_eq!(res[1].label, "Hemlighet/Nyckel");
    }

    #[test]
    fn detects_siths_card_and_intranet_url() {
        let text = "Ditt SITHS-kort nr. 9752267235710527661 förnyas via https://intranat.rjl.se/service-och-stod-i-arbetet/bestalla/?pageId=49904";
        let res = run_regex(text, &all_regex_labels());
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].label, "Kortnummer");
        assert_eq!(res[0].text, "9752267235710527661");
        assert_eq!(res[1].label, "Hemlighet/Nyckel");
        assert!(res[1].text.starts_with("https://intranat.rjl.se/"));
    }

    #[test]
    fn regex_wins_over_overlapping_ner() {
        let rx = vec![Match {
            text: "anna.svensson@example.se".into(),
            label: "E-post".into(),
            start: 10,
            end: 34,
            source: "regex".into(),
        }];
        let ner = vec![
            Match {
                text: "anna.svensson".into(),
                label: "Person".into(),
                start: 10,
                end: 23,
                source: "kblab".into(),
            },
            Match {
                text: "Stockholm".into(),
                label: "Plats".into(),
                start: 40,
                end: 49,
                source: "kblab".into(),
            },
        ];
        let merged = merge_regex_and_ner(rx, ner);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].label, "E-post");
        assert_eq!(merged[1].label, "Plats");
    }
}
