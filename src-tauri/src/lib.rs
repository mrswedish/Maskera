mod ner;

use std::sync::OnceLock;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager};
use ner::{NerModel, Match};
use tokenizers::Tokenizer;
use serde::Serialize;

// ── Global singleton ──────────────────────────────────────────────────────────

static SESSION: OnceLock<NerModel> = OnceLock::new();
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

const MODEL_FILES: &[&str] = &[
    "model_quantized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
];

const HF_BASE: &str =
    "https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx/resolve/main";

// ── Event-payloads ────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct NerProgress { chunk: usize, total: usize }

#[derive(Clone, Serialize)]
struct ModelStatus { message: String }

#[derive(Clone, Serialize)]
struct DownloadProgress { file: String, downloaded: u64, total: u64 }

// ── Hjälpfunktion ─────────────────────────────────────────────────────────────

fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Kan inte hitta app-datakatalog: {e}"))?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Kan inte skapa modellkatalog: {e}"))?;
    Ok(dir)
}

// ── Kommandon ─────────────────────────────────────────────────────────────────

/// Kontrollerar om alla modellfiler finns lokalt (finns också fallback-källa).
#[tauri::command]
fn models_exist(app: AppHandle) -> Result<bool, String> {
    let data = models_dir(&app)?;
    let resource = app.path().resource_dir().ok().map(|d| d.join("models"));

    for &f in MODEL_FILES {
        let in_data = data.join(f).exists();
        let in_res  = resource.as_ref().map_or(false, |d| d.join(f).exists());
        if !in_data && !in_res { return Ok(false); }
    }
    Ok(true)
}

/// Laddar ner modellfilerna från HuggingFace via async streaming.
/// Använder reqwest med stream-feature — ingen blocking, ingen UI-frysning.
/// Emitterar "download_progress" och "model_status" events löpande.
#[tauri::command]
async fn download_model(app: AppHandle) -> Result<(), String> {
    use futures_util::StreamExt;
    use std::io::Write;

    let dir = models_dir(&app)?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("HTTP-klient-fel: {e}"))?;

    for &filename in MODEL_FILES {
        let dest = dir.join(filename);
        if dest.exists() { continue; }

        let _ = app.emit("model_status", ModelStatus {
            message: format!("Laddar ner {filename}…"),
        });

        let resp = client
            .get(format!("{HF_BASE}/{filename}"))
            .send()
            .await
            .map_err(|e| format!("Nätverksfel för {filename}: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("HTTP {} för {filename}", resp.status()));
        }

        let total = resp.content_length().unwrap_or(0);
        let tmp = dest.with_extension("part");
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("Kan inte skapa tempfil för {filename}: {e}"))?;
        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Läsfel för {filename}: {e}"))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Skrivfel för {filename}: {e}"))?;
            downloaded += chunk.len() as u64;
            let _ = app.emit("download_progress", DownloadProgress {
                file: filename.to_string(),
                downloaded,
                total,
            });
        }

        drop(file);
        // Atomic rename: .part → slutlig fil (Windows-safe fallback till copy)
        if std::fs::rename(&tmp, &dest).is_err() {
            std::fs::copy(&tmp, &dest)
                .map_err(|e| format!("Kunde inte flytta {filename}: {e}"))?;
            let _ = std::fs::remove_file(&tmp);
        }
    }

    Ok(())
}

/// Kopierar modellfiler från en manuellt vald mapp (för brandväggsbegränsade miljöer).
#[tauri::command]
async fn load_model_from_dir(src_dir: String, app: AppHandle) -> Result<(), String> {
    let src  = PathBuf::from(&src_dir);
    let dest = models_dir(&app)?;

    let missing: Vec<&str> = MODEL_FILES.iter()
        .filter(|&&f| !src.join(f).exists())
        .copied().collect();
    if !missing.is_empty() {
        return Err(format!("Saknade filer i vald mapp: {}", missing.join(", ")));
    }

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        for &filename in MODEL_FILES {
            let _ = app_c.emit("model_status", ModelStatus {
                message: format!("Kopierar {filename}…"),
            });
            std::fs::copy(src.join(filename), dest.join(filename))
                .map_err(|e| format!("Kunde inte kopiera {filename}: {e}"))?;
        }
        load_and_store(&dest, &app_c)
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    app.emit("model_ready", ()).map_err(|e| e.to_string())
}

/// Laddar ONNX-modell + tokenizer från modell-katalogen.
/// Filer måste finnas på disk redan (nedladdade av download_model eller kopierade manuellt).
#[tauri::command]
async fn load_model(app: AppHandle) -> Result<(), String> {
    if SESSION.get().is_some() {
        app.emit("model_ready", ()).map_err(|e| e.to_string())?;
        return Ok(());
    }

    let _ = app.emit("model_status", ModelStatus {
        message: "Kontrollerar modellfiler…".into(),
    });

    let data_dir = models_dir(&app)?;
    let res_dir  = app.path().resource_dir().ok().map(|d| d.join("models"));

    // Dev-fallback: kopiera från resource_dir om filerna finns där men inte i data_dir
    for &filename in MODEL_FILES {
        let dst = data_dir.join(filename);
        if dst.exists() { continue; }
        if let Some(ref rd) = res_dir {
            let src = rd.join(filename);
            if src.exists() {
                eprintln!("[load_model] Kopierar {filename} från resource_dir");
                std::fs::copy(&src, &dst)
                    .map_err(|e| format!("Kunde inte kopiera {filename}: {e}"))?;
                continue;
            }
        }
        return Err(format!("Modellfil saknas: {filename}. Starta om appen för att ladda ner."));
    }

    let _ = app.emit("model_status", ModelStatus {
        message: "Optimerar AI-modell (tar ~30 sek första gången)…".into(),
    });

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || load_and_store(&data_dir, &app_c))
        .await
        .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    app.emit("model_ready", ()).map_err(|e| e.to_string())
}

fn load_and_store(dir: &std::path::Path, app: &AppHandle) -> Result<(), String> {
    let _ = app.emit("model_status", ModelStatus {
        message: "Optimerar AI-modell (tar ~30 sek första gången)…".into(),
    });
    let model = ner::load_model(&dir.join("model_quantized.onnx"))
        .map_err(|e| format!("Modell-laddning misslyckades: {e}"))?;
    let tokenizer = ner::load_tokenizer(&dir.join("tokenizer.json"))
        .map_err(|e| format!("Tokenizer-laddning misslyckades: {e}"))?;
    let _ = SESSION.set(model);
    let _ = TOKENIZER.set(tokenizer);
    eprintln!("[load_model] Modell laddad och redo");
    Ok(())
}

/// Kör regex + NER på text.
#[tauri::command]
async fn analyze_text(
    text: String,
    entities: Vec<String>,
    threshold: f32,
    app: AppHandle,
) -> Result<Vec<Match>, String> {
    if SESSION.get().is_none() {
        return Err("Modell ej laddad".into());
    }
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<Match>, String> {
        let mut results = run_regex(&text, &entities);
        // Regex-kategorier ska inte skickas till NER-pipelinen
        let regex_only = [
            "Personnummer", "E-post", "Telefonnummer",
            "Org.nummer", "Konto/Giro", "IP-adress", "Kortnummer",
        ];
        let ner_entities: Vec<String> = entities.iter()
            .filter(|e| !regex_only.iter().any(|r| r == e))
            .cloned().collect();
        if !ner_entities.is_empty() {
            let model     = SESSION.get().ok_or("Modell ej laddad")?;
            let tokenizer = TOKENIZER.get().ok_or("Tokenizer ej laddad")?;
            let ner_matches = ner::run_ner(&text, &ner_entities, threshold, model, tokenizer, |chunk, total| {
                let _ = app.emit("ner_progress", NerProgress { chunk, total });
            });
            results.extend(ner_matches);
        }
        results.sort_by_key(|m| m.start);
        Ok(results)
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))?
}

/// Luhn-algoritmen — validerar kreditkortsnummer. Returnerar false för
/// sekvenser med fel längd eller fel kontrollsiffra.
fn luhn_valid(s: &str) -> bool {
    let nums: Vec<u32> = s.chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c.to_digit(10).unwrap())
        .collect();
    if nums.len() < 13 || nums.len() > 19 { return false; }
    let sum: u32 = nums.iter().rev().enumerate().map(|(i, &d)| {
        if i % 2 == 1 {
            let x = d * 2;
            if x > 9 { x - 9 } else { x }
        } else { d }
    }).sum();
    sum % 10 == 0
}

fn run_regex(text: &str, requested: &[String]) -> Vec<Match> {
    let mut matches = Vec::new();

    // ── Personnummer (10 eller 12 siffror, med valfri separator) ──
    if requested.iter().any(|e| e == "Personnummer") {
        let re = regex::Regex::new(r"(?x)\b(?:\d{2})?\d{6}[-+]?\d{4}\b").unwrap();
        for m in re.find_iter(text) {
            matches.push(Match { text: m.as_str().to_string(), label: "Personnummer".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }

    // ── E-post ──
    if requested.iter().any(|e| e == "E-post") {
        let re = regex::Regex::new(r"(?i)\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap();
        for m in re.find_iter(text) {
            matches.push(Match { text: m.as_str().to_string(), label: "E-post".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }

    // ── Telefonnummer — tillåt mellanslag/bindestreck mellan grupper ──
    // Tidigare regex missade ~80% av vanliga svenska format som "+46 8 123 45 67".
    if requested.iter().any(|e| e == "Telefonnummer") {
        let re = regex::Regex::new(
            r"(?x)
            (?:\+46|0046|0)
            [\s\-]?
            [1-9]\d{0,2}
            (?:[\s\-]?\d{2,4}){2,4}
            "
        ).unwrap();
        for m in re.find_iter(text) {
            let s = m.as_str().trim_end();
            matches.push(Match { text: s.to_string(), label: "Telefonnummer".to_string(), start: m.start(), end: m.start() + s.len(), source: "regex".to_string() });
        }
    }

    // ── Organisationsnummer (XXXXXX-XXXX) ──
    // Negativ lookahead skulle behövas för att filtrera bort 12-siffriga personnummer
    // men Rust regex stöder inte lookahead — vi filtrerar bort kollisioner senare.
    if requested.iter().any(|e| e == "Org.nummer") {
        let re = regex::Regex::new(r"\b\d{6}-\d{4}\b").unwrap();
        for m in re.find_iter(text) {
            let s = m.as_str();
            // Hoppa över om det överlappar med ett redan funnit personnummer
            if matches.iter().any(|x| x.label == "Personnummer" && x.start <= m.start() && m.end() <= x.end) {
                continue;
            }
            matches.push(Match { text: s.to_string(), label: "Org.nummer".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }

    // ── IBAN / Bankgiro / Plusgiro ──
    if requested.iter().any(|e| e == "Konto/Giro") {
        // IBAN SE-format
        let iban = regex::Regex::new(r"\bSE\d{2}[0-9A-Z]{20}\b").unwrap();
        for m in iban.find_iter(text) {
            matches.push(Match { text: m.as_str().to_string(), label: "Konto/Giro".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
        // Bankgiro: 3-4 + 4 siffror (separator bindestreck eller mellanslag)
        let bg = regex::Regex::new(r"\b\d{3,4}[\s\-]\d{4}\b").unwrap();
        for m in bg.find_iter(text) {
            // Undvik krock med Personnummer, Telefon och redan funnen IBAN
            if matches.iter().any(|x|
                (x.label == "Personnummer" || x.label == "Telefonnummer" || x.label == "Konto/Giro")
                && !(m.end() <= x.start || m.start() >= x.end)
            ) { continue; }
            matches.push(Match { text: m.as_str().to_string(), label: "Konto/Giro".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
        // Plusgiro: 1-7 siffror + bindestreck + kontrollsiffra
        let pg = regex::Regex::new(r"\b\d{1,7}-\d\b").unwrap();
        for m in pg.find_iter(text) {
            // Längd minst 3 tecken (annars för många falska träffar t.ex. "1-2")
            if m.end() - m.start() < 3 { continue; }
            if matches.iter().any(|x|
                (x.label == "Personnummer" || x.label == "Telefonnummer" || x.label == "Konto/Giro" || x.label == "Org.nummer")
                && !(m.end() <= x.start || m.start() >= x.end)
            ) { continue; }
            matches.push(Match { text: m.as_str().to_string(), label: "Konto/Giro".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }

    // ── IPv4-adresser (publika) ──
    if requested.iter().any(|e| e == "IP-adress") {
        let re = regex::Regex::new(
            r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
        ).unwrap();
        for m in re.find_iter(text) {
            let s = m.as_str();
            // Filtrera bort privata och loopback-nät
            let is_private =
                s.starts_with("10.") ||
                s.starts_with("127.") ||
                s.starts_with("192.168.") ||
                {
                    if let Some(rest) = s.strip_prefix("172.") {
                        rest.split('.').next()
                            .and_then(|n| n.parse::<u32>().ok())
                            .map_or(false, |n| (16..=31).contains(&n))
                    } else { false }
                };
            if is_private { continue; }
            matches.push(Match { text: s.to_string(), label: "IP-adress".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }

    // ── Kreditkortsnummer (med Luhn-validering) ──
    if requested.iter().any(|e| e == "Kortnummer") {
        // Kandidatmönster: 13-19 siffror med valfri separator
        let re = regex::Regex::new(r"\b\d(?:[\s\-]?\d){12,18}\b").unwrap();
        for m in re.find_iter(text) {
            let s = m.as_str();
            if !luhn_valid(s) { continue; }
            // Undvik krock med Personnummer/Telefon/Bankgiro
            if matches.iter().any(|x|
                !(m.end() <= x.start || m.start() >= x.end)
            ) { continue; }
            matches.push(Match { text: s.to_string(), label: "Kortnummer".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }

    matches
}

#[tauri::command]
fn write_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content.as_bytes()).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            models_exist, download_model,
            load_model, load_model_from_dir, analyze_text, write_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
