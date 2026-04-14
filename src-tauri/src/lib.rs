mod ner;

use std::sync::OnceLock;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};
use ner::{NerModel, Match};
use tokenizers::Tokenizer;
use serde::Serialize;

// ── Global singleton (laddas en gång, lever hela appens livstid) ─────────────

static SESSION: OnceLock<NerModel> = OnceLock::new();
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

// ── HuggingFace-konfiguration ─────────────────────────────────────────────────

const HF_BASE: &str =
    "https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx/resolve/main";

const MODEL_FILES: &[&str] = &[
    "model_quantized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
];

// ── Event-payloads ────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct NerProgress {
    chunk: usize,
    total: usize,
}

#[derive(Clone, Serialize)]
struct DownloadProgress {
    file: String,
    downloaded: u64,
    total: u64,     // 0 = okänd storlek
}

#[derive(Clone, Serialize)]
struct ModelStatus {
    message: String,
}

// ── Hjälpfunktion: ladda ner en fil med progress-events ──────────────────────

fn download_file(
    url: &str,
    dest: &Path,
    app: &AppHandle,
    filename: &str,
) -> Result<(), String> {
    use std::io::Write;

    eprintln!("[download] {filename} → {dest:?}");

    // Bygg klient med timeout — annars hänger anslutningen tyst på Windows
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(600)) // max 10 min för 120 MB
        .build()
        .map_err(|e| format!("reqwest-klient-fel: {e}"))?;

    let mut resp = client.get(url).send()
        .map_err(|e| format!("Nedladdningsfel för {filename}: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {} för {filename}", resp.status()));
    }

    let total = resp.content_length().unwrap_or(0);
    let mut file = std::fs::File::create(dest)
        .map_err(|e| format!("Kan inte skapa {filename}: {e}"))?;

    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 64 * 1024]; // 64 KB-block

    loop {
        use std::io::Read;
        let n = resp.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        downloaded += n as u64;

        // Progress-event var 256 KB för att inte översvämma IPC
        if downloaded % (256 * 1024) < (n as u64) {
            let _ = app.emit("download_progress", DownloadProgress {
                file: filename.to_string(),
                downloaded,
                total,
            });
        }
    }
    // Slutlig progress
    let _ = app.emit("download_progress", DownloadProgress {
        file: filename.to_string(),
        downloaded,
        total,
    });
    eprintln!("[download] {filename} klar ({downloaded} bytes)");
    Ok(())
}

// ── Tauri-kommandon ───────────────────────────────────────────────────────────

/// Kontrollerar om modellfiler finns lokalt; laddar annars ner från HuggingFace.
/// Laddar sedan ONNX-modell + tokenizer och emitterar "model_ready".
#[tauri::command]
async fn load_model(app: AppHandle) -> Result<(), String> {
    if SESSION.get().is_some() {
        eprintln!("[load_model] Modell redan laddad — skickar model_ready direkt");
        app.emit("model_ready", ()).map_err(|e| e.to_string())?;
        return Ok(());
    }

    // Appens beständiga datakatalog (~/.local/share/com.patriksvensk.maskera på Linux,
    // ~/Library/Application Support/com.patriksvensk.maskera på macOS)
    let data_dir: PathBuf = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Kan inte hitta app-datakatalog: {e}"))?
        .join("models");

    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Kan inte skapa modellkatalog: {e}"))?;

    // Fallback-katalog: src-tauri/models/ (finns i dev-läge om modeller laddats ner manuellt)
    let resource_models = app.path().resource_dir().ok().map(|d| d.join("models"));

    eprintln!("[load_model] App-datakatalog: {data_dir:?}");
    if let Some(ref rd) = resource_models {
        eprintln!("[load_model] Resurs-fallback:  {rd:?}");
    }

    // Skicka omedelbar status till UI — syns innan spawn_blocking ens startar
    let _ = app.emit("model_status", ModelStatus { message: "Kontrollerar modellfiler…".into() });

    // Allt tungt (nedladdning + modell-optimering) körs i spawn_blocking.
    // reqwest::blocking FÅR INTE anropas direkt i async fn — blockerar tokio-tråden.
    let app_clone = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        // Ladda ner saknade filer — hoppa över om de redan finns i app_data_dir
        // ELLER i resource_dir (dev-fallback där modeller ofta finns lokalt)
        for &filename in MODEL_FILES {
            let dest = data_dir.join(filename);
            if dest.exists() {
                eprintln!("[load_model] {filename} finns i data-katalog");
                continue;
            }
            // Dev-fallback: kopiera från resource_dir om filen finns där
            if let Some(ref rd) = resource_models {
                let src = rd.join(filename);
                if src.exists() {
                    eprintln!("[load_model] {filename} kopieras från resurs-katalog");
                    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
                    std::fs::copy(&src, &dest).map_err(|e| e.to_string())?;
                    continue;
                }
            }
            // Ladda ner från HuggingFace
            eprintln!("[load_model] {filename} saknas — laddar ner från HuggingFace");
            let _ = app_clone.emit("model_status", ModelStatus {
                message: format!("Laddar ner {filename} från HuggingFace…"),
            });
            std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
            let url = format!("{HF_BASE}/{filename}");
            download_file(&url, &dest, &app_clone, filename)?;
        }

        let model_path = data_dir.join("model_quantized.onnx");
        let tok_path   = data_dir.join("tokenizer.json");

        let _ = app_clone.emit("model_status", ModelStatus {
            message: "Optimerar AI-modell (tar ~30 sek första gången)…".into(),
        });
        eprintln!("[load_model] Laddar och optimerar ONNX-modell…");
        let model = ner::load_model(&model_path)
            .map_err(|e| format!("Modell-laddning misslyckades: {e}"))?;
        eprintln!("[load_model] ONNX-modell klar");

        let tokenizer = ner::load_tokenizer(&tok_path)
            .map_err(|e| format!("Tokenizer-laddning misslyckades: {e}"))?;
        eprintln!("[load_model] Tokenizer laddad");

        let _ = SESSION.set(model);
        let _ = TOKENIZER.set(tokenizer);
        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    eprintln!("[load_model] Emitterar model_ready");
    app.emit("model_ready", ()).map_err(|e| e.to_string())?;
    Ok(())
}

/// Kör regex + NER på `text` och returnerar en lista av Match.
#[tauri::command]
async fn analyze_text(
    text: String,
    entities: Vec<String>,
    threshold: f32,
    app: AppHandle,
) -> Result<Vec<Match>, String> {
    eprintln!("[analyze_text] Startar — {} tecken, {:?}, tröskel={:.2}", text.len(), entities, threshold);

    if SESSION.get().is_none() {
        return Err("Modell ej laddad — anropa load_model först".into());
    }

    let result = tauri::async_runtime::spawn_blocking(move || -> Result<Vec<Match>, String> {
        let mut results = run_regex(&text, &entities);
        eprintln!("[analyze_text] Regex-pass: {} träffar", results.len());

        let ner_entities: Vec<String> = entities.iter()
            .filter(|e| !["Personnummer", "E-post", "Telefonnummer"].iter().any(|r| r == e))
            .cloned()
            .collect();

        if !ner_entities.is_empty() {
            let model     = SESSION.get().ok_or("Modell ej laddad")?;
            let tokenizer = TOKENIZER.get().ok_or("Tokenizer ej laddad")?;

            eprintln!("[analyze_text] NER för {:?}", ner_entities);
            let ner_matches = ner::run_ner(&text, &ner_entities, threshold, model, tokenizer, |chunk, total| {
                let _ = app.emit("ner_progress", NerProgress { chunk, total });
            });
            eprintln!("[analyze_text] NER klar: {} träffar", ner_matches.len());
            results.extend(ner_matches);
        }

        results.sort_by_key(|m| m.start);
        Ok(results)
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))?;

    result
}

// ── Regex-pass ────────────────────────────────────────────────────────────────

fn run_regex(text: &str, requested: &[String]) -> Vec<Match> {
    let mut matches = Vec::new();

    if requested.iter().any(|e| e == "Personnummer") {
        let re = regex::Regex::new(r"(?x)\b(?:\d{2})?\d{6}[-+]?\d{4}\b").unwrap();
        for m in re.find_iter(text) {
            matches.push(Match { text: m.as_str().to_string(), label: "Personnummer".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }
    if requested.iter().any(|e| e == "E-post") {
        let re = regex::Regex::new(r"(?i)\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap();
        for m in re.find_iter(text) {
            matches.push(Match { text: m.as_str().to_string(), label: "E-post".to_string(), start: m.start(), end: m.end(), source: "regex".to_string() });
        }
    }
    if requested.iter().any(|e| e == "Telefonnummer") {
        let re = regex::Regex::new(r"(?:(?:0|\+46|0046)\s?(?:[1-9]\d{1,2}\s?(?:[- ]?\d{2,3}){1,3}|\d{2,3}\s?(?:[- ]?\d{2,3}){1,3}))").unwrap();
        for m in re.find_iter(text) {
            let s = m.as_str().trim_end();
            matches.push(Match { text: s.to_string(), label: "Telefonnummer".to_string(), start: m.start(), end: m.start() + s.len(), source: "regex".to_string() });
        }
    }
    matches
}

/// Kopierar modellfiler från en användarvald mapp till app_data_dir och laddar dem.
/// Anropas när automatisk nedladdning misslyckas (t.ex. företags-proxy).
#[tauri::command]
async fn load_model_from_dir(src_dir: String, app: AppHandle) -> Result<(), String> {
    let src = PathBuf::from(&src_dir);
    let data_dir: PathBuf = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Kan inte hitta app-datakatalog: {e}"))?
        .join("models");

    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;

    // Verifiera att de nödvändiga filerna finns i källmappen
    let missing: Vec<&str> = MODEL_FILES.iter()
        .filter(|&&f| !src.join(f).exists())
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!("Saknade filer i vald mapp: {}", missing.join(", ")));
    }

    let app_clone = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        for &filename in MODEL_FILES {
            let from = src.join(filename);
            let to   = data_dir.join(filename);
            let _ = app_clone.emit("model_status", ModelStatus {
                message: format!("Kopierar {filename}…"),
            });
            std::fs::copy(&from, &to)
                .map_err(|e| format!("Kunde inte kopiera {filename}: {e}"))?;
        }
        let _ = app_clone.emit("model_status", ModelStatus {
            message: "Optimerar AI-modell…".into(),
        });
        let model = ner::load_model(&data_dir.join("model_quantized.onnx"))
            .map_err(|e| format!("Modell-laddning misslyckades: {e}"))?;
        let tokenizer = ner::load_tokenizer(&data_dir.join("tokenizer.json"))
            .map_err(|e| format!("Tokenizer-laddning misslyckades: {e}"))?;
        let _ = SESSION.set(model);
        let _ = TOKENIZER.set(tokenizer);
        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel: {e}"))??;

    app.emit("model_ready", ()).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Filsparning via native dialog ─────────────────────────────────────────────

#[tauri::command]
fn write_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content.as_bytes()).map_err(|e| e.to_string())
}

// ── Tauri-bootstrap ───────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![load_model, load_model_from_dir, analyze_text, write_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
