mod classifier;
mod documents;
mod info_classification;
mod info_judge;
mod ner;
mod regex_rules;
mod roles;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use classifier::{build_report, ClassificationReport, EmbeddingClassifier, ParagraphFinding};
use ner::{Match, NerModel};
use regex_rules::{merge_regex_and_ner, run_regex};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tokenizers::Tokenizer;

// ── Globala singletons ────────────────────────────────────────────────────────

static SESSION: OnceLock<NerModel> = OnceLock::new();
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
static CLASSIFIER: OnceLock<EmbeddingClassifier> = OnceLock::new();
static JUDGE_ENGINE: Mutex<Option<info_judge::JudgeEngine>> = Mutex::new(None);
static JUDGE_CANCEL: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "macos")]
static JUDGE_ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);

/// Frigör Lager 4-motorn (`LlamaContext` + `LlamaModel`) explicit innan processen avslutas.
/// På macOS med Metal kräver `ggml_metal_rsets_free` i C++ `atexit`-destruktorn att alla
/// GPU-residency-sets redan är avallokerade innan `std::process::exit` kör C++-destruktorerna.
fn shutdown_judge_engine() {
    if let Ok(mut guard) = JUDGE_ENGINE.lock() {
        *guard = None;
    }
}

#[cfg(target_os = "macos")]
extern "C" fn atexit_drop_judge_engine() {
    if let Ok(mut guard) = JUDGE_ENGINE.try_lock() {
        *guard = None;
    }
}

fn load_and_store_judge_engine(
    guard: &mut Option<info_judge::JudgeEngine>,
    path: &Path,
) -> Result<(), String> {
    let engine = info_judge::JudgeEngine::load(path)?;
    *guard = Some(engine);

    // Registrera en C `atexit`-hook EFTER att `ggml_metal_device` skapats av `JudgeEngine::load`.
    // Eftersom `atexit` körs i omvänd ordning (LIFO) garanterar detta att `JUDGE_ENGINE` alltid
    // frigörs FÖRE `ggml_metal_device_free` även om `std::process::exit` anropas direkt.
    #[cfg(target_os = "macos")]
    if !JUDGE_ATEXIT_REGISTERED.swap(true, Ordering::SeqCst) {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> std::ffi::c_int;
        }
        unsafe {
            let _ = atexit(atexit_drop_judge_engine);
        }
    }

    Ok(())
}

/// Filer för KBLab BERT NER (med ungefärlig storlek i byte för jämn total-progressbar)
const MODEL_FILES: &[(&str, &str, u64)] = &[
    ("onnx/model_quantized.onnx", "model_quantized.onnx", 125_800_000),
    ("tokenizer.json", "tokenizer.json", 1_150_000),
    ("tokenizer_config.json", "tokenizer_config.json", 1_300),
    ("special_tokens_map.json", "special_tokens_map.json", 125),
    ("config.json", "config.json", 1_200),
];

/// Filer för EmbeddingGemma 2 (Q4 text-backbone)
const GEMMA_FILES: &[(&str, &str, u64)] = &[
    ("onnx/model_q4.onnx", "model_q4.onnx", 490_000),
    ("onnx/model_q4.onnx_data", "model_q4.onnx_data", 174_000_000),
    ("tokenizer.json", "tokenizer.json", 32_500_000),
];

const HF_BASE_NER: &str =
    "https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx/resolve/main";
const HF_BASE_GEMMA: &str =
    "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main";

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

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelsStatus {
    ner_installed: bool,
    gemma_installed: bool,
    ner_ready: bool,
    gemma_ready: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SetupProgress {
    phase: String,
    step: usize,
    total_steps: usize,
    ner_done: bool,
    gemma_done: bool,
    label: String,
    downloaded_bytes: u64,
    total_bytes: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct JudgeDownloadProgress {
    phase: String,
    label: String,
    downloaded_bytes: u64,
    total_bytes: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct InfoClassifyPhase {
    phase: String,
    message: String,
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

/// Portabla / förpackade modellkataloger (bredvid Maskera-portabel.exe eller i resource_dir)
fn candidate_bundled_model_dirs(app: &AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
        }
    }
    if let Ok(res) = app.path().resource_dir() {
        dirs.push(res.join("models"));
    }
    dirs
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

/// Kontrollerar om en katalog innehåller alla nödvändiga filer för BERT NER
fn dir_has_all_ner_files(dir: &Path) -> bool {
    MODEL_FILES
        .iter()
        .all(|&(remote_rel, local_name, _)| find_source_model_file(dir, remote_rel, local_name).is_some())
}

/// Hittar katalog där alla BERT NER-filer redan finns (app_data_dir, exe_dir/models eller resource_dir/models)
fn resolve_existing_ner_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(data) = models_dir(app) {
        if dir_has_all_ner_files(&data) {
            return Some(data);
        }
    }
    candidate_bundled_model_dirs(app)
        .into_iter()
        .find(|cand| dir_has_all_ner_files(cand))
}

/// Kontrollerar om en katalog har alla tre EmbeddingGemma 2-filer direkt på plats (krävs av ONNX external_data)
fn dir_has_ready_gemma_files(dir: &Path) -> bool {
    GEMMA_FILES
        .iter()
        .all(|&(_, local_name, _)| dir.join(local_name).exists())
}

/// Hittar katalog där alla EmbeddingGemma 2-filer finns samlade
fn resolve_existing_gemma_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(gdir) = gemma_dir(app) {
        if dir_has_ready_gemma_files(&gdir) {
            return Some(gdir);
        }
    }
    for cand in candidate_bundled_model_dirs(app) {
        let sub = cand.join("embeddinggemma2");
        if dir_has_ready_gemma_files(&sub) {
            return Some(sub);
        }
        if dir_has_ready_gemma_files(&cand) {
            return Some(cand);
        }
    }
    None
}

// ── Samlad kontroll & engångsnedladdning (Lager 2 + Lager 3) ─────────────────

#[tauri::command]
fn check_models_status(app: AppHandle) -> Result<ModelsStatus, String> {
    let ner_ready = SESSION.get().is_some();
    let gemma_ready = CLASSIFIER.get().is_some();
    let ner_installed = ner_ready || resolve_existing_ner_dir(&app).is_some();
    let gemma_installed = gemma_ready || resolve_existing_gemma_dir(&app).is_some();
    Ok(ModelsStatus {
        ner_installed,
        gemma_installed,
        ner_ready,
        gemma_ready,
    })
}

struct QueuedDownload {
    step: usize,
    is_gemma: bool,
    model_title: &'static str,
    local_name: &'static str,
    url: String,
    dest: PathBuf,
    estimated_bytes: u64,
}

#[tauri::command]
async fn ensure_all_models(app: AppHandle) -> Result<ModelsStatus, String> {
    use futures_util::StreamExt;
    use std::io::Write;
    use std::time::{Duration, Instant};

    let ner_dir_existing = resolve_existing_ner_dir(&app);
    let gemma_dir_existing = resolve_existing_gemma_dir(&app);

    let need_ner_download = SESSION.get().is_none() && ner_dir_existing.is_none();
    let need_gemma_download = CLASSIFIER.get().is_none() && gemma_dir_existing.is_none();

    let total_steps = match (need_ner_download, need_gemma_download) {
        (true, true) => 2,
        (true, false) | (false, true) => 1,
        (false, false) => 2,
    };

    let mut queue: Vec<QueuedDownload> = Vec::new();
    let data_ner_dir = models_dir(&app)?;
    let data_gemma_dir = gemma_dir(&app)?;

    if need_ner_download {
        for &(remote_rel, local_name, est) in MODEL_FILES {
            let dest = data_ner_dir.join(local_name);
            if !dest.exists() {
                queue.push(QueuedDownload {
                    step: 1,
                    is_gemma: false,
                    model_title: "språkmodell (Svensk BERT NER)",
                    local_name,
                    url: format!("{HF_BASE_NER}/{remote_rel}"),
                    dest,
                    estimated_bytes: est,
                });
            }
        }
    }

    if need_gemma_download {
        let step_num = if need_ner_download { 2 } else { 1 };
        for &(remote_rel, local_name, est) in GEMMA_FILES {
            let dest = data_gemma_dir.join(local_name);
            if !dest.exists() {
                queue.push(QueuedDownload {
                    step: step_num,
                    is_gemma: true,
                    model_title: "kontextmodell (EmbeddingGemma 2)",
                    local_name,
                    url: format!("{HF_BASE_GEMMA}/{remote_rel}"),
                    dest,
                    estimated_bytes: est,
                });
            }
        }
    }

    let mut planned_total: u64 = queue.iter().map(|q| q.estimated_bytes).sum();
    let mut completed_bytes: u64 = 0;

    if !queue.is_empty() {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1200))
            .build()
            .map_err(|e| format!("HTTP-klient-fel: {e}"))?;

        for item in queue {
            let ner_done_now = !need_ner_download || item.is_gemma;
            let label = format!("Hämtar {}…", item.model_title);

            let _ = app.emit(
                "setup_progress",
                SetupProgress {
                    phase: "downloading".into(),
                    step: item.step,
                    total_steps,
                    ner_done: ner_done_now,
                    gemma_done: false,
                    label: label.clone(),
                    downloaded_bytes: completed_bytes,
                    total_bytes: planned_total,
                },
            );

            let resp = client
                .get(&item.url)
                .send()
                .await
                .map_err(|e| format!("Nätverksfel vid hämtning av {}: {e}", item.local_name))?;

            if !resp.status().is_success() {
                return Err(format!(
                    "Serverfel (HTTP {}) vid hämtning av {}",
                    resp.status(),
                    item.local_name
                ));
            }

            let actual_total = resp.content_length().unwrap_or(item.estimated_bytes);
            if actual_total > 0 {
                planned_total = planned_total
                    .saturating_sub(item.estimated_bytes)
                    .saturating_add(actual_total);
            }

            let tmp = item.dest.with_extension("part");
            let mut file = std::fs::File::create(&tmp)
                .map_err(|e| format!("Kan inte skapa temporär fil för {}: {e}", item.local_name))?;
            let mut file_downloaded: u64 = 0;
            let mut stream = resp.bytes_stream();
            let mut last_emit = Instant::now()
                .checked_sub(Duration::from_millis(100))
                .unwrap_or_else(Instant::now);

            while let Some(chunk) = stream.next().await {
                let chunk =
                    chunk.map_err(|e| format!("Läsfel vid hämtning av {}: {e}", item.local_name))?;
                file.write_all(&chunk)
                    .map_err(|e| format!("Skrivfel för {}: {e}", item.local_name))?;
                file_downloaded += chunk.len() as u64;

                if last_emit.elapsed() >= Duration::from_millis(40) {
                    last_emit = Instant::now();
                    let _ = app.emit(
                        "setup_progress",
                        SetupProgress {
                            phase: "downloading".into(),
                            step: item.step,
                            total_steps,
                            ner_done: ner_done_now,
                            gemma_done: false,
                            label: label.clone(),
                            downloaded_bytes: completed_bytes + file_downloaded,
                            total_bytes: planned_total.max(completed_bytes + file_downloaded),
                        },
                    );
                }
            }

            drop(file);
            if std::fs::rename(&tmp, &item.dest).is_err() {
                std::fs::copy(&tmp, &item.dest)
                    .map_err(|e| format!("Kunde inte spara {}: {e}", item.local_name))?;
                let _ = std::fs::remove_file(&tmp);
            }

            completed_bytes += file_downloaded;
            planned_total = planned_total.max(completed_bytes);
            let _ = app.emit(
                "setup_progress",
                SetupProgress {
                    phase: "downloading".into(),
                    step: item.step,
                    total_steps,
                    ner_done: ner_done_now,
                    gemma_done: false,
                    label,
                    downloaded_bytes: completed_bytes,
                    total_bytes: planned_total,
                },
            );
        }
    }

    // Ladda in BERT NER i minnet om den inte redan är laddad
    if SESSION.get().is_none() {
        let ner_dir = resolve_existing_ner_dir(&app).unwrap_or(data_ner_dir);
        let _ = app.emit(
            "setup_progress",
            SetupProgress {
                phase: "loading".into(),
                step: 1,
                total_steps,
                ner_done: false,
                gemma_done: CLASSIFIER.get().is_some(),
                label: "Initierar språkmodell (Svensk BERT NER) i minnet…".into(),
                downloaded_bytes: planned_total,
                total_bytes: planned_total,
            },
        );
        let app_c = app.clone();
        tauri::async_runtime::spawn_blocking(move || load_and_store(&ner_dir, &app_c))
            .await
            .map_err(|e| format!("spawn_blocking-fel (BERT): {e}"))??;
        let _ = app.emit("model_ready", ());
    }

    // Ladda in EmbeddingGemma 2 i minnet om den inte redan är laddad
    if CLASSIFIER.get().is_none() {
        let g_dir = resolve_existing_gemma_dir(&app).unwrap_or(data_gemma_dir);
        let _ = app.emit(
            "setup_progress",
            SetupProgress {
                phase: "loading".into(),
                step: total_steps,
                total_steps,
                ner_done: true,
                gemma_done: false,
                label: "Initierar kontextmodell (EmbeddingGemma 2) & säkerhetsankare…".into(),
                downloaded_bytes: planned_total,
                total_bytes: planned_total,
            },
        );
        let app_c = app.clone();
        tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
            let _ = app_c.emit(
                "gemma_status",
                ModelStatus {
                    message: "Initierar EmbeddingGemma 2 & beräknar klassificeringsankare…".into(),
                },
            );
            let clf = EmbeddingClassifier::load(&g_dir)?;
            let _ = CLASSIFIER.set(clf);
            Ok(())
        })
        .await
        .map_err(|e| format!("spawn_blocking-fel (EmbeddingGemma 2): {e}"))??;
        let _ = app.emit("gemma_ready", ());
    }

    let _ = app.emit(
        "setup_progress",
        SetupProgress {
            phase: "ready".into(),
            step: total_steps,
            total_steps,
            ner_done: true,
            gemma_done: true,
            label: "Alla lokala AI-modeller är redo.".into(),
            downloaded_bytes: planned_total,
            total_bytes: planned_total,
        },
    );

    Ok(ModelsStatus {
        ner_installed: true,
        gemma_installed: true,
        ner_ready: SESSION.get().is_some(),
        gemma_ready: CLASSIFIER.get().is_some(),
    })
}

/// Importera modellfiler offline från en lokal mapp (t.ex. utdelad av IT-avdelning)
#[tauri::command]
async fn import_models_from_dir(src_dir: String, app: AppHandle) -> Result<ModelsStatus, String> {
    let root = PathBuf::from(&src_dir);
    if !root.exists() {
        return Err("Den valda mappen finns inte.".into());
    }

    let dest_ner = models_dir(&app)?;
    let dest_gemma = gemma_dir(&app)?;

    // Sök i vald mapp samt undermappar `models` och `embeddinggemma2`
    let search_dirs = [
        root.clone(),
        root.join("models"),
        root.join("embeddinggemma2"),
        root.join("models").join("embeddinggemma2"),
    ];

    let mut ner_found: Vec<(PathBuf, &'static str)> = Vec::new();
    for &(remote_rel, local_name, _) in MODEL_FILES {
        for sdir in [&root, &root.join("models")] {
            if let Some(p) = find_source_model_file(sdir, remote_rel, local_name) {
                ner_found.push((p, local_name));
                break;
            }
        }
    }

    let mut gemma_found: Vec<(PathBuf, &'static str)> = Vec::new();
    for &(remote_rel, local_name, _) in GEMMA_FILES {
        for sdir in &search_dirs {
            if let Some(p) = find_source_model_file(sdir, remote_rel, local_name) {
                // Undvik att blanda ihop BERTs tokenizer.json (1.1 MB) med Gemmas tokenizer.json (31 MB)
                if local_name == "tokenizer.json" {
                    if let Ok(meta) = std::fs::metadata(&p) {
                        if meta.len() < 5_000_000 {
                            continue;
                        }
                    }
                }
                gemma_found.push((p, local_name));
                break;
            }
        }
    }

    if ner_found.len() < MODEL_FILES.len() && gemma_found.len() < GEMMA_FILES.len() {
        return Err(
            "Hittade inte kompletta modellfiler i den valda mappen. Mappen behöver innehålla BERT NER (model_quantized.onnx, tokenizer.json m.fl.) och/eller undermappen embeddinggemma2 (model_q4.onnx, model_q4.onnx_data, tokenizer.json).".into(),
        );
    }

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        if ner_found.len() == MODEL_FILES.len() {
            let _ = app_c.emit(
                "setup_progress",
                SetupProgress {
                    phase: "copying".into(),
                    step: 1,
                    total_steps: 2,
                    ner_done: false,
                    gemma_done: false,
                    label: "Kopierar språkmodell (BERT NER) från lokal mapp…".into(),
                    downloaded_bytes: 0,
                    total_bytes: 0,
                },
            );
            for (src_path, local_name) in ner_found {
                let dst = dest_ner.join(local_name);
                if src_path != dst {
                    std::fs::copy(&src_path, &dst)
                        .map_err(|e| format!("Kunde inte kopiera {local_name}: {e}"))?;
                }
            }
        }
        if gemma_found.len() == GEMMA_FILES.len() {
            let _ = app_c.emit(
                "setup_progress",
                SetupProgress {
                    phase: "copying".into(),
                    step: 2,
                    total_steps: 2,
                    ner_done: true,
                    gemma_done: false,
                    label: "Kopierar kontextmodell (EmbeddingGemma 2) från lokal mapp…".into(),
                    downloaded_bytes: 0,
                    total_bytes: 0,
                },
            );
            for (src_path, local_name) in gemma_found {
                let dst = dest_gemma.join(local_name);
                if src_path != dst {
                    std::fs::copy(&src_path, &dst)
                        .map_err(|e| format!("Kunde inte kopiera {local_name}: {e}"))?;
                }
            }
        }
        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel (import_models): {e}"))??;

    // Om båda nu finns på plats, ladda in dem direkt; annars ladda det som finns
    if resolve_existing_ner_dir(&app).is_some() && SESSION.get().is_none() {
        let dir = resolve_existing_ner_dir(&app).unwrap();
        let app_c = app.clone();
        tauri::async_runtime::spawn_blocking(move || load_and_store(&dir, &app_c))
            .await
            .map_err(|e| format!("spawn_blocking-fel: {e}"))??;
        let _ = app.emit("model_ready", ());
    }
    if resolve_existing_gemma_dir(&app).is_some() && CLASSIFIER.get().is_none() {
        let dir = resolve_existing_gemma_dir(&app).unwrap();
        tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
            let clf = EmbeddingClassifier::load(&dir)?;
            let _ = CLASSIFIER.set(clf);
            Ok(())
        })
        .await
        .map_err(|e| format!("spawn_blocking-fel: {e}"))??;
        let _ = app.emit("gemma_ready", ());
    }

    check_models_status(app)
}

// ── KBLab BERT NER Kommandon ──────────────────────────────────────────────────

#[tauri::command]
fn models_exist(app: AppHandle) -> Result<bool, String> {
    Ok(resolve_existing_ner_dir(&app).is_some())
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

    for &(remote_rel, local_name, _) in MODEL_FILES {
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
    for &(remote_rel, local_name, _) in MODEL_FILES {
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

    let dir = resolve_existing_ner_dir(&app).ok_or_else(|| {
        "Språkmodellen saknas på disk. Starta om appen för att hämta modellerna.".to_string()
    })?;

    let app_c = app.clone();
    tauri::async_runtime::spawn_blocking(move || load_and_store(&dir, &app_c))
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
    let model_path = find_source_model_file(dir, "onnx/model_quantized.onnx", "model_quantized.onnx")
        .unwrap_or_else(|| dir.join("model_quantized.onnx"));
    let tok_path = find_source_model_file(dir, "tokenizer.json", "tokenizer.json")
        .unwrap_or_else(|| dir.join("tokenizer.json"));
    let model = ner::load_model(&model_path)?;
    let tokenizer = ner::load_tokenizer(&tok_path)?;
    let _ = SESSION.set(model);
    let _ = TOKENIZER.set(tokenizer);
    Ok(())
}

// ── EmbeddingGemma 2 Kommandon (Lager 3 AI-Gatekeeper) ────────────────────────

#[tauri::command]
fn gemma_exists(app: AppHandle) -> Result<bool, String> {
    Ok(resolve_existing_gemma_dir(&app).is_some())
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

    for &(remote_rel, local_name, _) in GEMMA_FILES {
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

    let dir = resolve_existing_gemma_dir(&app).ok_or_else(|| {
        "EmbeddingGemma 2 saknas på disk. Ladda ner modellen först.".to_string()
    })?;

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

#[allow(clippy::too_many_arguments)]
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
        "Roll/Titel",
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

#[tauri::command]
fn write_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content.as_bytes()).map_err(|e| e.to_string())
}

fn percent_decode_utf8(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(val) = u8::from_str_radix(hex, 16) {
                    out.push(val);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

#[tauri::command]
async fn import_document_bytes(
    request: tauri::ipc::Request<'_>,
) -> Result<documents::ImportedDocument, String> {
    let file_name = request
        .headers()
        .get("x-file-name")
        .and_then(|v| v.to_str().ok())
        .map(percent_decode_utf8)
        .unwrap_or_else(|| "dokument.txt".to_string());

    let bytes = match request.body() {
        tauri::ipc::InvokeBody::Raw(raw) => raw.clone(),
        tauri::ipc::InvokeBody::Json(_) => {
            return Err("Förväntade binär filpayload vid dokumentimport.".into());
        }
    };

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

#[tauri::command]
async fn export_masked_xlsx(
    path: String,
    original_bytes: Vec<u8>,
    replacements: Vec<documents::ReplacementPair>,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let masked_bytes = documents::export_masked_xlsx_bytes(&original_bytes, &replacements)?;
        std::fs::write(&path, &masked_bytes).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel (export_xlsx): {e}"))?
}

// ── Lager 4 (Judge / Helhetsbedömning — enbart för Klassificera information) ─

fn judge_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = models_dir(app)?.join("judge");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Kan inte skapa judge-katalog: {e}"))?;
    Ok(dir)
}

fn is_valid_judge_gguf(path: &Path) -> bool {
    path.is_file()
        && std::fs::metadata(path)
            .map(|m| m.len() >= 100_000_000)
            .unwrap_or(false)
}

fn resolve_existing_judge_file(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(jdir) = judge_dir(app) {
        let p = jdir.join(info_judge::JUDGE_FILENAME);
        if is_valid_judge_gguf(&p) {
            return Some(p);
        }
    }
    if let Ok(mdir) = models_dir(app) {
        let p = mdir.join(info_judge::JUDGE_FILENAME);
        if is_valid_judge_gguf(&p) {
            return Some(p);
        }
    }
    for cand in candidate_bundled_model_dirs(app) {
        let sub = cand.join("judge").join(info_judge::JUDGE_FILENAME);
        if is_valid_judge_gguf(&sub) {
            return Some(sub);
        }
        let direct = cand.join(info_judge::JUDGE_FILENAME);
        if is_valid_judge_gguf(&direct) {
            return Some(direct);
        }
    }
    None
}

fn build_judge_status(app: &AppHandle) -> info_judge::JudgeModelStatus {
    let loaded = JUDGE_ENGINE
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false);
    let existing = resolve_existing_judge_file(app);
    let file_size_bytes = existing
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    info_judge::JudgeModelStatus {
        installed: existing.is_some(),
        loaded,
        file_size_bytes,
        expected_bytes: info_judge::JUDGE_EXPECTED_BYTES,
        model_name: info_judge::JUDGE_MODEL_LABEL.to_string(),
    }
}

#[tauri::command]
fn check_judge_status(app: AppHandle) -> Result<info_judge::JudgeModelStatus, String> {
    Ok(build_judge_status(&app))
}

#[tauri::command]
fn cancel_judge_download() -> Result<(), String> {
    JUDGE_CANCEL.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
async fn ensure_judge_loaded(app: AppHandle) -> Result<info_judge::JudgeModelStatus, String> {
    let already_loaded = JUDGE_ENGINE
        .lock()
        .map_err(|e| format!("Låsfel för Lager 4: {e}"))?
        .is_some();
    if already_loaded {
        return Ok(build_judge_status(&app));
    }

    let path = resolve_existing_judge_file(&app).ok_or_else(|| {
        "Lager 4-modellen (Gemma 4 E2B) är inte nedladdad på datorn ännu.".to_string()
    })?;

    let _ = app.emit(
        "judge_download_progress",
        JudgeDownloadProgress {
            phase: "loading".into(),
            label: "Laddar in Lager 4 (Gemma 4 E2B) i arbetsminnet…".into(),
            downloaded_bytes: info_judge::JUDGE_EXPECTED_BYTES,
            total_bytes: info_judge::JUDGE_EXPECTED_BYTES,
        },
    );

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let mut guard = JUDGE_ENGINE
            .lock()
            .map_err(|e| format!("Låsfel för Lager 4: {e}"))?;
        if guard.is_none() {
            load_and_store_judge_engine(&mut guard, &path)?;
        }
        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking-fel (ensure_judge_loaded): {e}"))??;

    let _ = app.emit(
        "judge_download_progress",
        JudgeDownloadProgress {
            phase: "ready".into(),
            label: "Lager 4 (Judge) är redo.".into(),
            downloaded_bytes: info_judge::JUDGE_EXPECTED_BYTES,
            total_bytes: info_judge::JUDGE_EXPECTED_BYTES,
        },
    );

    Ok(build_judge_status(&app))
}

#[tauri::command]
async fn download_and_load_judge(app: AppHandle) -> Result<info_judge::JudgeModelStatus, String> {
    use futures_util::StreamExt;
    use std::io::Write;
    use std::time::{Duration, Instant};

    JUDGE_CANCEL.store(false, Ordering::SeqCst);

    if resolve_existing_judge_file(&app).is_none() {
        let jdir = judge_dir(&app)?;
        let dest = jdir.join(info_judge::JUDGE_FILENAME);
        let tmp = jdir.join(format!("{}.part", info_judge::JUDGE_FILENAME));

        let _ = app.emit(
            "judge_download_progress",
            JudgeDownloadProgress {
                phase: "downloading".into(),
                label: "Ansluter och hämtar Lager 4 (Gemma 4 E2B)…".into(),
                downloaded_bytes: 0,
                total_bytes: info_judge::JUDGE_EXPECTED_BYTES,
            },
        );

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(7200))
            .build()
            .map_err(|e| format!("HTTP-klient-fel vid hämtning av Lager 4: {e}"))?;

        let resp = client
            .get(info_judge::JUDGE_HF_URL)
            .send()
            .await
            .map_err(|e| format!("Nätverksfel vid hämtning av Lager 4: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Serverfel (HTTP {}) vid hämtning av Lager 4-modellen.",
                resp.status()
            ));
        }

        let total_bytes = resp
            .content_length()
            .unwrap_or(info_judge::JUDGE_EXPECTED_BYTES)
            .max(1);

        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("Kunde inte skapa temporär fil på disk: {e}"))?;
        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();
        let mut last_emit = Instant::now()
            .checked_sub(Duration::from_millis(120))
            .unwrap_or_else(Instant::now);

        while let Some(chunk) = stream.next().await {
            if JUDGE_CANCEL.load(Ordering::SeqCst) {
                drop(file);
                let _ = std::fs::remove_file(&tmp);
                let _ = app.emit(
                    "judge_download_progress",
                    JudgeDownloadProgress {
                        phase: "cancelled".into(),
                        label: "Nedladdningen avbröts.".into(),
                        downloaded_bytes: 0,
                        total_bytes,
                    },
                );
                return Err("Nedladdningen avbröts.".to_string());
            }

            let chunk =
                chunk.map_err(|e| format!("Läsfel under hämtning av Lager 4: {e}"))?;
            file.write_all(&chunk).map_err(|e| {
                format!(
                    "Skrivfel till disk (kontrollera ledigt lagringsutrymme ~2,6 GB): {e}"
                )
            })?;
            downloaded += chunk.len() as u64;

            if last_emit.elapsed() >= Duration::from_millis(60) {
                last_emit = Instant::now();
                let _ = app.emit(
                    "judge_download_progress",
                    JudgeDownloadProgress {
                        phase: "downloading".into(),
                        label: "Hämtar Lager 4 (Gemma 4 E2B)…".into(),
                        downloaded_bytes: downloaded,
                        total_bytes: total_bytes.max(downloaded),
                    },
                );
            }
        }

        drop(file);
        if std::fs::rename(&tmp, &dest).is_err() {
            std::fs::copy(&tmp, &dest)
                .map_err(|e| format!("Kunde inte spara modellfilen för Lager 4: {e}"))?;
            let _ = std::fs::remove_file(&tmp);
        }
    }

    ensure_judge_loaded(app).await
}

#[tauri::command]
async fn import_judge_from_dir(
    src_dir: String,
    app: AppHandle,
) -> Result<info_judge::JudgeModelStatus, String> {
    let root = PathBuf::from(&src_dir);
    if !root.exists() {
        return Err("Den valda mappen eller filen finns inte.".into());
    }

    let found_gguf: Option<PathBuf> = if root.is_file() {
        Some(root)
    } else {
        let search_dirs = [
            root.clone(),
            root.join("judge"),
            root.join("models"),
            root.join("models").join("judge"),
        ];
        let mut candidate: Option<PathBuf> = None;
        for d in &search_dirs {
            let exact = d.join(info_judge::JUDGE_FILENAME);
            if is_valid_judge_gguf(&exact) {
                candidate = Some(exact);
                break;
            }
        }
        if candidate.is_none() {
            for d in &search_dirs {
                if let Ok(entries) = std::fs::read_dir(d) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        let is_gguf = p
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|e| e.eq_ignore_ascii_case("gguf"))
                            .unwrap_or(false);
                        if is_gguf && is_valid_judge_gguf(&p) {
                            candidate = Some(p);
                            break;
                        }
                    }
                }
                if candidate.is_some() {
                    break;
                }
            }
        }
        candidate
    };

    let src_file = found_gguf.ok_or_else(|| {
        format!(
            "Hittade ingen giltig GGUF-modellfil (t.ex. {}) i den valda mappen.",
            info_judge::JUDGE_FILENAME
        )
    })?;

    let dest = judge_dir(&app)?.join(info_judge::JUDGE_FILENAME);
    if src_file != dest {
        let src_clone = src_file.clone();
        let dest_clone = dest.clone();
        let _ = app.emit(
            "judge_download_progress",
            JudgeDownloadProgress {
                phase: "loading".into(),
                label: "Kopierar lokal Lager 4-modellfil…".into(),
                downloaded_bytes: 0,
                total_bytes: info_judge::JUDGE_EXPECTED_BYTES,
            },
        );
        tauri::async_runtime::spawn_blocking(move || std::fs::copy(&src_clone, &dest_clone))
            .await
            .map_err(|e| format!("spawn_blocking-fel (kopiera GGUF): {e}"))?
            .map_err(|e| format!("Kunde inte kopiera GGUF-filen: {e}"))?;
    }

    ensure_judge_loaded(app).await
}

#[tauri::command]
async fn delete_judge_model(app: AppHandle) -> Result<info_judge::JudgeModelStatus, String> {
    JUDGE_CANCEL.store(true, Ordering::SeqCst);

    // Frigör LlamaModel ur RAM först så att Windows släpper filhandtaget (mmap) innan borttagning!
    {
        let mut guard = JUDGE_ENGINE
            .lock()
            .map_err(|e| format!("Låsfel vid avinstallation av Lager 4: {e}"))?;
        *guard = None;
    }

    if let Ok(jdir) = judge_dir(&app) {
        let main_file = jdir.join(info_judge::JUDGE_FILENAME);
        let part_file = jdir.join(format!("{}.part", info_judge::JUDGE_FILENAME));
        if main_file.exists() {
            std::fs::remove_file(&main_file)
                .map_err(|e| format!("Kunde inte ta bort modellfilen på disk: {e}"))?;
        }
        if part_file.exists() {
            let _ = std::fs::remove_file(&part_file);
        }
    }
    if let Ok(mdir) = models_dir(&app) {
        let alt_file = mdir.join(info_judge::JUDGE_FILENAME);
        if alt_file.exists() {
            let _ = std::fs::remove_file(&alt_file);
        }
    }

    Ok(build_judge_status(&app))
}

#[tauri::command]
async fn classify_information_document(
    text: String,
    use_judge: Option<bool>,
    app: AppHandle,
) -> Result<info_classification::InfoClassificationReport, String> {
    let run_judge = use_judge.unwrap_or(false);
    tauri::async_runtime::spawn_blocking(
        move || -> Result<info_classification::InfoClassificationReport, String> {
            let _ = app.emit(
                "info_classify_phase",
                InfoClassifyPhase {
                    phase: "layer1_3".into(),
                    message: "Steg 1: Granskar struktur och enskilda stycken (Lager 1–3)…".into(),
                },
            );

            let regex_labels: Vec<String> = vec![
                "Personnummer".into(),
                "E-post".into(),
                "Telefonnummer".into(),
                "Konto/Giro".into(),
                "Kortnummer".into(),
                "Hemlighet/Nyckel".into(),
            ];
            let regex_matches = run_regex(&text, &regex_labels);

            let structural_matches =
                if let (Some(model), Some(tokenizer)) = (SESSION.get(), TOKENIZER.get()) {
                    let ner_labels = vec!["Person".to_string()];
                    let ner_matches = ner::run_ner(
                        &text,
                        &ner_labels,
                        0.72,
                        model,
                        tokenizer,
                        |chunk, total| {
                            let _ = app.emit("ner_progress", NerProgress { chunk, total });
                        },
                    );
                    merge_regex_and_ner(regex_matches, ner_matches)
                } else {
                    regex_matches
                };

            let mut report = info_classification::classify_information_text(
                &text,
                &structural_matches,
                CLASSIFIER.get(),
                |chunk, total| {
                    let _ = app.emit("gemma_scan_progress", NerProgress { chunk, total });
                },
            );

            if run_judge {
                let _ = app.emit(
                    "info_classify_phase",
                    InfoClassifyPhase {
                        phase: "layer4".into(),
                        message: "Steg 2: Lager 4 (Judge · Gemma 4 E2B) bedömer dokumentets helhet och sammanhang…".into(),
                    },
                );

                let mut guard = JUDGE_ENGINE
                    .lock()
                    .map_err(|e| format!("Låsfel för Lager 4 (Judge): {e}"))?;

                if guard.is_none() {
                    let path = resolve_existing_judge_file(&app).ok_or_else(|| {
                        "Lager 4-modellen (Gemma 4 E2B) är inte installerad ännu. Aktivera och ladda ner den via knappen Helhetsbedömning (Lager 4).".to_string()
                    })?;
                    load_and_store_judge_engine(&mut guard, &path)?;
                }

                if let Some(engine) = guard.as_mut() {
                    let (raw_judge, judge_ms) =
                        engine.assess_document(&text, &report, &structural_matches)?;
                    info_judge::apply_judge_to_report(&mut report, raw_judge, judge_ms);
                }
            }

            Ok(report)
        },
    )
    .await
    .map_err(|e| format!("spawn_blocking-fel (classify_information_document): {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            check_models_status,
            ensure_all_models,
            import_models_from_dir,
            models_exist,
            download_model,
            load_model,
            load_model_from_dir,
            gemma_exists,
            download_gemma,
            load_gemma,
            check_judge_status,
            ensure_judge_loaded,
            download_and_load_judge,
            cancel_judge_download,
            import_judge_from_dir,
            delete_judge_model,
            classify_document,
            classify_information_document,
            analyze_text,
            write_file,
            import_document_bytes,
            export_masked_docx,
            export_masked_xlsx
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app, event| {
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. }
                    | tauri::RunEvent::Exit
                    | tauri::RunEvent::WindowEvent {
                        event: tauri::WindowEvent::Destroyed,
                        ..
                    }
            ) {
                shutdown_judge_engine();
            }
        });
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn judge_metal_exit_does_not_abort() {
        let Ok(home) = std::env::var("HOME") else {
            return;
        };
        let gguf = Path::new(&home)
            .join("Library/Application Support/com.patriksvensk.maskera/models/judge")
            .join(info_judge::JUDGE_FILENAME);
        if !gguf.exists() {
            return;
        }

        if std::env::var("MASKERA_SUBPROC_METAL_EXIT").as_deref() == Ok("1") {
            let mut guard = JUDGE_ENGINE.lock().unwrap();
            load_and_store_judge_engine(&mut guard, &gguf).expect("load Judge on Metal");
            drop(guard);
            // Simulera exakt vad tao::EventLoop::run gör vid avslut om statics ligger kvar
            std::process::exit(0);
        }

        let current_exe = std::env::current_exe().expect("test binary path");
        let status = std::process::Command::new(current_exe)
            .env("MASKERA_SUBPROC_METAL_EXIT", "1")
            .arg("--exact")
            .arg("tests::judge_metal_exit_does_not_abort")
            .arg("--nocapture")
            .status()
            .expect("spawn child process");
        assert!(
            status.success(),
            "Processavslut med laddad Metal-modell kraschade (status: {status})"
        );
    }
}
