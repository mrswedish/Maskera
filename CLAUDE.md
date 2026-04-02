# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Maskera** is a Swedish PII-masking desktop application. Users upload `.txt` files, the app highlights sensitive entities, and exports a redacted version. The app is built with Tauri 2 (Rust shell) + React/TypeScript frontend, with a Python FastAPI backend (`masking_engine.py`) that runs as a Tauri sidecar binary.

## Commands

### Frontend / Tauri

```bash
npm run dev          # Start Vite dev server only (no Tauri)
npm run tauri dev    # Start full Tauri app with hot reload (starts both Vite + Tauri)
npm run tauri build  # Bundle production app (calls tsc && vite build internally)
npm run build        # TypeScript compile + Vite build (frontend only)
```

### Python Backend (masking_engine)

```bash
# Activate venv first
source venv/bin/activate

# Run the engine manually (dev/test)
python masking_engine.py

# Build into a standalone binary (required for Tauri sidecar)
pip install pyinstaller
pyinstaller --onefile masking_engine.py
# Move the resulting binary to src-tauri/bin/masking_engine-<target-triple>
```

The engine listens on `http://127.0.0.1:8594`. It prints `"redo"` to stdout when the AI model is loaded — the frontend polls for this string to unlock the "Granska Text" button.

## Architecture

### Two-process design

```
Tauri shell (Rust)
  └── Spawns sidecar: bin/masking_engine  ←── PyInstaller-bundled FastAPI server
           │
           └── POST http://127.0.0.1:8594/analyze
                    │
                    ├── Regex pass (personnummer, e-post, telefon) — always runs
                    └── KBLab BERT NER (PER/ORG/LOC/MISC) — runs if entities selected
```

The Tauri frontend communicates with the Python engine **entirely over localhost HTTP** — there are no Tauri `invoke` commands used for the core masking logic. The Rust `lib.rs` currently only exposes a stub `greet` command.

### Key files

| File | Role |
|---|---|
| [src/App.tsx](src/App.tsx) | Entire UI — file upload, entity selection, text viewer with highlights, save |
| [masking_engine.py](masking_engine.py) | FastAPI server: regex + KBLab BERT NER, returns `List[Match]` |
| [src-tauri/src/lib.rs](src-tauri/src/lib.rs) | Tauri app bootstrap, registers shell plugin |
| [src-tauri/tauri.conf.json](src-tauri/tauri.conf.json) | App config, registers `bin/masking_engine` as external binary |

### Entity label mapping

The UI uses Swedish labels; the backend maps them to KBLab NER tags:

| UI label | KBLab tag |
|---|---|
| Person | PER |
| Organisation | ORG |
| Plats | LOC |
| Övrigt | MISC |

Regex-detected entities (`Personnummer`, `E-post`, `Telefonnummer`) always run regardless of checkbox state.

### BERT token limit handling

The engine chunks input text at ≤1500 characters, breaking at sentence/word boundaries, to stay within BERT's 512-token limit. Chunk-relative character offsets are adjusted to global positions before returning.

### Sidecar binary naming

Tauri requires platform-specific suffixes on sidecar binaries. The binary must be placed at `src-tauri/bin/masking_engine-<target-triple>` (e.g. `masking_engine-aarch64-apple-darwin` on Apple Silicon). The `tauri.conf.json` references it as `bin/masking_engine` without the suffix.
