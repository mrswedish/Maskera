# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

**Maskera** is a local Swedish PII-masking desktop application for Windows and macOS. Users upload `.txt` files, the app detects sensitive entities via deterministic Rust regular expressions and Swedish BERT NER (`KBLab/bert-base-swedish-cased-ner` quantized INT8 ONNX), highlights matches in the UI, and exports a redacted text file plus a translation table (`.json` / `.txt`).

## Commands

```bash
npm install                                           # Install frontend dependencies
npm run dev                                           # Start Vite dev server only
npm run tauri dev                                     # Start full Tauri 2 desktop app with hot reload
npm run build                                         # TypeScript check + Vite frontend build
npm run tauri build                                   # Build production installer / portable binary
cargo test --manifest-path src-tauri/Cargo.toml       # Run Rust unit tests
```

## Architecture

```
Tauri 2 shell (Rust) + React 19 / TypeScript WebView
  └── Tauri invoke("analyze_text")
        ├── Precompiled Rust Regex pass (< 1 ms, runs immediately even without AI model)
        │     └── Personnummer/Samordningsnummer, Org.nummer, E-post, Telefon, Konto/Giro, IPv4, Kortnummer (Luhn)
        └── Microsoft ONNX Runtime (`ort`, statically linked)
              ├── Downloads `onnx/model_quantized.onnx` + tokenizer from HuggingFace on first launch
              ├── Caches in OS app-data `models/` directory
              └── Multi-threaded SIMD + length-sorted mini-batching (`BATCH_SIZE = 8`) with smart paragraph/sentence chunking
```

### Key files

| File | Role |
|---|---|
| `src/App.tsx` | Main UI — file upload / drag-and-drop, entity selection, highlighted viewer, manual additions, export |
| `src/translationUtils.ts` | Consistent entity-to-placeholder mapping (`Person A`, `Personnummer 1`, etc.) |
| `src-tauri/src/lib.rs` | Tauri commands, HuggingFace streaming model downloader, precompiled Swedish PII regex engine, overlap resolution |
| `src-tauri/src/ner.rs` | `ort` ONNX Runtime session, zero-copy paragraph/sentence chunker, parallel batch tokenization, batched inference, word-level aggregation |
| `.github/workflows/build.yml` | GitHub Actions Windows build producing both NSIS setup installer and single-file `Maskera-portabel.exe` |
