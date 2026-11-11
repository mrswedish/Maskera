# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

**Maskera** is a 100% local Swedish PII-masking, AI-gatekeeper, and document-anonymization desktop application for Windows and macOS (`Maskera-portabel.exe` single-file portable + NSIS installer):
1. **Input & Document Import:** Users can paste text from clipboard (`Ctrl+V`), type directly, or open/drag-and-drop Word (`.docx`, `.odt`), Excel (`.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.ods`), PDF (`.pdf`), and text/CSV files (`.txt`, `.csv`, `.md`, `.json`, `.xml`) with zero external dependencies (no Office or LibreOffice required).
2. **3-Layer Local Analysis:**
   - **Layer 1 (Structure & Secrets, < 1 ms):** Precompiled Rust regular expressions with Luhn and date validation for Swedish `Personnummer`/`Samordningsnummer`, `Org.nummer`, email, phone, IBAN/Bankgiro/Plusgiro, IPv4, payment cards, 16–22 digit service/SITHS cards, HSA-ID, internal intranet URLs, and API keys/JWT/PEM secrets.
   - **Layer 2 (Named Entities via `ort`):** `KBLab/bert-base-swedish-cased-ner` (quantized INT8 ONNX) with multi-threaded SIMD and length-sorted mini-batching (`BATCH_SIZE = 8`).
   - **Layer 3 (Semantic AI Gatekeeper via `EmbeddingGemma 2`):** `onnx-community/embeddinggemma-300m-ONNX` (`model_q4.onnx`, 768-dim L2-normalized embeddings) classifying document sections across privacy (GDPR Art. 9/10 health, crime/social services, SITHS/eID/intranet routines, HR), business secrets, and IT security, with inline paragraph highlighting in the review view.
3. **Export & Local AI Response Unmasking:** One-click copy of masked text, format-preserving `.docx` (split-run-safe) and `.xlsx` export, and instant local unmasking of AI responses (or masked `.docx`/`.xlsx` files) using the in-memory translation table.

## Commands

```bash
npm install                                           # Install frontend dependencies
npm run dev                                           # Start Vite dev server only
npm run tauri dev                                     # Start full Tauri 2 desktop app with hot reload
npm run build                                         # TypeScript check + Vite frontend build
npm run tauri build                                   # Build production installer / portable binary
cargo test --manifest-path src-tauri/Cargo.toml       # Run Rust unit tests (8 tests)
```

## Architecture

```
Tauri 2 shell (Rust) + React 19 / TypeScript WebView (WCAG 2.2 AA, Light/Dark/System theme)
  ├── Tauri invoke("import_document_bytes" / "export_masked_docx" / "export_masked_xlsx")
  │     └── Pure-Rust portable parsers (`zip` + `quick-xml`, `calamine`, `pdf-extract`)
  ├── Tauri invoke("analyze_text")
  │     ├── Layer 1: Precompiled Rust Regex pass (< 1 ms)
  │     └── Layer 2: Swedish BERT NER via Microsoft ONNX Runtime (`ort`, statically linked)
  └── Tauri invoke("classify_document")
        └── Layer 3: EmbeddingGemma 2 (768d q4 ONNX) semantic paragraph classifier + secret check
```

### Key files

| File | Role |
|---|---|
| `src/App.tsx` | Main UI — toolbar, Light/Dark/System theme switcher, document import/export, Gatekeeper banner, inline entity & paragraph highlighting, AI response & document unmasking |
| `src/index.css` | WCAG 2.2 AA styling using CSS `color-scheme: light dark` and `light-dark()` tokens |
| `src/translationUtils.ts` | Consistent entity-to-placeholder mapping (`[PERSON A]`, `[ORGANISATION A]`, etc.) and local unmasking |
| `src-tauri/src/lib.rs` | Tauri commands, HuggingFace streaming model downloaders, precompiled Swedish PII & secret regex engine, overlap resolution |
| `src-tauri/src/ner.rs` | `ort` ONNX Runtime session for Swedish BERT NER, zero-copy paragraph/sentence chunker, parallel batch tokenization, batched inference |
| `src-tauri/src/classifier.rs` | `EmbeddingGemma 2` (768d) semantic classifier, smart paragraph merging (`MERGE_TARGET_CHARS = 320`), Swedish anchor prototypes, traffic-light safety verdicts |
| `src-tauri/src/documents.rs` | Pure-Rust portable import for `.docx`/`.odt`/`.xlsx`/`.xls`/`.ods`/`.pdf`/`.txt` and split-run-safe format-preserving `.docx` & `.xlsx` export |
| `.github/workflows/build.yml` | GitHub Actions Windows build producing both NSIS setup installer and single-file `Maskera-portabel.exe` |
