# Maskera

Maskera är ett lokalt verktyg för automatisk identifiering och maskering av personuppgifter (PII) i svenska texter. Allt körs på din dator — ingen data skickas till externa tjänster.

---

## Vad gör det?

Ladda upp en `.txt`-fil, välj vilka entitetstyper som ska sökas, och låt verktyget:

1. **Identifiera** personuppgifter med regelbaserad sökning (regex) och AI direkt i webbläsaren
2. **Markera** träffarna i texten med färgkodning per kategori
3. **Granska** och justera — avmarkera felaktiga träffar eller lägg till missade ord manuellt
4. **Exportera** en maskerad version av texten samt en översättningstabell

### Vad detekteras?

| Kategori | Metod | Exempel |
|---|---|---|
| Personnummer | Regex | `850515-1234` |
| E-postadress | Regex | `namn@example.se` |
| Telefonnummer | Regex | `+46 8 123 45 67` |
| Person | AI (BERT) | `Anders Andersson` |
| Organisation | AI (BERT) | `Försäkringskassan` |
| Plats | AI (BERT) | `Stockholm` |
| Övrigt | AI (BERT) | Diverse namngivna entiteter |
| Tid | AI (BERT) | `idag`, `fem år` *(av som standard)* |
| Händelse | AI (BERT) | `coronapandemin` *(av som standard)* |

Regex-kategorier körs alltid. AI-kategorier kan slås av/på individuellt.

---

## Hur fungerar det?

```
Tauri-app (Rust-shell + WebView)
  └── React-gränssnitt
        └── Web Worker
              ├── Regex-pass (personnummer, e-post, telefon) — alltid
              └── Transformers.js → ONNX-modell (KBLab BERT NER)
                    └── Körs lokalt i webbläsaren — ingen server behövs
```

### Analysflödet

1. **Regex-pass** — tre mönster för personnummer, e-post och telefonnummer
2. **BERT NER** — texten delas i 1 500-teckens chunk för att hålla sig under BERT:s 512-tokengräns; varje chunk analyseras av ONNX-modellen via Transformers.js
3. **Manuell aggregering** — råa token-prediktioner grupperas till hela namnentiteter med exakta teckenpositioner
4. **Deduplicering** — överlappande träffar filtreras bort
5. **Granskning** — användaren kan klicka för att ignorera träffar, eller lägga till egna
6. **Maskering** — aktiva träffar ersätts med etiketter som `[PERSON A]`, `[PERSONNUMMER 1]`

### Översättningstabellen

Varje unik text mappas till ett konsekvent maskerat label:
- AI-entiteter → bokstavssuffix: `Person A`, `Person B` …
- Regex/manuella → numeriskt suffix: `Personnummer 1`, `E-post 2` …

Tabellen kan exporteras som JSON eller tab-separerad TXT.

### Modell

AI-modellen ([KBLab/bert-base-swedish-cased-ner](https://huggingface.co/KBLab/bert-base-swedish-cased-ner)) är konverterad till ONNX-format och finns på HuggingFace: [psvensk/bert-base-swedish-cased-ner-onnx](https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx).

Modellen (~120 MB, kvantiserad int8) laddas ner automatiskt vid första körning och cachas i webbläsarens lokala lagring.

---

## Teknik

| Lager | Teknologi |
|---|---|
| AI-modell | [KBLab/bert-base-swedish-cased-ner](https://huggingface.co/KBLab/bert-base-swedish-cased-ner) — svensk BERT tränad på NER |
| ML i webbläsaren | [Transformers.js](https://huggingface.co/docs/transformers.js) + ONNX Runtime WebAssembly |
| Gränssnitt | React 19 + TypeScript |
| Desktop-shell | [Tauri 2](https://tauri.app) (Rust) — ~5 MB binär |
| CI/CD | GitHub Actions — automatiskt Windows-bygge vid push till `main` |

**Ingen Python krävs** — varken för att köra eller bygga appen.

---

## Köra lokalt (källkod)

```bash
npm install
npm run tauri dev
```

Tauri-fönstret öppnas automatiskt. Första gången laddas ONNX-modellen ner (~120 MB).

## Bygga fristående app

```bash
# Mac
npm run tauri build
# → src-tauri/target/release/bundle/macos/maskera.app
# → src-tauri/target/release/bundle/dmg/maskera_x.x.x_aarch64.dmg

# Windows (via GitHub Actions vid push till main)
```

---

## Releases

Färdigbyggda Windows-versioner finns under [Releases](../../releases). Ladda ner installationsfilen och kör.
