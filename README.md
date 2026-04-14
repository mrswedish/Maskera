# Maskera

Maskera är ett lokalt verktyg för automatisk identifiering och maskering av personuppgifter (PII) i svenska texter. Allt körs på din dator — ingen data skickas till externa tjänster.

---

## Vad gör det?

Ladda upp en `.txt`-fil, välj vilka entitetstyper som ska sökas, och låt verktyget:

1. **Identifiera** personuppgifter med regelbaserad sökning (regex) och AI (BERT NER)
2. **Markera** träffarna i texten med färgkodning per kategori
3. **Granska** och justera — avmarkera felaktiga träffar eller lägg till ord manuellt
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
  └── React-gränssnitt (TypeScript)
        └── Tauri-kommandon (invoke)
              ├── Regex-pass (personnummer, e-post, telefon) — Rust
              └── BERT NER — tract-onnx i Rust
                    ├── Modell laddas ner från HuggingFace vid första start
                    ├── Cachas i app-datakatalogen (aldrig bundlad i appen)
                    └── Körs helt lokalt — ingen server behövs
```

### Analysflödet

1. **Regex-pass** — tre mönster för personnummer, e-post och telefonnummer (Rust)
2. **BERT NER** — texten delas i 1 200-teckens chunks (med 80-teckens överlapp); varje chunk analyseras av ONNX-modellen via `tract-onnx`
3. **Ord-aggregering** — tokenizerns `word_ids` används för att gruppera subword-tokens till hela ord, undviker att delar av sammansatta ord felidentifieras som entiteter
4. **Konfidenströskel** — tokens med för svag modellsäkerhet filtreras bort (justerbar i inställningar)
5. **Deduplicering** — överlappande träffar från olika chunks filtreras bort
6. **Granskning** — användaren kan klicka för att ignorera träffar, eller lägga till egna
7. **Maskering** — aktiva träffar ersätts med etiketter som `[PERSON A]`, `[PERSONNUMMER 1]`

### Inställningar

Klicka på kugghjulet (⚙) i övre högra hörnet för att justera **konfidenströskel** (30–95 %). Lägre värde ger fler träffar men fler falska positiv; högre värde ger färre men säkrare träffar. Standardvärde: 60 %.

### Översättningstabellen

Varje unik text mappas till ett konsekvent maskerat label:
- AI-entiteter → bokstavssuffix: `Person A`, `Person B` …
- Regex/manuella → numeriskt suffix: `Personnummer 1`, `E-post 2` …

Tabellen kan exporteras som JSON eller tab-separerad TXT.

### Modell

AI-modellen ([KBLab/bert-base-swedish-cased-ner](https://huggingface.co/KBLab/bert-base-swedish-cased-ner)) är kvantiserad till ONNX int8-format och publicerad på HuggingFace: [psvensk/bert-base-swedish-cased-ner-onnx](https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx).

Modellen (~120 MB) laddas ner automatiskt vid första start och cachas i appens datakatalog. Nedladdningen visas med en progress-bar i gränssnittet.

---

## Teknik

| Lager | Teknologi |
|---|---|
| AI-modell | [KBLab/bert-base-swedish-cased-ner](https://huggingface.co/KBLab/bert-base-swedish-cased-ner) — svensk BERT tränad på NER |
| ML-inferens | [tract-onnx](https://github.com/sonos/tract) — native Rust ONNX-körning, fullt systemminne |
| Tokenisering | [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) (Rust) |
| HTTP-nedladdning | [reqwest](https://github.com/seanmonstar/reqwest) med rustls-TLS |
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

Tauri-fönstret öppnas. Första gången laddas ONNX-modellen ner från HuggingFace (~120 MB) — detta sker med progress-bar i appen.

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

Färdigbyggda Windows-versioner finns under [Releases](../../releases). Ladda ner installationsfilen och kör — modellen laddas ner automatiskt vid första start.
