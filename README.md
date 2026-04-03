# Maskera

Maskera är ett lokalt verktyg för automatisk identifiering och maskering av personuppgifter (PII) i svenska texter. Allt körs på din dator — ingen data skickas till externa tjänster.

---

## Vad gör det?

Ladda upp en `.txt`-fil, välj vilka entitetstyper som ska sökas, och låt verktyget:

1. **Identifiera** personuppgifter med en kombination av regelbaserad sökning (regex) och AI
2. **Markera** träffarna direkt i texten med färgkodning per kategori
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
| Övrigt | AI (BERT) | Övriga namngivna entiteter |

Regex-kategorier körs alltid. AI-kategorier kan slås av/på individuellt.

---

## Hur fungerar det?

```
masking_engine (Python-binär)
  ├── FastAPI-server på localhost:8594
  │     ├── POST /analyze  →  regex-pass + BERT NER
  │     └── GET  /         →  serverar React-gränssnittet
  └── Öppnar webbläsaren automatiskt vid start
```

### Analysflödet

1. **Regex-pass** — tre mönster för personnummer, e-post och telefonnummer körs alltid
2. **BERT NER** — texten delas i 1 500-teckens chunk för att hålla sig under BERT:s 512-tokengräns; varje chunk analyseras av KBLab-modellen
3. **Deduplicering** — överlappande träffar filtreras bort (regex prioriteras vid konflikt)
4. **Granskning** — användaren kan klicka på träffar för att ignorera dem, eller lägga till egna
5. **Maskering** — aktiva träffar ersätts med etiketter som `[PERSON A]`, `[PERSONNUMMER 1]`

### Översättningstabellen

Varje unik text mappas till ett konsekvent maskerat label:
- AI-entiteter → bokstavssuffix: `Person A`, `Person B` …
- Regex/manuella → numeriskt suffix: `Personnummer 1`, `E-post 2` …

Tabellen kan exporteras som JSON eller tab-separerad TXT för vidare hantering.

---

## Teknik

| Lager | Teknologi |
|---|---|
| AI-modell | [KBLab/bert-base-swedish-cased-ner](https://huggingface.co/KBLab/bert-base-swedish-cased-ner) — svensk BERT tränad på NER |
| ML-ramverk | Hugging Face Transformers + PyTorch |
| API-server | FastAPI + Uvicorn |
| Gränssnitt | React 19 + TypeScript (byggs till statiska filer, serveras av FastAPI) |
| Paketering | PyInstaller — allt i en enda körbar fil |
| CI/CD | GitHub Actions — automatiskt Windows-bygge vid push till `main` |

---

## Köra lokalt (källkod)

```bash
# Installera Python-beroenden
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install fastapi uvicorn pydantic transformers torch pyinstaller

# Bygg gränssnittet
npm install
npm run build

# Starta
python masking_engine.py
```

Webbläsaren öppnas automatiskt på `http://127.0.0.1:8594`. Stoppa med `Ctrl+C`.

> **Första start:** AI-modellen (~500 MB) laddas ner från Hugging Face automatiskt och cachas lokalt.

## Bygga fristående binär

```bash
# Mac (Apple Silicon)
npm run build
pyinstaller --onefile masking_engine.py --name Maskera \
  --add-data "dist:dist" --icon src-tauri/icons/icon.icns

# Windows
npm run build
pyinstaller --onefile masking_engine.py --name Maskera ^
  --add-data "dist;dist" --icon src-tauri\icons\icon.ico
```

---

## Releases

Färdigbyggda Windows-versioner finns under [Releases](../../releases). Ladda ner `Maskera.exe` och kör — webbläsaren öppnas automatiskt.
