# Maskera

Maskera är ett snabbt, helt lokalt skrivbordsverktyg för automatisk identifiering och maskering av personuppgifter (PII), tekniska hemligheter och känslig kontext i svenska texter — samt lokal återställning (avmaskering) av AI-svar. Allt körs **100 % lokalt** på din dator utan molnanrop.

---

## Vad gör det?

Klistra in text direkt från urklipp (`Ctrl+V` eller knappen **Klistra in**), skriv i editorn eller öppna/dra-och-släpp **Word (`.docx`, `.odt`), Excel (`.xlsx`, `.xls`, `.ods`), PDF (`.pdf`) eller textfiler (`.txt`, `.csv`, `.md`)**, välj snabbprofil och låt Maskera arbeta i tre lager:

1. **Lager 1 — Struktur & Hemligheter (< 1 ms):** Förkompilerade reguljära uttryck med Luhn- och datumvalidering identifierar personnummer, samordningsnummer, organisationsnummer, e-post, telefonnummer, IBAN/Bankgiro/Plusgiro, betalkort, 16–22-siffriga tjänstekort/SITHS-kort, HSA-ID, interna intranätlänkar (`https://intranat...`) samt API-nycklar/JWT/privata nycklar.
2. **Lager 2 — Namngivna entiteter (Svensk BERT NER via `ort`):** Identifierar personnamn, organisationer, platser, tider och händelser i kontext (~33× snabbare exekvering via statiskt länkad Microsoft ONNX Runtime med flertrådad SIMD och längdsorterad mini-batchning).
3. **Lager 3 — Semantisk AI-gatekeeper (`EmbeddingGemma 2`):** Analyserar dokumentets avsnitt semantiskt (768-dimensionella inbäddningar) och flaggar samt **markerar känsliga stycken direkt i textvyn** om texten innehåller:
   - **Hälsa & Vård (GDPR Art. 9)** — diagnoser, sjukskrivning, behandling, symtom, vårdjournaler.
   - **Känslig personuppgift / Lagöverträdelse (GDPR Art. 9–10)** — brottsmisstanke, dom, socialtjänst, LVU/LVM, facklig/politisk/religiös tillhörighet.
   - **Tjänstekort, eID, Behörighet & Intranät** — SITHS-kort, e-legitimation, certifikat, reserv-/konsultkort, korthandläggare och interna intranätrutiner.
   - **Affärskritiskt & Företagshemligheter** — opublicerade siffror, M&A-bud, marginaler, kundofferter, avtalsvillkor.
   - **Säkerhet, IT-arkitektur & Intern kod** — sårbarheter, penetrationstester, brandväggsregler, intern infrastruktur.
   - **HR & Personalärende** — lönesättning, medarbetarsamtal, arbetsrättsliga ärenden.
4. **Granska, kopiera & avmaskera AI-svar:**
   - Klicka (eller tryck `Retur`/`Mellanslag`) på ett markerat ord i granskningsvyn för att undanta/avmaskera det — trafikljusbedömningen uppdateras direkt (< 1 ms).
   - Klicka på ett flaggat avsnitt i Gatekeeper-bannern (`Visa i text →`) för att hoppa direkt till det inramade stycket i texten.
   - Kopiera den maskerade texten med 1 klick (**Kopiera maskerad**), spara som `.txt` eller **exportera ett maskerat Word-dokument (`*_maskerad.docx`) eller en maskerad Excel-arbetsbok (`*_maskerad.xlsx`) med bibehållen formatering** (inklusive hantering av ord som Word har delat upp över flera XML-runs `<w:r><w:t>`), och klistra in svaret från ChatGPT/Claude/Copilot/Gemini under fliken **Avmaskera AI-svar** — eller välj **Avmaskera .docx / .xlsx-fil** för att återställa ett helt dokument till originalvärdena.

---

### Vad detekteras?

| Kategori | Lager / Metod | Exempel |
|---|---|---|
| Hemlighet/Nyckel | Lager 1 (Regex) | API-nycklar (`sk-...`, `AIza...`, `ghp_...`, `AKIA...`), JWT, PEM-nycklar, HSA-ID (`SE...`), intranät-URL:er (`https://intranat...`) |
| Personnummer | Lager 1 (Regex + datum) | `850515-1234`, `19850515-1234`, `701063-2391` (samordningsnummer) |
| Organisationsnummer | Lager 1 (Regex) | `556614-5180` (skiljs automatiskt från personnummer via månadsfält >= 20) |
| Kortnummer | Lager 1 (Regex + Luhn / SITHS) | Betalkort (`4532-...` med Luhn) samt 16–22-siffriga tjänste-/SITHS-kort (`9752267235710527661`) |
| E-postadress | Lager 1 (Regex) | `namn@example.se` |
| Telefonnummer | Lager 1 (Regex) | `+46 8 123 45 67`, `08-123 45 67` |
| Konto/Giro | Lager 1 (Regex) | IBAN `SE9550000000054910000003`, Bankgiro `5402-1234`, Plusgiro `12345-6` |
| IP-adress | Lager 1 (Regex) | Publika IPv4-adresser (privata nät `10.x`, `192.168.x`, `127.x` filtreras bort) |
| Person | Lager 2 (BERT NER) | `Anders Andersson` |
| Organisation | Lager 2 (BERT NER) | `Försäkringskassan`, `Region Jönköpings län` |
| Plats | Lager 2 (BERT NER) | `Stockholm`, `Jönköping` |
| Övrigt | Lager 2 (BERT NER) | Diverse namngivna entiteter |
| Tid | Lager 2 (BERT NER) | `idag`, `2027-01-04` |
| Händelse | Lager 2 (BERT NER) | Namngivna händelser |
| Känsliga stycken | Lager 3 (`EmbeddingGemma 2`) | Vård/hälsa, lagöverträdelser, SITHS/eID/intranät, företagshemligheter, IT-säkerhet, HR |

---

## Tillgänglighet (WCAG 2.2 AA) & Tema

- **Ljust och mörkt tema:** Följer operativsystemets färgschema (`color-scheme: light dark` och CSS `light-dark()`) som standard, med möjlighet att växla direkt i sidhuvudet (**Ljust tema** / **Mörkt tema**) eller välja **System / Ljust / Mörkt** i konfigurationspanelen utan vit blinkning (FOUC) vid uppstart.
- **Kontraster (WCAG 1.4.3 & 1.4.11):** Alla textfärger och kategorietiketter är kalibrerade för minst `4.5:1` kontrast mot sin bakgrund, och interaktiva ramar/fokusringar håller minst `3:1` kontrast (samt stöd för `@media (prefers-contrast: more)` och Windows High Contrast `@media (forced-colors: active)`).
- **Full tangentbordsnavigering:** Stöd för piltangentsnavigering i verktygsrad och flikar (`focusgroup`), tydliga `:focus-visible`-ramar samt tangentbordsstyrning (`Retur`/`Mellanslag`) på alla ordmarkeringar och styckehopp.

---

## Hur fungerar arkitekturen?

```
Tauri 2-app (Rust-shell + WebView)
  └── React 19-gränssnitt (TypeScript, WCAG AA, Ljust/Mörkt tema)
        └── Tauri-kommandon (invoke + live-events)
              ├── Lager 1: Förkompilerad Rust Regex-motor (< 1 ms)
              ├── Lager 2: KBLab/bert-base-swedish-cased-ner via Microsoft ONNX Runtime (`ort`)
              └── Lager 3: EmbeddingGemma 2 (768d q4 ONNX) semantisk styckeklassificerare
```

### Modeller & lokal lagring

- **Lager 2 (BERT NER, ~120 MB):** [`psvensk/bert-base-swedish-cased-ner-onnx`](https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx) laddas ner automatiskt vid första start.
- **Lager 3 (EmbeddingGemma 2, ~197 MB):** [`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) (`model_q4.onnx`) hämtas med ett klick inifrån appen och cachas i användarens lokala appdata-mapp.

---

## Köra lokalt (källkod)

```bash
npm install
npm run tauri dev
```

## Bygga fristående app

```bash
# Kontrollera frontend + Rust-tester
npm run build
cargo test --manifest-path src-tauri/Cargo.toml

# Bygg release-paket
npm run tauri build
```

Vid push till `main` bygger GitHub Actions automatiskt både:
1. **`Maskera-portabel.exe`** — en fristående fil som kan köras direkt utan installation eller administratörsrättigheter.
2. **`Maskera-installation`** — vanlig Windows NSIS-installerare (`*-setup.exe`).
