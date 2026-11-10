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
| Telefonnummer | Regex | `+46 8 123 45 67`, `08-123-45-67` |
| Organisationsnummer | Regex | `556614-5180` |
| Konto/Giro | Regex | IBAN `SE9550000000054910000003`, Bankgiro `5402-1234`, Plusgiro `12345-6` |
| IP-adress | Regex | `8.8.8.8` (publika; privata nät 10.x, 192.168.x m.fl. filtreras bort) |
| Kortnummer | Regex + Luhn | `4532-1234-5678-9123` (Visa/Mastercard/Amex, valideras med Luhn-algoritmen) |
| Person | AI (BERT) | `Anders Andersson` |
| Organisation | AI (BERT) | `Försäkringskassan` |
| Plats | AI (BERT) | `Stockholm` |
| Övrigt | AI (BERT) | Diverse namngivna entiteter |
| Tid | AI (BERT) | `idag`, `fem år` *(av som standard)* |
| Händelse | AI (BERT) | `coronapandemin` *(av som standard)* |

Alla kategorier kan slås av/på individuellt i vänsterpanelen.

---

## Hur fungerar det?

```
Tauri-app (Rust-shell + WebView)
  └── React-gränssnitt (TypeScript)
        └── Tauri-kommandon (invoke)
              ├── Regex-pass (förkompilerad Rust-motor, < 1 ms)
              └── BERT NER — Microsoft ONNX Runtime (`ort`, statiskt länkad i Rust)
                    ├── Modell laddas ner från HuggingFace vid första start
                    ├── Cachas i app-datakatalogen (aldrig bundlad i appen)
                    └── Flertrådad SIMD + mini-batchning helt lokalt — ingen server behövs
```

### Analysflödet

1. **Regex-pass** — förkompilerade mönster med svensk datum-/kontrollsiffrevalidering (skiljer t.ex. automatiskt personnummer/samordningsnummer från organisationsnummer). Om endast Regex-kategorier är valda körs analysen på under 1 ms utan att vänta på AI-modellen.
2. **Smart chunking & BERT NER** — texten delas vid naturliga stycke- och meningsgränser (~1 600 tecken, noll onödigt överlapp vid ren brytpunkt) och analyseras parallellt i längdsorterade mini-batchar via Microsoft ONNX Runtime (`ort`).
3. **Ord-aggregering** — tokenizerns `word_ids` används för att gruppera subword-tokens till hela ord, undviker att delar av sammansatta ord felidentifieras som entiteter
4. **Konfidenströskel** — tokens med för svag modellsäkerhet filtreras bort (justerbar i inställningar)
5. **Deduplicering & prioritering** — exakta Regex-träffar prioriteras framför överlappande NER-träffar
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

Modellfilerna (~120 MB totalt) laddas ner automatiskt vid första start via Rust async streaming och cachas i appens datakatalog. Nedladdningen visas med en progress-bar i gränssnittet. Vid eventuella nätverksproblem finns ett manuellt alternativ — se nedan.

### Nätverksbegränsade miljöer (företagsproxy)

Om automatisk nedladdning misslyckas (t.ex. brandvägg eller intern proxy) kan modellfilerna laddas ner manuellt:

1. Ladda ner alla filer från [HuggingFace](https://huggingface.co/psvensk/bert-base-swedish-cased-ner-onnx) och spara dem i en lokal mapp
2. Klicka på **Välj modellmapp manuellt** i appen och peka på den mappen

Filer som krävs: `model_quantized.onnx`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`

---

## Teknik

| Lager | Teknologi |
|---|---|
| AI-modell | [KBLab/bert-base-swedish-cased-ner](https://huggingface.co/KBLab/bert-base-swedish-cased-ner) — svensk BERT tränad på NER |
| ML-inferens | [ort (Microsoft ONNX Runtime)](https://ort.pyke.io) — statiskt länkad native SIMD + batchad inferens |
| Tokenisering | [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) (Rust) |
| HTTP-nedladdning | [reqwest](https://github.com/seanmonstar/reqwest) — async streaming med native-TLS (respekterar OS-certifikat och systemproxy) |
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
