# Roadmap — Framtida utvecklingssteg för Maskera

Detta dokument samlar arkitekturella vägval, lärdomar och planerade nästa steg för **Maskera** — både för lokal AI-avidentifiering (**Läge 1**) och fristående informationsklassning enligt trafikljusmodellen Grön / Gul / Röd (**Läge 2**).

---

## Grundprinciper som styr vidareutvecklingen

1. **100 % datakontroll (ingen extern molnexponering):** All analys ska köras antingen helt lokalt på medarbetarens dator eller — i ett framtida serverläge — inom organisationens egen kontrollerade infrastruktur.
2. **Beslutsstöd, aldrig ”automatisk sanning”:** Maskinella bedömningar är ett pedagogiskt stöd som hjälper medarbetaren att stanna upp. Det är alltid människan som väger in dokumentets helhet och verksamhetens styrdokument.
3. **Modulär separation mellan Läge 1 och Läge 2:** Avidentifiering för AI och informationsklassning (Grön / Gul / Röd) hålls fortsatt separerade i kod och gränssnitt så att de vid behov kan drivas som två skilda produkter eller tjänster.

---

## Spår 1: Identifiering av yrkestitlar, roller och kvasidentifierare (Lager 1 & 2)

### Bakgrund & nuläge
Nuvarande språkmodell i Lager 2 (`KBLab/bert-base-swedish-cased-ner`, tränad på *SUC 3.0*) är byggd för att hitta **egennamn** (`Person`, `Organisation`, `Plats`, `Tid`, `Händelse`). Eftersom yrkestitlar och befattningar (*verksamhetschef*, *läkare*, *miljösamordnare*, *rektor*) rent språkligt är allmänna substantiv klassas de idag som vanlig text (`O`).

I kommunal och regional verksamhet fungerar dock smala befattningar ofta som **indirekta identifierare (kvasidentifierare)**: det finns många *läkare*, men ofta bara en enda *miljösamordnare* eller *verksamhetschef* på en specifik enhet (`[ORGANISATION A]`). Samtidigt vill användaren ibland *behålla* generella yrkesord för att en AI-tjänst ska förstå sammanhanget.

### Möjliga genomförandesteg

- **Steg 1A — Svensk sammansättnings- och ordlistemotor i Lager 1 (✅ Klar — 0 MB extra, < 1 ms):**
  - Utnyttjar svenskans regelbundna ordbildning i `src-tauri/src/roles.rs` genom att matcha vanliga befattningsefterled (`-chef`, `-samordnare`, `-läkare`, `-sköterska`, `-handläggare`, `-direktör`, `-rektor`, `-kurator`, `-psykolog`, `-sekreterare`, `-ansvarig`, `-ledare` m.fl. i obestämd/bestämd form, plural och genitiv) kombinerat med kurerad basordlista, flerordsfraser och förställda attribut (*biträdande*, *tf.*, *vård- och omsorgschef* m.fl.) samt undantagsfilter mot falska vänner.
  - Exponeras som en direkt snabbknapp i verktygsraden (`+ Roll/Titel` → `[ROLL A]`) i Läge 1 (*Avidentifiera för AI*), valbar i *Filter* och förvalt aktiv i profilen **Strikt (Allt)**.
- **Steg 1B — Finjusterad svensk NER-modell med egen `TITEL/ROLL`-klass:**
  - Finjustera grundmodellen `KB/bert-base-swedish-cased` på svensk korpus med uppmärkta yrkesroller/titlar (t.ex. *Swe-NERC* kombinerat med syntetiska exempel från vård, skola, kollektivtrafik och förvaltning) och exportera till kvantiserad INT8 ONNX (~121 MB).
  - Ger full meningskontext utan att öka modellstorleken eller sänka hastigheten jämfört med dagens Lager 2.
  - *Alternativt spår att bevaka:* Zero-shot NER-modeller i ONNX (t.ex. *GLiNER*) där etiketter kan anges dynamiskt utan omträning.

---

## Spår 2: ”Lager 4” — Helhetstolkning med lokal lättvikts-LLM (`Gemma 4 E2B`)

### Vilken begränsning i Lager 3 löser Lager 4?
Lager 3 (`EmbeddingGemma 2`) är en inbäddningsmodell (*bi-encoder*) som omvandlar enskilda textavsnitt till 768-dimensionella vektorer och mäter **ämnes- och vokabulärlikhet** — den ”resonerar” inte över hela dokumentet.
- Ett dokument som beskriver *hur man generellt håller svåra samtal eller förhandlingar* (t.ex. ett kurs- eller seminariereferat) innehåller nästan samma ord som ett *faktiskt personal- eller upphandlingsärende*.
- En kompakt generativ språkmodell med *cross-attention* kan däremot hålla hela dokumentets ramverk i minnet och skilja på **omnämnande/metodbeskrivning** och **faktiskt skyddsvärt ärende**.

### Varför `Gemma 4 E2B` (mobil/text-varianten) är en stark kandidat
- **Byggd för lokal kantkörning (Edge):** Effektiv 2B-arkitektur med *Per-Layer Embeddings (PLE)* och upp till **128k tokens** kontextfönster.
- **Hanterbar fotavtrycksstorlek:** Googles *Mobile Text-Only*-kvantisering ligger på ca **0,84 GB** (`.litertlm`) och 4-bitars QAT `GGUF` (`Q4_K_XL`) på **2,62 GB**, fullt körbart på vanliga kontorsdatorer (16 GB RAM) helt på CPU.
- **Verifierad vinnare i lokal CPU-benchmark (november 2026):** Jämfört mot rena textutmanare på exakt samma svenska 10-dokumentssvit (`n_gpu_layers = 0`):
  - **`Gemma 4 E2B IT QAT` (`Q4_K_XL`, 2,62 GB):** **100 % (20/20)** giltig JSON, kortast svar (~38 tokens), naturlig svenska och ~28 % *snabbare* på ren CPU än `Qwen3-1.7B` tack vare PLE-arkitekturen och effektivare svensk tokenisering (~11–12,5 s på kontors-PC).
  - **`Qwen3-1.7B` (`Q4_K_M`, 1,11 GB, non-thinking):** Långsammare på CPU (~70 ut-tokens, slog i taket i 7/20 körningar), svagare svensk grammatik och missade bl.a. pågående anbudssekretess.
  - **`Gemma 3 1B IT QAT` (`Q4_0`, 0,72 GB):** Mycket snabb (~4–5 s), men för liten för pålitlig JSON-instruktionsföljning (kopierade promptmallen ordagrant i 11/20 körningar).
- **Rätt svårighetsgrad på uppgiften:** Lager 4 behöver inte skriva långa fria texter, utan enbart göra en **avgränsad kontextuell rimlighetsbedömning** utifrån tydliga trafikljuskriterier.

### Föreslagen ”Tratt-arkitektur” (Cascade) för hög hastighet på CPU
För att undvika 10–20 sekunders väntetid på långa Word- och PDF-dokument ska Lager 4 **inte** ersätta Lager 1–3, utan fungera som en **överprövare (”domare”)** ovanpå dem:

```
Dokument in
  ├── Lager 1 (Regex, < 1 ms)      ──► Hårda identifierare (Personnr, Bankkonto, Kort, Nycklar)
  ├── Lager 2 (BERT NER, ~50 ms)   ──► Namngivna entiteter (Person, Organisation, Plats...)
  ├── Lager 3 (EmbeddingGemma 2)   ──► Snabb grovscanning av alla avsnitt (< 0,5 s)
  │                                      │
  │     Komprimerat beslutsunderlag (~500–1 200 tokens):
  │     • Dokumentets titel, inledning & avslutning (syfte/dokumenttyp)
  │     • Sammanfattning av fynd från Lager 1 & 2
  │     • Endast de avsnitt som Lager 3 flaggat som Gula eller Röda
  ▼
  └── Lager 4 (Gemma 4 E2B)        ──► Helhetsbedömning i låst JSON-format (~60–80 ut-tokens)
                                       (~1,5–3 s på vanlig kontorsprocessor)
```

**Exempel på strukturerat svar från Lager 4 (`temperature = 0.0`, låst JSON-schema):**
- Identifierad dokumenttyp (t.ex. *Kursreferat*, *APT-anteckning*, *Utkast till broschyr*, *Enskilt HR-/rehabärende*).
- Helhetsbedömd nivå (`green` / `yellow` / `red`) och eventuell omklassning av lösryckta stycketräffar från Lager 3.
- Kort, begriplig motivering på svenska (1–2 meningar) som förklarar *varför* dokumentet i sin helhet bedöms som Grönt, Gult eller Rött.

### Tekniska vägval och skyddsräcken för Lager 4
1. **Säkerhetsprincip — ”Lager 1 har alltid vetorätt”:**
   Lager 4 får **aldrig** nedgradera bekräftade strukturfynd från Lager 1 (t.ex. giltiga personnummer, personliga bankkontonummer, SITHS-/kortnummer eller lösenord/API-nycklar). Lager 4 får enbart nyansera och överpröva *semantiska* gråzoner från Lager 3.
2. **Val av lokal inferensmotor (Runtime):**
   Idag kör Lager 2 och 3 på statiskt länkad Microsoft ONNX Runtime (`ort`). För Gemma 4 E2B utvärderas två huvudspår för Windows/macOS:
   - **LiteRT-LM (`.litertlm` med XNNPACK):** Googles egen edge-motor som har direkt stöd för 0,84 GB mobil/text-kvantiseringen och PLE-arkitekturen.
   - **`llama.cpp` / `GGUF` (t.ex. via `llama-cpp-2` i Rust):** Mogen paketering för Windows CPU (AVX2) och integrerade kretsar (Vulkan).
3. **Stegvis/valbar modellhämtning:**
   För att hålla basinstallationen snabb (~318 MB) kan Lager 4 antingen hämtas vid behov som *”Fördjupad helhetsanalys”* eller förpackas färdigt i en intern organisationsdistribution (`models/`).

---

## Spår 3: Verksamhetsanpassning & Framtida serverarkitektur

### 1. Extern, signerad JSON-profil för klassningsexempel
- Lyft ut referensankare (`INFO_GROUPS`, `INFO_GREEN_PROMPTS`) och hanteringsvägledningar (lagringsytor som Teams, OneDrive, SharePoint, diarium och verksamhetssystem) ur Rust-koden till en **versionshanterad JSON-konfigurationsfil**.
- Gör det möjligt för informationssäkerhetssamordnare och superanvändare att löpande ”träna”/kalibrera Lager 3 med nya verksamhetsnära exempel från vård, skola, administration och kollektivtrafik utan ny programrelease.

### 2. Samma 4-lagersarkitektur lokalt och som central servertjänst
Eftersom klassificeringsmotorn (`src-tauri/src/info_classification.rs`) är frikopplad från maskeringsflödet med ett rent JSON-kontrakt (`InfoClassificationReport`) kan arkitekturen växa i två kompatibla spår:

| Egenskap | Lokalt skrivbordsläge (`Maskera-portabel.exe`) | Framtida intern servertjänst (On-prem / Intranät) |
|---|---|---|
| **Målgrupp / Användning** | Fristående på medarbetarens PC (även offline) | Centralt beslutsstöd via webbgränssnitt, intranät eller Office/SharePoint-tillägg |
| **Lager 1–3** | Regex + Svensk BERT + `EmbeddingGemma 2` lokalt | Exakt samma Rust-motor på intern server |
| **Lager 4 (Helhetstolkning)** | Lättviktsmodell: **`Gemma 4 E2B`** (~0,84–1,3 GB) på CPU | Större syskonmodell på intern GPU-server: **`Gemma 4 E4B`** eller **`Gemma 4 26B-A4B (MoE)`** |
| **Klientnedladdning** | ~0,3 GB (Lager 1–3) + ev. ~0,84 GB (Lager 4) | **0 MB** på klientdatorn |
| **Styrning av regler & MDAP-stöd** | Lokal/distribuerad JSON-profil | Centralt uppdaterad i realtid för hela organisationen |

---

## Föreslagen etappindelning

| Etapp | Fokusområde | Insats | Effekt |
|---|---|---|---|
| **Etapp 1** | **Lager 4 (`Gemma 4 E2B` Judge i *Klassificera information*):** Valbar lokal helhetsbedömning (`llama.cpp` / `GGUF` QAT Q4_K_XL, ~2,62 GB) med tratt-arkitektur, varm KV-cache för systemprompt, förifylld kompakt JSON utan GBNF-overhead, Lager 1-vetorätt och en-klicks avinstallation | ✅ **Klar i v0.6.0+** | Löser kontextfällan i Lager 3 (skiljer t.ex. kursreferat/rutin från skarpt ärende) |
| **Etapp 2** | **Externa klassningsprofiler (JSON):** Konfigurerbara gröna/gula/röda typexempel och lagringsanvisningar utan omkompilering | Låg | Superanvändare/infosäk kan kalibrera exempel löpande |
| **Etapp 3** | **Titlar & roller (Steg 1A):** Svensk sammansättnings- och ordlistemotor i Lager 1 + valbart filter `+ Roll/Titel` (`[ROLL A]`) | ✅ **Klar** | Skyddar mot bakvägsidentifiering via unika befattningar (0 MB extra, < 1 ms) |
| **Etapp 4** | **Finjusterad BERT (`Steg 1B`) & ev. Server-API:** Kontexthantering av roller i Lager 2 samt möjlighet att köra `info_classification` + större `Gemma 4`-modell som central intern servertjänst | Medel–Hög | Skalbar organisationslösning med noll klientunderhåll för klassificering |

