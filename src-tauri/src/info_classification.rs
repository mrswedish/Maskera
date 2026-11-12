use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::classifier::{
    clean_for_embedding, dot, is_standalone_heading, split_paragraphs, EmbeddingClassifier,
};
use crate::info_judge::JudgeAssessment;
use crate::Match;

/// Svenskt bankkontonummer med clearingnummer (t.ex. "8327-9, 123 456 789-0" eller "clearing 8105-9 konto 943 122 811-2")
/// eller kontonummer angivet efter nyckelord som bankkonto/lönekonto/personkonto.
static RE_PERSONAL_BANK_ACCOUNT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?ix)",
        r"(?:\b\d{4}(?:-\d)?\s*[,/\s]\s*\d{2,3}[\s\-]?\d{3}[\s\-]?\d{3,4}(?:-\d)?\b)",
        r"|(?:\b(?:bankkonto|personkonto|lönekonto|clearing(?:nr|nummer)?)\s*[:=]?\s*\d{4}(?:-\d)?[\s,\-/]+\d[\d\s\-]{5,14}\b)"
    ))
    .unwrap()
});

/// Markörer för publikt Bankgiro / Plusgiro (som ofta förekommer i gröna broschyrer och fakturainformation).
static RE_PUBLIC_GIRO_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:bankgiro|plusgiro|\bbg\b|\bpg\b)\s*[:.]?\s*$").unwrap()
});

/// Övergripande informationsklassning enligt signalsystemet (Grön / Gul / Röd).
/// Utformad som ett fristående datakontrakt som även kan exponeras via en central servertjänst (HTTP/JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InfoClassificationReport {
    /// "green" | "yellow" | "red"
    pub level: String,
    /// Kort etikett för nivån, t.ex. "GUL — Var försiktig (Internt)"
    pub badge_title: String,
    /// Svar på Fråga 1: Vad innehåller informationen?
    pub content_headline: String,
    pub content_explanation: String,
    /// Konkreta orsaker/signaler som drev bedömningen (struktur + semantik)
    pub signals: Vec<InfoSignal>,
    /// Svar på Fråga 2: Hur får vi hantera informationen?
    pub handling: HandlingGuidance,
    /// Markerade avsnitt i originaltexten (utan maskering) som motiverar Gul eller Röd nivå
    pub section_findings: Vec<InfoSectionFinding>,
    pub paragraphs_scanned: usize,
    pub scan_duration_ms: u64,
    pub semantic_available: bool,
    /// Valbar helhetsbedömning från Lager 4 (Judge / Gemma 4 E2B) när användaren aktiverat den
    pub judge: Option<JudgeAssessment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InfoSignal {
    /// "yellow" | "red" | "green"
    pub level: String,
    /// "struktur" | "kontext"
    pub source: String,
    pub title: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HandlingGuidance {
    pub heading: String,
    pub summary: String,
    pub storage_guidance: String,
    pub sharing_guidance: String,
    pub examples_note: String,
    pub disclaimer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InfoSectionFinding {
    pub paragraph_index: usize,
    pub preview: String,
    pub start: usize,
    pub end: usize,
    pub category: String,
    /// "yellow" | "red"
    pub level: String,
    pub confidence: f32,
    pub reason: String,
}

struct InfoPrototypeGroup {
    category: &'static str,
    level: &'static str, // "yellow" | "red"
    reason: &'static str,
    prompts: &'static [&'static str],
}

/// Ankare för RÖD och GUL informationsklassning enligt scenariobeskrivningen för trafikljusmodellen.
const INFO_GROUPS: &[InfoPrototypeGroup] = &[
    // ── RÖDA KATEGORIER (Sekretess enligt lag / kräver särskilt skydd) ────────
    InfoPrototypeGroup {
        category: "Patient-, Hälso- & Elevuppgifter",
        level: "red",
        reason: "Innehåller hälso-/patientuppgifter, sjukskrivning eller individkopplade elevuppgifter som omfattas av sekretess eller särskilt skydd.",
        prompts: &[
            "Patientjournal för enskild patient, namngiven patients sjukdomshistoria, medicinsk diagnos, läkemedelsbehandling, psykiatrisk bedömning, sjukskrivning eller vårdkontakt.",
            "Beskrivning av en specifik persons fysiska eller psykiska hälsotillstånd, symtom, operation, provsvar, missbruk eller funktionsnedsättning.",
            "Mottagningsanteckning, daganteckning eller epikris för namngiven patient som inkommer för uppföljning med smärta, symtom, röntgensvar, misstanke om tumör eller metastasering, läkemedelsdosering och remiss.",
            "Individuella elevuppgifter inom skola eller folkhögskola, elevhälsa, åtgärdsprogram, särskilt stöd, kränkningsärende eller känsliga studieomdömen om enskild elev.",
        ],
    },
    InfoPrototypeGroup {
        category: "Skyddade personuppgifter, Socialtjänst & Bakgrundskontroll",
        level: "red",
        reason: "Berör skyddade personuppgifter, socialtjänst, orosanmälan, lagöverträdelse eller bakgrundskontroll av enskild individ.",
        prompts: &[
            "Skyddade personuppgifter, sekretessmarkering, skyddad folkbokföring, hotbild eller kvinnofrid.",
            "Brottsmisstanke, polisanmälan, förundersökning, åtal, dom i brottmål, belastningsregisterutdrag, bakgrundskontroll eller tvångsvård.",
            "Orosanmälan kring barn eller individ, ekonomiskt bistånd, individutredning hos socialtjänsten eller omhändertagande enligt LVU eller LVM.",
        ],
    },
    InfoPrototypeGroup {
        category: "Känsligt personalärende, Rehab, Löneunderlag & Avtal (HR)",
        level: "red",
        reason: "Berör rehabsamtal, sjukskrivning, disciplinärende, individuellt löneunderlag eller enskilt anställningsavtal.",
        prompts: &[
            "Rehabsamtal, rehabiliteringsutredning för namngiven medarbetare, upprepad sjukfrånvaro, läkarintyg eller arbetsanpassning efter sjukskrivning.",
            "Anteckning från uppföljningssamtal om namngiven medarbetare med sömnsvårigheter, hjärtklappning, stress eller utmattningssymtom, bedömning av företagshälsovårdens läkare, halvtidssjukskrivning och anpassade arbetsuppgifter.",
            "Skriftlig erinran, varning, misskötsamhet, disciplinärende, uppsägning av personliga skäl, avsked eller facklig tvisteförhandling kring enskild anställd.",
            "Individuellt löneunderlag med anställds månadslön, enskilt anställningsavtal med personliga villkor eller urvals- och bakgrundskontroll av namngiven kandidat.",
        ],
    },
    InfoPrototypeGroup {
        category: "Upphandlingssekretess, Affärshemligheter & Ej offentliggjord politik",
        level: "red",
        reason: "Berör pågående upphandling under anbudssekretess, konfidentiella affärsvillkor eller politiskt beslutsunderlag före offentliggörande.",
        prompts: &[
            "Sekretessbelagda upphandlingsuppgifter, inkomna anbud under pågående anbudsutvärdering, anbudspriser eller utvärderingsprotokoll före tilldelningsbeslut.",
            "Konfidentiell affärsstrategi, företagshemlighet, opublicerat bokslut, vinstmarginal, uppköpsbud eller sekretessbelagt avtalsvillkor.",
            "Politisk information före offentliggörande, konfidentiellt beredningsunderlag inför nämnd eller regionstyrelse, embargobelagt politiskt förslag eller ej offentliggjort beslut.",
        ],
    },
    InfoPrototypeGroup {
        category: "Krisberedskap, Fysisk säkerhet (Lås & Larm) & IT-säkerhet",
        level: "red",
        reason: "Innehåller uppgifter om krisberedskap, fysisk säkerhet (lås, larm, bevakning), inloggningsuppgifter/pinkoder eller tekniska sårbarheter och konfigurationer.",
        prompts: &[
            "Krisberedskap, beredskapsplan för samhällsviktig verksamhet, krigsorganisation, reservkraft, nödvatten, kontinuitetsplanering eller allvarlig säkerhetsincident.",
            "Fysisk säkerhet med lås och larm, faktiska larmkoder, pinkoder, lösenord i klartext, inloggningsuppgifter, bevakningsrutiner, kameraövervakning av skyddsobjekt eller skalskydd.",
            "Känsliga tekniska konfigurationer, brandväggsregler, intern nätverkstopologi, sårbarhetsrapport, penetrationstest, SITHS-kort med PIN/PUK-kod eller säkerhetsincidenter.",
        ],
    },

    // ── GULA KATEGORIER (Intern information inom region/avdelning/arbetsgrupp) ─
    InfoPrototypeGroup {
        category: "APT, Mötesanteckningar & Interna kursreferat",
        level: "yellow",
        reason: "Liknar interna APT-/mötesanteckningar eller en kollegas sammanfattning från kurs/seminarium avsedd för arbetsgruppen.",
        prompts: &[
            "Minnesanteckningar från arbetsplatsträff APT, enhetsmöte, veckomöte eller arbetsgruppsmöte om pågående arbete, lokala rutiner och uppföljning.",
            "Sammanfattning av kursen jag gick, egna anteckningar från seminarium, konferens, workshop eller föreläsning som delas med kollegorna i arbetsgruppen.",
        ],
    },
    InfoPrototypeGroup {
        category: "Internt arbetsmaterial, Styrdokument, Utkast & Planering",
        level: "yellow",
        reason: "Liknar interna arbetsinstruktioner, processbeskrivningar, schema/semesterplanering eller opublicerade utkast.",
        prompts: &[
            "Internt arbetsmaterial eller utkast på dokument som ännu inte är beslutade, ekonomiskt, administrativt eller kommunikationsrelaterat arbetsutkast.",
            "Interna styrdokument, arbetsinstruktioner, processbeskrivningar, kravspecifikationer eller lathund på intranätet för intern hantering inom avdelningen.",
            "Schema, bemanning och semesterplanering inom arbetsgruppen, fördelning av pass inför jul och sommar eller allmän ärendeinformation utan utpekade individer.",
        ],
    },
    InfoPrototypeGroup {
        category: "Verksamhetsutveckling, Vård-/Skolstrategi & Generella rutiner",
        level: "yellow",
        reason: "Diskuterar vård, patientjournaler, diagnoskodning, sekretess eller behörighetsrutiner på generell system- eller metodnivå utan enskilda individfall.",
        prompts: &[
            "Övergripande verksamhetsutveckling, workshop, AI-sammanfattning eller kartläggning inom hälso- och sjukvård, cancervård eller skola om personcentrering, samordning, kontaktsjuksköterskans roll, patientenkäter, vårdprocesser och resurser utan enskilda patienter.",
            "Intern rutin, riktlinje eller instruktion om hur medarbetare generellt hanterar patientjournaler, diagnoskodning, sekretess, patientdatalagen, SITHS-kort, samordnad vårdplanering eller lösenordsbyte utan uppgifter om enskild patient eller faktiska koder.",
        ],
    },
    InfoPrototypeGroup {
        category: "Medarbetarsamtal (Kontrollera om innehållet är Rött)",
        level: "yellow",
        reason: "Vanliga medarbetarsamtal om mål och kompetensutveckling är normalt Gul information — men kontrollera noga: om samtalet berör ohälsa, sjukskrivning, rehab, individuell lön eller misskötsamhet ska det hanteras som RÖD.",
        prompts: &[
            "Mall och anteckningar från årligt medarbetarsamtal om kompetensutveckling, trivsel, arbetsmål och framtida utvecklingsplan på arbetsplatsen.",
        ],
    },
];

/// Ankare för GRÖN (offentlig/publik) information samt allmänna utbildnings-/metodbeskrivningar
/// som inte innehåller intern eller sekretessbelagd verksamhetsinformation.
const INFO_GREEN_PROMPTS: &[&str] = &[
    "Publicerad informationsbroschyr för invånare, patientinformation för allmänheten, erbjudande, öppettider eller besöksinformation på extern webbplats.",
    "Offentlig patientguide och invånarinformation på 1177 eller regionens externa webbplats om hur standardiserade vårdförlopp, cancerutredning, diagnosbesked, kontaktsjuksköterska och din journal via nätet fungerar för dig som invånare.",
    "Offentlig busstidtabell, tågtider, linjekarta, biljettpriser och reseinformation för kollektivtrafik på publik webb eller offentlig anslagstavla.",
    "Publicerat pressmeddelande, offentlig nyhetsartikel, öppen inbjudan till publik konferens eller information på offentlig anslagstavla.",
    "Publicerad verksamhetsbeskrivning, offentliggjord årsredovisning, öppen kurskatalog eller allmän informationstext som vi själva vill dela externt.",
    "Allmän guide och metodtips om digitala möten, hybridmöten, mötesteknik, kameraplacering, ljud och ljus, moderatorns roll, breakout rooms, Menti och Miro.",
    "Generella råd om mötesmetodik, ledarskap i mötesrummet, presentationsteknik, gruppdiskussioner, fysiska eller digitala möten och när olika mötesformer passar bäst.",
];

/// Neutrala e-postfraser (t.ex. automatisk varningsrad för extern avsändare) som inte på egen hand ska höja klassningen.
const INFO_BOILERPLATE_PROMPTS: &[&str] = &[
    "Viktig information: Detta e-postmeddelande kommer från en extern avsändare utanför organisationen. Klicka aldrig på länkar och öppna aldrig bifogade filer om du är osäker på avsändaren eller innehållet.",
    "Automatisk e-postfotnot eller standardhälsning i slutet av ett e-postmeddelande.",
];

struct EmbeddedInfoProto {
    group_idx: usize,
    vec: Vec<f32>,
}

pub struct InfoClassifierIndex {
    protos: Vec<EmbeddedInfoProto>,
    green_vecs: Vec<Vec<f32>>,
    boilerplate_vecs: Vec<Vec<f32>>,
}

impl InfoClassifierIndex {
    pub fn empty() -> Self {
        Self {
            protos: Vec::new(),
            green_vecs: Vec::new(),
            boilerplate_vecs: Vec::new(),
        }
    }

    pub fn build(clf: &EmbeddingClassifier) -> Result<Self, String> {
        let mut all_texts: Vec<String> = Vec::new();
        // tag: 0 = proto(g_idx), 1 = green, 2 = boilerplate
        let mut tags: Vec<(u8, usize)> = Vec::new();

        for (g_idx, grp) in INFO_GROUPS.iter().enumerate() {
            for &p in grp.prompts {
                all_texts.push(format!("task: classification | query: {p}"));
                tags.push((0, g_idx));
            }
        }
        for &g in INFO_GREEN_PROMPTS {
            all_texts.push(format!("task: classification | query: {g}"));
            tags.push((1, 0));
        }
        for &b in INFO_BOILERPLATE_PROMPTS {
            all_texts.push(format!("task: classification | query: {b}"));
            tags.push((2, 0));
        }

        let refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();
        let vecs = clf.embed_texts_with_progress(&refs, |_, _| {})?;

        let mut protos = Vec::new();
        let mut green_vecs = Vec::new();
        let mut boilerplate_vecs = Vec::new();

        for ((kind, idx), vec) in tags.into_iter().zip(vecs) {
            match kind {
                0 => protos.push(EmbeddedInfoProto {
                    group_idx: idx,
                    vec,
                }),
                1 => green_vecs.push(vec),
                _ => boilerplate_vecs.push(vec),
            }
        }

        Ok(Self {
            protos,
            green_vecs,
            boilerplate_vecs,
        })
    }
}

fn is_public_function_email(email: &str) -> bool {
    let lower = email.trim().to_lowercase();
    let Some(local) = lower.split('@').next() else {
        return false;
    };
    matches!(
        local,
        "info"
            | "kontakt"
            | "kundtjanst"
            | "kundservice"
            | "registrator"
            | "vaxel"
            | "reception"
            | "press"
            | "kommunikation"
            | "noreply"
            | "no-reply"
    )
}

fn is_public_service_phone(phone: &str) -> bool {
    let digits: String = phone.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.starts_with("0771")
        || digits.starts_with("46771")
        || digits.starts_with("020")
        || digits.starts_with("4620")
        || digits == "1177"
        || digits == "11313"
        || digits == "11414"
}

fn has_explicit_draft_or_internal_markers(text: &str) -> bool {
    let lower = text.to_lowercase();
    const MARKERS: &[&str] = &[
        "utkast",
        "arbetsmaterial",
        "arbetskopia",
        "ej för spridning",
        "intern remiss",
        "internt underlag",
        "ej publicerad",
        "minnesanteckning",
        "arbetsplatsträff",
        "arbetsinstruktion",
        "semesterplanering",
        "sammanfattning av kurs",
        "kursen jag gick",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

/// Identifierar om dokumentets inledning tydligt visar att helheten är ett kurs-/seminarie-/föreläsningsreferat
/// eller en allmän metod-/utbildningstext (vilket ska motverka att lösryckta ord som "förhandling" eller
/// "extern kompetens" i ett senare stycke feltolkas som ett skarpt Rött ärende).
fn has_training_or_seminar_document_context(text: &str) -> bool {
    let intro: String = text.chars().take(450).collect::<String>().to_lowercase();
    const TRAINING_MARKERS: &[&str] = &[
        "sammanfattning av kurs",
        "kursen jag gick",
        "anteckningar från kurs",
        "seminarium",
        "seminarie",
        "föreläsning",
        "webbinarium",
        "utbildningsdag",
        "kursreferat",
        "workshopmetod",
    ];
    TRAINING_MARKERS.iter().any(|m| intro.contains(m))
}

/// Kontrollerar om ett stycke innehåller skarpa markörer för ett faktiskt rött individ-/sekretessärende.
/// Obs: Breda allmänna substantiv (såsom "patient", "diagnos", "sjukskrivning", "socialtjänst" eller "lösenord")
/// exkluderas medvetet här eftersom de förekommer i helt vanliga rutiner, strategidokument och 1177-texter.
fn has_explicit_red_case_markers(slice: &str) -> bool {
    let lower = slice.to_lowercase();
    const RED_MARKERS: &[&str] = &[
        "mottagningsanteckning",
        "daganteckning",
        "epikris",
        "metastas",
        "rehabsamtal",
        "rehabiliteringsutredning",
        "utmattningssymtom",
        "halvtidssjukskriv",
        "heltidssjukskriv",
        "elevhälsoteam",
        "åtgärdsprogram",
        "orosanmälan",
        "skyddade personuppgifter",
        "sekretessmarkering",
        "anbudspris",
        "anbudssekretess",
        "anbudsutvärdering",
        "konfidentiellt beredningsunderlag",
        "beredskapsplan",
        "larmkod",
        "pinkod",
        "puk-kod",
        "cve-",
        "opatchad",
        "sårbarhetsrapport",
        "penetrationstest",
        "bakgrundskontroll",
        "belastningsregister",
        "löneunderlag",
        "månadslön",
        "skriftlig varning",
        "disciplinärende",
    ];
    RED_MARKERS.iter().any(|m| lower.contains(m))
}

/// Hoppar över både enradiga rubriker och rena stolpordsstaplar utan meningsbyggnad
/// (t.ex. Word-listor som "Framgångsfaktorer / Ljud / Ljus / Placering / Beteende / Roll").
fn is_heading_or_keyword_stack(raw_slice: &str, clean_slice: &str) -> bool {
    if is_standalone_heading(raw_slice, clean_slice) {
        return true;
    }

    let lines: Vec<&str> = clean_slice
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return true;
    }

    // Om varje rad bara är korta nyckelord (<= 6 ord, genomsnitt <= 4 ord) och ingen rad avslutas med punkt/frågetecken/utropstecken
    let any_sentence_punct = lines
        .iter()
        .any(|l| l.ends_with('.') || l.ends_with('!') || l.ends_with('?'));
    if !any_sentence_punct && lines.len() >= 2 {
        let word_counts: Vec<usize> = lines
            .iter()
            .map(|l| l.split_whitespace().count())
            .collect();
        let max_words = word_counts.iter().copied().max().unwrap_or(0);
        let total_words: usize = word_counts.iter().sum();
        if max_words <= 6 && total_words <= lines.len() * 4 {
            return true;
        }
    }

    false
}

/// Avgör om en `Konto/Giro`-träff avser ett personligt bankkonto/IBAN (Röd) eller ett giro i publik kontext.
fn classify_account_matches(text: &str, structural_matches: &[Match]) -> (usize, usize) {
    let personal_spans: Vec<(usize, usize)> = RE_PERSONAL_BANK_ACCOUNT
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();
    let mut red_bank_accounts = personal_spans.len();
    let mut yellow_giro = 0usize;

    for m in structural_matches {
        if m.label != "Konto/Giro" {
            continue;
        }
        if personal_spans
            .iter()
            .any(|&(s, e)| !(m.end <= s || m.start >= e))
        {
            continue;
        }
        let upper = m.text.trim().to_ascii_uppercase();
        if upper.starts_with("SE") && upper.len() >= 20 {
            red_bank_accounts += 1;
            continue;
        }

        // Kolla om numret föregås av "Bankgiro"/"Plusgiro"/"BG"/"PG" -> ofta publik betalningsinfo, ej Röd
        let prefix_start = m.start.saturating_sub(24);
        // Säkerställ giltig UTF-8-gräns bakåt
        let mut safe_start = prefix_start;
        while safe_start > 0 && !text.is_char_boundary(safe_start) {
            safe_start -= 1;
        }
        let prefix = &text[safe_start..m.start];
        if !RE_PUBLIC_GIRO_CONTEXT.is_match(prefix) {
            yellow_giro += 1;
        }
    }

    (red_bank_accounts, yellow_giro)
}

pub fn build_handling_guidance(level: &str) -> HandlingGuidance {
    let disclaimer = "Detta är en automatisk, maskinell bedömning — inte en fastställd sanning. Eftersom den lokala AI-modellen granskar texten stycke för stycke utan att fullt ut förstå sammanhanget kan enstaka ord ge för högt eller för lågt utslag. Väg alltid in dokumentets helhet och syfte mot verksamhetens styrdokument.".to_string();

    match level {
        "red" => HandlingGuidance {
            heading: "Hur bör röd information hanteras?".to_string(),
            summary: "Röd information omfattas ofta av sekretess enligt lag eller kräver särskilt skydd eftersom den kan orsaka skada för enskild person, verksamheten eller regionen.".to_string(),
            storage_guidance: "Spara endast i avsett verksamhetssystem, diarium eller särskilt behörighetsstyrd yta — aldrig på öppna eller breda samarbetsytor.".to_string(),
            sharing_guidance: "Dela endast med specifikt behöriga mottagare via godkända, säkra kanaler. Klistra aldrig in omaskerad i öppna AI-tjänster.".to_string(),
            examples_note: "Typexempel på Röd: Patient- och elevuppgifter, rehabsamtal, sjukskrivningar, löneunderlag, anställningsavtal, bakgrundskontroller, pågående upphandlingar, politiskt underlag före offentliggörande, krisberedskap samt lås, larm och lösenord.".to_string(),
            disclaimer,
        },
        "yellow" => HandlingGuidance {
            heading: "Hur bör gul information hanteras?".to_string(),
            summary: "Gul information är intern arbetsinformation som vi vill hålla inom regionen, avdelningen eller arbetsgruppen — den kan delas med rätt personer men inte spridas fritt.".to_string(),
            storage_guidance: "Spara i verksamhetens avsedda interna samarbetsytor med rätt behörighetsstyrning för gruppen — inte på ytor som är öppna för alla eller externt.".to_string(),
            sharing_guidance: "Kan delas med berörda kollegor som behöver underlaget i arbetet. Kontrollera mottagare innan du delar — och tänk på att t.ex. medarbetarsamtal blir Röda om de innehåller rehab-, hälso- eller löneuppgifter.".to_string(),
            examples_note: "Typexempel på Gul: APT- och mötesanteckningar, arbetsmaterial och utkast, arbetsinstruktioner, processbeskrivningar, schema/semesterplanering, kursreferat samt opublicerade arbetskontaktuppgifter.".to_string(),
            disclaimer,
        },
        _ => HandlingGuidance {
            heading: "Hur får grön information hanteras?".to_string(),
            summary: "Grön information är sådant vi själva avser att dela öppet utan särskild risk.".to_string(),
            storage_guidance: "Kan lagras i gemensamma ytor och publiceras på extern webbplats eller offentliga anslagstavlor när materialet är beslutat och klargjort för publicering.".to_string(),
            sharing_guidance: "Kan spridas öppet. Obs: Att en handling är allmän och kan begäras ut är inte detsamma som att informationen är öppen för fri publicering — vid utlämnande görs alltid en sekretessbedömning i det enskilda fallet.".to_string(),
            examples_note: "Typexempel på Grön: Busstidtabeller, publicerade informationsbroschyrer, erbjudanden och godkänt material på extern webbplats.".to_string(),
            disclaimer,
        },
    }
}

/// Kör komplett informationsklassificering (Grön / Gul / Röd) utan att ändra/maskera källtexten.
pub fn classify_information_text<F>(
    text: &str,
    structural_matches: &[Match],
    classifier: Option<&EmbeddingClassifier>,
    mut on_progress: F,
) -> InfoClassificationReport
where
    F: FnMut(usize, usize),
{
    let t0 = std::time::Instant::now();
    let semantic_available = classifier.is_some();

    // 1. Analysera strukturella signaler (Lager 1 + Lager 2)
    let mut pnr_count = 0usize;
    let mut card_count = 0usize;
    let mut secret_key_count = 0usize;
    let mut internal_url_count = 0usize;
    let mut person_count = 0usize;
    let mut personal_email_count = 0usize;
    let mut personal_phone_count = 0usize;

    for m in structural_matches {
        match m.label.as_str() {
            "Personnummer" => pnr_count += 1,
            "Kortnummer" => card_count += 1,
            "Hemlighet/Nyckel" => {
                let lower = m.text.to_lowercase();
                if lower.starts_with("http://") || lower.starts_with("https://") {
                    internal_url_count += 1;
                } else {
                    secret_key_count += 1;
                }
            }
            "Person" => person_count += 1,
            "E-post" => {
                if !is_public_function_email(&m.text) {
                    personal_email_count += 1;
                }
            }
            "Telefonnummer" => {
                if !is_public_service_phone(&m.text) {
                    personal_phone_count += 1;
                }
            }
            _ => {}
        }
    }

    let (red_bank_account_count, yellow_giro_count) =
        classify_account_matches(text, structural_matches);

    // 2. Dokumentövergripande kontext + Semantisk avsnittsanalys (Lager 3 — EmbeddingGemma 2)
    let seminar_doc_context = has_training_or_seminar_document_context(text);
    let mut section_findings: Vec<InfoSectionFinding> = Vec::new();
    let mut paragraphs_scanned = 0usize;

    if let Some(clf) = classifier {
        let paragraphs = split_paragraphs(text);
        paragraphs_scanned = paragraphs.len();

        if !paragraphs.is_empty() {
            let cleaned_queries: Vec<String> = paragraphs
                .iter()
                .map(|(_, _, slice)| clean_for_embedding(slice))
                .collect();
            let formatted: Vec<String> = cleaned_queries
                .iter()
                .map(|clean| format!("task: classification | query: {clean}"))
                .collect();
            let refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();

            if let Ok(vecs) = clf.embed_texts_with_progress(&refs, &mut on_progress) {
                let idx = &clf.info_index;
                for (((p_idx, (byte_start, byte_end, raw_slice)), clean), vec) in paragraphs
                    .into_iter()
                    .enumerate()
                    .zip(cleaned_queries.iter())
                    .zip(vecs)
                {
                    if is_heading_or_keyword_stack(raw_slice, clean) {
                        continue;
                    }

                    // Hoppa över ren standardiserad e-postvarningsfot
                    let boilerplate_sim = idx
                        .boilerplate_vecs
                        .iter()
                        .map(|bv| dot(&vec, bv))
                        .fold(f32::NEG_INFINITY, f32::max);
                    if boilerplate_sim >= 0.85 {
                        continue;
                    }

                    let green_best = idx
                        .green_vecs
                        .iter()
                        .map(|gv| dot(&vec, gv))
                        .fold(f32::NEG_INFINITY, f32::max);

                    let mut best_red: Option<(usize, f32)> = None;
                    let mut best_yellow: Option<(usize, f32)> = None;

                    for proto in &idx.protos {
                        let sim = dot(&vec, &proto.vec);
                        let grp = &INFO_GROUPS[proto.group_idx];
                        if grp.level == "red" {
                            if best_red.is_none_or(|(_, b)| sim > b) {
                                best_red = Some((proto.group_idx, sim));
                            }
                        } else if best_yellow.is_none_or(|(_, b)| sim > b) {
                            best_yellow = Some((proto.group_idx, sim));
                        }
                    }

                    let y_sim_val = best_yellow.map(|(_, s)| s).unwrap_or(f32::NEG_INFINITY);
                    let explicit_red_in_slice = has_explicit_red_case_markers(raw_slice);

                    let mut matched_finding: Option<(String, &'static str, f32, String)> = None;

                    if let Some((r_idx, r_sim)) = best_red {
                        let margin_vs_green = r_sim - green_best;
                        let margin_vs_yellow = r_sim - y_sim_val;

                        // Om dokumentet i sin helhet är ett kurs-/seminariereferat och stycket saknar
                        // skarpa ärendeord krävs mycket hög likhet för att fälla dokumentet som RÖD.
                        let strong_red = if seminar_doc_context && !explicit_red_in_slice {
                            r_sim >= 0.845 && margin_vs_green >= 0.075 && margin_vs_yellow >= 0.025
                        } else if explicit_red_in_slice {
                            r_sim >= 0.770 && margin_vs_green >= 0.032 && margin_vs_yellow >= 0.002
                        } else {
                            (r_sim >= 0.805 && margin_vs_green >= 0.048 && margin_vs_yellow >= 0.012)
                                || (r_sim >= 0.835 && margin_vs_green >= 0.038 && margin_vs_yellow >= 0.008)
                        };

                        if strong_red {
                            let grp = &INFO_GROUPS[r_idx];
                            let conf =
                                (0.68 + ((margin_vs_green - 0.035) / 0.090) * 0.28).clamp(0.68, 0.98);
                            matched_finding = Some((
                                grp.category.to_string(),
                                "red",
                                conf,
                                grp.reason.to_string(),
                            ));
                        } else if !seminar_doc_context
                            && r_sim >= 0.785
                            && margin_vs_green >= 0.038
                            && margin_vs_yellow >= 0.006
                        {
                            // Gråzonsträff: liknar rött ämnesområde men når inte tröskeln för säker RÖD -> flagga som GUL gränsfallskontroll
                            let grp = &INFO_GROUPS[r_idx];
                            let conf =
                                (0.58 + ((margin_vs_green - 0.032) / 0.080) * 0.22).clamp(0.58, 0.78);
                            matched_finding = Some((
                                format!("Gränsfall — {}", grp.category),
                                "yellow",
                                conf,
                                format!(
                                    "Maskinell gränsfallsträff: Avsnittet innehåller ord som kan förekomma inom '{}', men kopplingen är inte entydig. Bedöm utifrån dokumentets helhet om detta avser ett verkligt ärende (Röd) eller generell/intern beskrivning (Gul).",
                                    grp.category
                                ),
                            ));
                        }
                    }

                    // Annars pröva GUL (interna APT-/mötesanteckningar, utkast, arbetsinstruktioner, kursreferat, medarbetarsamtal)
                    if matched_finding.is_none() {
                        if let Some((y_idx, y_sim)) = best_yellow {
                            let margin_vs_green = y_sim - green_best;
                            if y_sim >= 0.755 && margin_vs_green >= 0.022 {
                                let grp = &INFO_GROUPS[y_idx];
                                let conf = (0.60 + ((margin_vs_green - 0.022) / 0.090) * 0.34)
                                    .clamp(0.60, 0.96);
                                matched_finding = Some((
                                    grp.category.to_string(),
                                    "yellow",
                                    conf,
                                    grp.reason.to_string(),
                                ));
                            }
                        }
                    }

                    if let Some((category, level, conf, reason)) = matched_finding {
                        let clean_line = clean.replace('\n', " ");
                        let preview: String = clean_line.chars().take(135).collect();
                        let preview = if clean_line.chars().count() > 135 {
                            format!("{}…", preview.trim_end())
                        } else {
                            preview
                        };

                        section_findings.push(InfoSectionFinding {
                            paragraph_index: p_idx + 1,
                            preview,
                            start: byte_start,
                            end: byte_end,
                            category,
                            level: level.to_string(),
                            confidence: conf,
                            reason,
                        });
                    }
                }
            }
        }
    }

    // 3. Bygg förklarande signaler för "Vad innehåller informationen?"
    let mut signals: Vec<InfoSignal> = Vec::new();

    if pnr_count > 0 {
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "struktur".to_string(),
            title: format!("{pnr_count} personnummer / samordningsnummer"),
            detail: "Särskilt skyddsvärd personidentitet — omfattas normalt av strängare behörighets- och lagringskrav.".to_string(),
        });
    }
    if red_bank_account_count > 0 {
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "struktur".to_string(),
            title: format!("{red_bank_account_count} bankkontonummer / IBAN"),
            detail: "Personliga bankkontouppgifter räknas som särskilt skyddsvärda uppgifter (till skillnad från öppna Bankgiro/Plusgiro för inbetalning).".to_string(),
        });
    }
    if card_count > 0 {
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "struktur".to_string(),
            title: format!("{card_count} kortnummer (tjänste-/SITHS- eller betalkort)"),
            detail: "Kortnummer för e-legitimation eller betalning ska inte lagras eller spridas i öppna ytor.".to_string(),
        });
    }
    if secret_key_count > 0 {
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "struktur".to_string(),
            title: format!("{secret_key_count} lösenord, teknisk nyckel eller behörighetskod"),
            detail: "Inloggningsuppgifter, tekniska nycklar och behörighetsidentifierare kräver särskilt skydd.".to_string(),
        });
    }

    let red_sections: Vec<&InfoSectionFinding> = section_findings
        .iter()
        .filter(|f| f.level == "red")
        .collect();
    let yellow_sections: Vec<&InfoSectionFinding> = section_findings
        .iter()
        .filter(|f| f.level == "yellow")
        .collect();

    if !red_sections.is_empty() {
        let mut cats: Vec<&str> = Vec::new();
        for f in &red_sections {
            if !cats.contains(&f.category.as_str()) {
                cats.push(&f.category);
            }
        }
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "kontext".to_string(),
            title: format!(
                "{} avsnitt med indikerat rött innehåll ({})",
                red_sections.len(),
                cats.join(", ")
            ),
            detail: red_sections[0].reason.clone(),
        });
    }

    if !yellow_sections.is_empty() {
        let mut cats: Vec<&str> = Vec::new();
        for f in &yellow_sections {
            if !cats.contains(&f.category.as_str()) {
                cats.push(&f.category);
            }
        }
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "kontext".to_string(),
            title: format!(
                "{} avsnitt med internt arbetsinnehåll ({})",
                yellow_sections.len(),
                cats.join(", ")
            ),
            detail: yellow_sections[0].reason.clone(),
        });
    } else if has_explicit_draft_or_internal_markers(text) {
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "struktur".to_string(),
            title: "Markering för utkast, kursanteckning eller internt arbetsmaterial".to_string(),
            detail: "Texten innehåller uttryck som tyder på internt arbetsmaterial, utkast eller egna anteckningar för arbetsgruppen.".to_string(),
        });
    }

    if internal_url_count > 0 {
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "struktur".to_string(),
            title: format!("{internal_url_count} intern intranät- eller systemlänk"),
            detail: "Hänvisar till intranät eller interna arbetsytor som inte är avsedda för extern spridning.".to_string(),
        });
    }

    let pii_contact_total =
        person_count + personal_email_count + personal_phone_count + yellow_giro_count;
    if pii_contact_total > 0 {
        let mut parts = Vec::new();
        if person_count > 0 {
            parts.push(format!("{person_count} personnamn"));
        }
        if personal_email_count > 0 {
            parts.push(format!("{personal_email_count} arbetsepost/e-post"));
        }
        if personal_phone_count > 0 {
            parts.push(format!("{personal_phone_count} telefonnummer"));
        }
        if yellow_giro_count > 0 {
            parts.push(format!("{yellow_giro_count} giro-/betalningsuppgift"));
        }
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "struktur".to_string(),
            title: format!("Icke-publicerade person-/kontaktuppgifter ({})", parts.join(", ")),
            detail: "Harmlösa men personkopplade uppgifter (t.ex. namn, arbetsepost eller arbetstelefon i interna dokument) hanteras normalt som Gul information om dokumentet inte är avsett för offentlig publicering.".to_string(),
        });
    }

    // 4. Avgör maskinellt beslutsförslag (RÖD > GUL > GRÖN)
    let has_red = signals.iter().any(|s| s.level == "red");
    let has_yellow = signals.iter().any(|s| s.level == "yellow");

    let (level, badge_title, content_headline, content_explanation) = if has_red {
        (
            "red".to_string(),
            "Förslag: RÖD — Kräver särskilt skydd".to_string(),
            "Maskinell bedömning: Indikerar RÖD information".to_string(),
            "Analysen hittade uppgifter eller avsnitt som liknar sekretessbelagd eller särskilt skyddsvärd information. Kontrollera de markerade avsnitten i sitt sammanhang nedan.".to_string(),
        )
    } else if has_yellow {
        (
            "yellow".to_string(),
            "Förslag: GUL — Intern försiktighet".to_string(),
            "Maskinell bedömning: Indikerar GUL (intern) information".to_string(),
            "Analysen tyder på internt arbetsmaterial, mötes-/kursanteckningar, utkast eller opublicerade kontaktuppgifter som bör hållas inom behörig krets.".to_string(),
        )
    } else {
        signals.push(InfoSignal {
            level: "green".to_string(),
            source: "kontext".to_string(),
            title: "Inga röda eller gula markörer upptäcktes i styckena".to_string(),
            detail: if semantic_available {
                "Varken skyddsvärda sakavsnitt, interna utkastmarkörer, personnummer, bankkonton eller personkopplade kontaktuppgifter identifierades.".to_string()
            } else {
                "Inga personnummer, bankkonton, kortnummer, nycklar eller utkastmarkörer hittades (aktivera lokal AI för semantisk avsnittsanalys).".to_string()
            },
        });
        (
            "green".to_string(),
            "Förslag: GRÖN — Kan delas öppet".to_string(),
            "Maskinell bedömning: Indikerar GRÖN information".to_string(),
            "Texten liknar offentlig eller allmän information utan identifierade interna eller särskilt skyddsvärda uppgifter (förutsatt att materialet är avsett att publiceras).".to_string(),
        )
    };

    let handling = build_handling_guidance(&level);
    let scan_duration_ms = t0.elapsed().as_millis() as u64;

    InfoClassificationReport {
        level,
        badge_title,
        content_headline,
        content_explanation,
        signals,
        handling,
        section_findings,
        paragraphs_scanned,
        scan_duration_ms,
        semantic_available,
        judge: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn calibrate_green_yellow_red_classification() {
        let Ok(home) = std::env::var("HOME") else {
            return;
        };
        let dir = Path::new(&home)
            .join("Library/Application Support/com.patriksvensk.maskera/models/embeddinggemma2");
        if !dir.join("model_q4.onnx_data").exists() {
            return;
        }

        let clf = EmbeddingClassifier::load(&dir).expect("load EmbeddingGemma 2");

        // 1. GRÖN: Publicerad tidtabell / offentlig broschyr (inkl. publikt Bankgiro & kundtjänst)
        let green_doc = "\
Välkommen till länstrafikens vintertidtabell!\n\n\
Från och med den 15 december gäller nya avgångstider för regionbussarna mellan resecentrum och kransorterna. Bussarna i rusningstrafik avgår nu var tionde minut för att ge pendlare och skolungdomar en smidigare resa.\n\n\
Du kan ladda ner hela tidtabellen som broschyr på vår webbplats, betala företagskort via Bankgiro 5402-1234 eller mejla kundtjanst@lanstrafiken.se vid allmänna frågor.";
        let green_matches = vec![
            Match {
                text: "5402-1234".to_string(),
                label: "Konto/Giro".to_string(),
                start: green_doc.find("5402-1234").unwrap(),
                end: green_doc.find("5402-1234").unwrap() + 9,
                source: "regex".to_string(),
            },
            Match {
                text: "kundtjanst@lanstrafiken.se".to_string(),
                label: "E-post".to_string(),
                start: green_doc.find("kundtjanst@lanstrafiken.se").unwrap(),
                end: green_doc.find("kundtjanst@lanstrafiken.se").unwrap() + 26,
                source: "regex".to_string(),
            },
        ];
        let green_rep = classify_information_text(green_doc, &green_matches, Some(&clf), |_, _| {});
        println!(
            "1. GRÖN test => level={}, signals={:?}",
            green_rep.level,
            green_rep.signals.iter().map(|s| &s.title).collect::<Vec<_>>()
        );
        assert_eq!(
            green_rep.level, "green",
            "Offentlig broschyr/tidtabell med publikt Bankgiro ska klassas som GRÖN"
        );

        // 2. GUL (Verkligt regressionsfall): "Hybridmöten seminarie 210825.docx" med Word-\n\n mellan alla rader!
        let hybrid_raw_lines = "\
Hybridmöten Erik Mattson Ordrum 210824
Sammanfattning av kursen jag gick Såg den först när jag börjat anteckna. Kanske blir lite upprepningar men det får ni ha överinseende med. Finns även med saker jag inte hann med att anteckna.
Definition hybridmöte
https://www.svenskamoten.se/nyheter/digitala-moten-studios-hybridmoten-digitala-events-livestreams-vad-ar-vad/
Möjligheter
Fler kan vara med
Möjlighet till delvis parallella möten. Alla behöver inte göra samma sak.
Fallgropar
De som är på plats kan få mer makt.
De som är på plats kan känns sig störda av de som är med digitalt
Teknikberoende
Möjligheter
Mötesledare viktig. Måste vara mer aktiv än vid vanligt möte.
Ledaren har en viktig uppgift i att sammanlänka.
Ibland krävs 2 ledare. En fysisk och en digital som håller koll på chatt etc
Planering
Syfte och mål
Verktyg och plattform
Fördela ansvaret
Se över teknikbehov. Viktigt med både bra ljud och bild
Håll det kort. De som är med på distans har en annan tidsuppfattning. Undvik om möjligt att köra heldagsseminarier.
Arbeta med variation
Zoom, nytt sedan i somras. Menti app går lägga in direkt i Zoom.
Mötesrummet
Extra mikrofon är viktigt. Alternativt för mötesledare repetera frågor och svar
Olika exempel på utformning av mötesrum gicks igenom (se bilder i första länken)
Skillnad på utformning beroende på hur många deltagare som medverkar. Stående eller sittande ledare. Ledarens placering. Vi mellanstora hybridmöten (ca 25 st medverkande fysiskt) rekommenderas en talyta i mitten med egen skärm. Större hybridmöten typ 100 medverkande fysiskt liknar mer en TV produktion och där kanske man måste anlita extern kompetens om inte kompetensen finns i organisationen. Parallella sejourer. Ofta behov av programledare. Interaktivitetskrav påverkar utformningen.
Mötesmetoder
Samtala
Involvera
Inspirera
Bearbeta
Värdera
- Blixtintervju, moderator ställer frågor direkt, börja alltid med dem som inte är i rummet
- Gruppsamtal, break out rooms. Går även använda telefoner, surfplattor, facetime
- Parpromenad, som ett inslag i hybridmötet
- Dialog, fysiskt och/eller digitala whitboards. Tips miro.com mural.co .
- Värdering. T.ex menti
Framgångsfaktorer digitalt förhållningssätt
Ljud
Ljus
Placering
Beteende
Roll, privat eller professionell
Framgångsfaktor fysiskt förhållningssätt
Kom ihåg dem som medverkar digitalt
Tala till kameran och använd mikrofon när du ska säga något
Mötesdisciplin/mötesregler
Avbryt som talar för mycket
Regeltavla t.ex kamera på
Snabba åtgärder
Klargör syfte och formalia
Börja med distansdeltagarna
Repetera frågor och svar
Tänk parallella sejourer
Övrigt
Vid viss avancerade möten är hybridmöten inte att föredra som första alternativ. Kan t.ex gälla förhandlingar. Kanske bättre att köra fysiskt eller helt digitalt.";
        let hybrid_docx_style = hybrid_raw_lines.lines().collect::<Vec<_>>().join("\n\n");
        let hybrid_matches = vec![Match {
            text: "Erik Mattson".to_string(),
            label: "Person".to_string(),
            start: 12,
            end: 24,
            source: "ner".to_string(),
        }];
        let hybrid_rep =
            classify_information_text(&hybrid_docx_style, &hybrid_matches, Some(&clf), |_, _| {});
        println!(
            "2. HYBRIDMÖTEN seminarie (.docx-stil) => level={}, findings={:?}",
            hybrid_rep.level,
            hybrid_rep
                .section_findings
                .iter()
                .map(|f| (f.paragraph_index, &f.category, &f.level, f.confidence))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            hybrid_rep.level, "yellow",
            "Sammanfattning av kurs om hybridmöten får ALDRIG klassas som RÖD — ska landa på GUL!"
        );
        assert!(
            hybrid_rep.section_findings.iter().all(|f| f.level != "red"),
            "Inga röda avsnitt får triggas i hybridmötesdokumentet"
        );

        // 3. GUL: Vanlig APT-minnesanteckning + semesterplanering
        let apt_doc = "\
**MINNESANTECKNINGAR – ARBETSPLATSTRÄFF (APT)**\n\n\
* **Enhet:** Vårdcentralen Centrum\n\
* **Datum och tid:** 2026-11-11, kl. 13:00–14:30\n\
* **Plats:** Konferensrum Eken / Digitalt via intern videolänk\n\
* **Närvarande:** 18 medarbetare\n\n\
## 1. Verksamhetsläge och semesterplanering\n\n\
Beläggningen i telefonrådgivningen ligger något högre än motsvarande vecka föregående år. Utkast till arbetsinstruktioner och sommarens semesterplanering ligger i den interna mappen på intranätet.";
        let apt_rep = classify_information_text(apt_doc, &[], Some(&clf), |_, _| {});
        println!("3. GUL APT test => level={}", apt_rep.level);
        assert_eq!(apt_rep.level, "yellow");

        // 4. GUL: Allmänt medarbetarsamtal (utan rehab/sjukdom/lön)
        let medarb_doc = "\
Underlag inför årligt medarbetarsamtal på enheten.\n\n\
Vi går igenom hur arbetsåret har fungerat, trivsel i arbetsgruppen, önskemål om kompetensutveckling och mål för kommande år.";
        let medarb_rep = classify_information_text(medarb_doc, &[], Some(&clf), |_, _| {});
        println!(
            "4. GUL Medarbetarsamtal => level={}, findings={:?}",
            medarb_rep.level,
            medarb_rep
                .section_findings
                .iter()
                .map(|f| (&f.category, &f.level))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            medarb_rep.level, "yellow",
            "Allmänt medarbetarsamtal om kompetensutveckling ska klassas som GUL"
        );

        // 5. RÖD: Känsligt personalärende (rehabsamtal / sjukskrivning)
        let hr_red_doc = "\
Anteckning från rehabsamtal avseende medarbetare på enheten.\n\n\
Medarbetaren har varit sjukskriven på halvtid under hösten på grund av utmattningssymtom och läkarintyg har inkommit. Vi upprättar en individuell plan för arbetsanpassning och uppföljning med företagshälsovården.";
        let hr_rep = classify_information_text(hr_red_doc, &[], Some(&clf), |_, _| {});
        assert_eq!(hr_rep.level, "red");

        // 6. RÖD: Pågående upphandling under anbudssekretess
        let proc_red_doc = "\
Utvärderingsunderlag inför tilldelningsbeslut — pågående upphandling av medicinteknisk utrustning (omfattas av absolut anbudssekretess).\n\n\
Leverantör A har lämnat ett konfidentiellt anbudspris om 14,2 MSEK med 18 procents volymrabatt, medan Leverantör B ligger på 16,8 MSEK före förhandling.";
        let proc_rep = classify_information_text(proc_red_doc, &[], Some(&clf), |_, _| {});
        assert_eq!(proc_rep.level, "red");

        // 7. RÖD: Krisberedskap samt fysisk säkerhet (lås & larm)
        let sec_red_doc = "\
Beredskapsplan och skalskyddsrutin för tekniska försörjningsbyggnaden.\n\n\
Reservkraftaggregatet testkörs varje månad. Larmkod för nattlarm i sektion B byts kvartalsvis och huvudnycklar till lås och passersystem kvitteras ut i säkerhetsskåpet.";
        let sec_rep = classify_information_text(sec_red_doc, &[], Some(&clf), |_, _| {});
        println!("7. RÖD Beredskap/Lås & larm => level={}", sec_rep.level);
        assert_eq!(sec_rep.level, "red");

        // 8. RÖD: Privat bankkontonummer i annars harmlös utbetalningsnotis
        let bank_doc = "Utbetalning av ersättning sker till medborgarens bankkonto 8327-9, 123 456 789-0 nästa vecka.";
        let bank_rep = classify_information_text(bank_doc, &[], Some(&clf), |_, _| {});
        println!(
            "8. RÖD Bankkonto => level={}, signals={:?}",
            bank_rep.level,
            bank_rep.signals.iter().map(|s| &s.title).collect::<Vec<_>>()
        );
        assert_eq!(bank_rep.level, "red");

        // 9. RÖD: Politisk information före offentliggörande + Elevuppgifter
        let pol_doc = "\
Konfidentiellt beredningsunderlag inför regionstyrelsens arbetsutskott (ej offentliggjort förslag).\n\n\
Förslaget innebär en omstrukturering av tre enheter och omfattas av sekretess under pågående politisk beredning fram till offentliggörandet nästa månad.";
        let pol_rep = classify_information_text(pol_doc, &[], Some(&clf), |_, _| {});
        println!("9. RÖD Politik före offentliggörande => level={}", pol_rep.level);
        assert_eq!(pol_rep.level, "red");

        let elev_doc = "\
Minnesanteckning från elevhälsoteamet gällande elev i årskurs 2.\n\n\
Eleven har haft hög frånvaro under terminen och utreds för behov av särskilt stöd samt anpassad studiegång i samråd med skolpsykolog.";
        let elev_rep = classify_information_text(elev_doc, &[], Some(&clf), |_, _| {});
        println!("10. RÖD Elevuppgifter => level={}", elev_rep.level);
        assert_eq!(elev_rep.level, "red");

        // 11. Lösryckt mittstycke om hybridmöten UTAN kurs-intro på rad 1 — får inte heller bli RÖD!
        let hybrid_excerpt_no_intro = "\
Skillnad på utformning beroende på hur många deltagare som medverkar. Stående eller sittande ledare. Ledarens placering. Vid mellanstora hybridmöten (ca 25 st medverkande fysiskt) rekommenderas en talyta i mitten med egen skärm. Större hybridmöten typ 100 medverkande fysiskt liknar mer en TV produktion och där kanske man måste anlita extern kompetens om inte kompetensen finns i organisationen.\n\n\
Vid vissa avancerade möten är hybridmöten inte att föredra som första alternativ. Kan t.ex gälla förhandlingar. Kanske bättre att köra fysiskt eller helt digitalt.";
        let excerpt_rep =
            classify_information_text(hybrid_excerpt_no_intro, &[], Some(&clf), |_, _| {});
        println!(
            "11. Hybridmöten utan kurs-intro => level={}, findings={:?}",
            excerpt_rep.level, excerpt_rep.section_findings
        );
        assert_ne!(
            excerpt_rep.level, "red",
            "Generell metodtext om hybridmöten utan kurs-intro får inte heller klassas som RÖD"
        );

        // 12. GUL: Intern IT-/journalrutin med generella termer (patientjournaler, diagnos, sjukskrivning, lösenord)
        let routine_doc = "\
Rutin för behörighet, lösenordsbyte och dokumentation i patientjournaler\n\
Fastställd av: Helena Berg, vård-IT (helena.berg@regionexempel.se)\n\n\
Varje vårdkontakt ska dokumenteras sakligt i patientjournalen. Vid utskrivning anges huvuddiagnos enligt ICD-10 och vid behov av sjukskrivning utfärdas elektroniskt läkarintyg via journalsystemet. Reservlösenord byts var 90:e dag och får aldrig skrivas ned.";
        let routine_matches = vec![Match {
            text: "Helena Berg".to_string(),
            label: "Person".to_string(),
            start: routine_doc.find("Helena Berg").unwrap(),
            end: routine_doc.find("Helena Berg").unwrap() + 11,
            source: "kblab".to_string(),
        }];
        let routine_rep =
            classify_information_text(routine_doc, &routine_matches, Some(&clf), |_, _| {});
        println!("12. GUL Intern journal-/lösenordsrutin => level={}", routine_rep.level);
        assert_eq!(
            routine_rep.level, "yellow",
            "Generell rutin som nämner patientjournal, diagnos, sjukskrivning och lösenord ska bli GUL, inte RÖD"
        );

        // 13. GRÖN: Publik 1177-invånarguide om cancerutredning, diagnos och journal på nätet
        let public_1177_doc = "\
Så fungerar standardiserat vårdförlopp vid misstanke om cancer — Information till dig som invånare på 1177.se\n\n\
När din läkare på vårdcentralen misstänker att dina symtom kan bero på cancer kan du remitteras till ett standardiserat vårdförlopp (SVF) så att du får svar och eventuell diagnos så snabbt och tryggt som möjligt.\n\n\
Du får kontakt med en kontaktsjuksköterska och kan läsa dina anteckningar i din patientjournal genom att logga in på 1177.se.";
        let public_1177_rep =
            classify_information_text(public_1177_doc, &[], Some(&clf), |_, _| {});
        println!("13. GRÖN Publik 1177-guide => level={}", public_1177_rep.level);
        assert_eq!(
            public_1177_rep.level, "green",
            "Publik 1177-invånarinformation om cancerutredning och patientjournal ska bli GRÖN"
        );

        // 14. RÖD: Klinisk patientanteckning helt utan personnummer
        let clinical_no_pnr = "\
Mottagningsanteckning — Onkologmottagningen\n\
Patient: Karin Lindqvist, 64 år\n\n\
Karin inkommer för uppföljning efter DT torax/buk med tilltagande ryggsmärta. Radiologin visar förändringar i kota L2–L3 med misstanke om skelettmetastasering. Vi justerar smärtlindringen med oxikodon 10 mg x 2 och skickar remiss till kurator.";
        let clinical_rep = classify_information_text(clinical_no_pnr, &[], Some(&clf), |_, _| {});
        println!("14. RÖD Klinisk patientnotis utan pnr => level={}", clinical_rep.level);
        assert_eq!(
            clinical_rep.level, "red",
            "Verklig klinisk patientanteckning ska bli RÖD även när personnummer saknas"
        );

        // 15. RÖD: Mjukt formulerad HR-/rehabuppföljning utan ordet 'rehabsamtal'
        let soft_rehab_doc = "\
Anteckning från uppföljningssamtal — Enhet Ekonomi\n\
Medarbetare: Anders Ek\n\n\
Anders berättar att sömnsvårigheterna och hjärtklappningen har förvärrats efter vårens arbetsbelastning. Företagshälsovårdens läkare bedömer att det rör sig om utmattningssymtom och har förlängt hans halvtidssjukskrivning till den 30 november.";
        let soft_rehab_rep = classify_information_text(soft_rehab_doc, &[], Some(&clf), |_, _| {});
        println!("15. RÖD Mjuk HR-/rehabuppföljning => level={}", soft_rehab_rep.level);
        assert_eq!(
            soft_rehab_rep.level, "red",
            "Individuell ohälso-/sjukskrivningsuppföljning ska bli RÖD även utan ordet rehabsamtal"
        );
    }
}
