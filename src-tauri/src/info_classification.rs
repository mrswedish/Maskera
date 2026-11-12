use serde::{Deserialize, Serialize};

use crate::classifier::{
    clean_for_embedding, dot, is_standalone_heading, split_paragraphs, EmbeddingClassifier,
};
use crate::Match;

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

/// Ankare för RÖD och GUL informationsklassning (separata från AI-maskeringsläget).
const INFO_GROUPS: &[InfoPrototypeGroup] = &[
    // ── RÖDA KATEGORIER (Information som kräver särskilt skydd) ──────────────
    InfoPrototypeGroup {
        category: "Patient- & Hälsouppgifter",
        level: "red",
        reason: "Innehåller medicinska uppgifter, diagnos, vårdplan, sjukskrivning eller behandling av enskild person.",
        prompts: &[
            "Patientjournal för enskild patient, sjukdomshistoria, medicinsk diagnos, läkemedelsbehandling, psykiatrisk bedömning, sjukskrivning eller vårdkontakt.",
            "Beskrivning av en specifik persons fysiska eller psykiska hälsotillstånd, symtom, operation, provsvar, missbruk eller funktionsnedsättning.",
            "Patienten vårdas på mottagning eller avdelning och genomgår medicinsk utredning, rehabilitering eller samtalsbehandling.",
        ],
    },
    InfoPrototypeGroup {
        category: "Skyddade personuppgifter, Socialtjänst & Rättsväsende",
        level: "red",
        reason: "Berör skyddade personuppgifter, socialtjänst, orosanmälan, lagöverträdelse eller andra särskilt integritetskänsliga förhållanden.",
        prompts: &[
            "Skyddade personuppgifter, sekretessmarkering, skyddad folkbokföring, hotbild eller kvinnofrid.",
            "Brottsmisstanke, polisanmälan, förundersökning, åtal, dom i brottmål, tvångsvård eller individutredning hos socialtjänsten.",
            "Orosanmälan kring barn eller individ, ekonomiskt bistånd, omhändertagande enligt LVU eller LVM eller uppgifter om facklig, politisk eller religiös tillhörighet.",
        ],
    },
    InfoPrototypeGroup {
        category: "Känsligt personalärende (HR)",
        level: "red",
        reason: "Berör enskild medarbetares rehabilitering, sjukfrånvaro, arbetsrättsliga åtgärd, disciplinärende eller individuella anställningsvillkor.",
        prompts: &[
            "Känsligt personalärende, rehabiliteringsutredning, upprepad korttidsfrånvaro, anpassning av arbetsuppgifter efter sjukskrivning eller läkarintyg för medarbetare.",
            "Skriftlig erinran, varning, misskötsamhet, omplacering, arbetsrättslig förhandling, uppsägning, avsked eller konfliktutredning kring namngiven medarbetare.",
            "Individuell lönesättning, löneöversyn för specifik anställd, kandidatbedömning med personliga omdömen eller psykologiskt testresultat i rekrytering.",
        ],
    },
    InfoPrototypeGroup {
        category: "Sekretessbelagd upphandling & Affärshemligheter",
        level: "red",
        reason: "Berör anbudssekretess, pågående upphandling före tilldelning, konfidentiella avtalsvillkor eller opublicerade ekonomiska uppgifter.",
        prompts: &[
            "Sekretessbelagda upphandlingsuppgifter, inkomna anbud under pågående anbudsutvärdering, anbudspriser eller utvärderingsprotokoll före tilldelningsbeslut.",
            "Konfidentiell affärsstrategi, företagshemlighet, opublicerat bokslut, vinstmarginal, uppköpsbud, prissättningsmodell eller konfidentiellt avtal.",
        ],
    },
    InfoPrototypeGroup {
        category: "Skyddsvärda säkerhetssystem & IT-säkerhet",
        level: "red",
        reason: "Innehåller skyddsvärda uppgifter om säkerhetssystem, behörigheter, sårbarheter, nätverksarkitektur eller inloggnings-/certifikatuppgifter.",
        prompts: &[
            "Skyddsvärda uppgifter om säkerhetssystem, fysiskt skalskydd, larmkoder, passersystem, reservkraft, säkerhetsskydd eller beredskapsplanering.",
            "Teknisk sårbarhetsrapport, penetrationstest, brandväggsregler, intern nätverkstopologi, aktivt dataintrång eller oskyddad databas i produktionsmiljö.",
            "Ditt SITHS eID certifikat på ditt SITHS-kort, tjänstekort, reservkort eller konsultkort med kortnummer, PIN-/PUK-kod eller inloggningscertifikat.",
        ],
    },

    // ── GULA KATEGORIER (Intern information att hantera med försiktighet) ────
    InfoPrototypeGroup {
        category: "APT- & Mötesanteckningar (Internt)",
        level: "yellow",
        reason: "Liknar interna minnesanteckningar, APT-protokoll eller arbetsmötesunderlag som är avsedda för berörda kollegor men inte för fri spridning.",
        prompts: &[
            "Minnesanteckningar från arbetsplatsträff APT, mötesanteckningar, intern dagordning, föregående protokoll, närvarande medarbetare, mötesledare och sekreterare.",
            "Anteckningar från enhetsmöte, veckomöte, styrgruppsmöte eller arbetsgruppsmöte om pågående arbete, uppföljning och åtgärdspunkter inom verksamheten.",
        ],
    },
    InfoPrototypeGroup {
        category: "Internt arbetsmaterial, Utkast & Planering",
        level: "yellow",
        reason: "Liknar internt arbetsmaterial, ej färdigställt utkast, bemannings-/verksamhetsplanering eller intern rutin.",
        prompts: &[
            "Internt arbetsmaterial, utkast till broschyr eller informationsblad under arbete, arbetskopia för synpunkter inför publicering eller internt diskussionsunderlag.",
            "Verksamhetsläge och intern bemanningsplanering inför jul, nyår och semester, schemaförslag, täckning av arbetspass, intern arbetsfördelning och lokalanvändning.",
            "Intern rutinbeskrivning, lathund för medarbetare på intranätet, intern projektplan, intern remissversion eller arbetsunderlag som inte är fastställt för allmänheten.",
        ],
    },
];

/// Ankare för GRÖN (offentlig/publik) information som är avsedd att kunna spridas öppet.
const INFO_GREEN_PROMPTS: &[&str] = &[
    "Publicerad broschyr för invånare, patientinformation för allmänheten, öppettider för mottagning eller besöksinformation på offentlig webbplats.",
    "Offentlig tidtabell för kollektivtrafik, busslinjer, tågtider, biljettpriser och reseinformation för resenärer.",
    "Publicerat pressmeddelande, offentlig nyhetsartikel, öppen inbjudan till publik konferens eller webbinarium för medborgare.",
    "Publicerad verksamhetsbeskrivning, årsredovisning som offentliggjorts, kurskatalog för skola eller allmän informationstext på extern hemsida.",
    "Allmän teknisk dokumentation för öppen källkod, generella skrivtips, recept eller offentlig myndighetsvägledning för allmänheten.",
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

fn has_explicit_draft_markers(text: &str) -> bool {
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
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

pub fn build_handling_guidance(level: &str) -> HandlingGuidance {
    let disclaimer = "Vägledande lokalt beslutsstöd — stanna upp och stäm alltid av din slutliga bedömning mot verksamhetens fastställda styrdokument.".to_string();

    match level {
        "red" => HandlingGuidance {
            heading: "Hur bör röd information hanteras?".to_string(),
            summary: "Information som kräver särskilt skydd eftersom den kan orsaka skada för en enskild person, verksamheten eller organisationen om den hamnar i fel händer.".to_string(),
            storage_guidance: "Lagring: Spara endast i avsett verksamhetssystem, diarium eller särskilt behörighetsstyrd yta — inte på öppna eller breda samarbetsytor.".to_string(),
            sharing_guidance: "Delning: Dela endast med specifikt behöriga mottagare via säkra kanaler enligt verksamhetens rutiner. Klistra aldrig in omaskerad i öppna AI-tjänster.".to_string(),
            examples_note: "Typexempel på Röd: Patientuppgifter, skyddade personuppgifter, sekretessbelagda upphandlingsuppgifter, känsliga personalärenden och skyddsvärda uppgifter om säkerhetssystem.".to_string(),
            disclaimer,
        },
        "yellow" => HandlingGuidance {
            heading: "Hur bör gul information hanteras?".to_string(),
            summary: "Information som används internt och som vi behöver vara försiktiga med. Den kan delas med rätt personer, men inte spridas fritt hur som helst.".to_string(),
            storage_guidance: "Lagring: Spara i interna arbetsytor med rätt behörighetskrets — inte på ytor som är öppna för hela organisationen eller publikt åtkomliga.".to_string(),
            sharing_guidance: "Delning: Kan delas med berörda kollegor och parter som behöver underlaget i arbetet, men kontrollera mottagarkretsen innan du delar vidare.".to_string(),
            examples_note: "Typexempel på Gul: Vanliga APT-anteckningar, interna arbetsmaterial, utkast till broschyrer, interna planeringar och mötesunderlag utan särskilt skyddsvärda uppgifter.".to_string(),
            disclaimer,
        },
        _ => HandlingGuidance {
            heading: "Hur får grön information hanteras?".to_string(),
            summary: "Information som är avsedd att vara offentlig och kan delas öppet utan särskild risk.".to_string(),
            storage_guidance: "Lagring: Kan lagras i öppna eller gemensamma samarbetsytor samt publiceras på externa webbplatser.".to_string(),
            sharing_guidance: "Delning: Kan spridas öppet både internt och externt (kontrollera dock alltid att dokumentet verkligen är färdigställt och godkänt för publicering).".to_string(),
            examples_note: "Typexempel på Grön: Publicerade broschyrer, tidtabeller, offentligt informationsmaterial och publicerade verksamhetsbeskrivningar.".to_string(),
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
    let mut account_count = 0usize;

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
            "Konto/Giro" => account_count += 1,
            _ => {}
        }
    }

    // 2. Semantisk avsnittsanalys (Lager 3 — EmbeddingGemma 2) mot Grön / Gul / Röd
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
                    if is_standalone_heading(raw_slice, clean) {
                        continue;
                    }

                    // Hoppa över ren standardiserad e-postvarningsfot ("Detta e-postmeddelande kommer från en avsändare utanför...")
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

                    // Pröva RÖD först: måste slå både gröna ankare och inte vara tydligt närmare ett vanligt gult APT-/planeringsankare
                    let mut matched_group: Option<(usize, &'static str, f32)> = None;
                    if let Some((r_idx, r_sim)) = best_red {
                        let margin_vs_green = r_sim - green_best;
                        let margin_vs_yellow = r_sim - y_sim_val;
                        if r_sim >= 0.76 && margin_vs_green >= 0.032 && margin_vs_yellow >= -0.005 {
                            let conf =
                                (0.64 + ((margin_vs_green - 0.032) / 0.090) * 0.34).clamp(0.64, 0.98);
                            matched_group = Some((r_idx, "red", conf));
                        }
                    }

                    // Annars pröva GUL (interna APT-/mötesanteckningar, utkast, planering)
                    if matched_group.is_none() {
                        if let Some((y_idx, y_sim)) = best_yellow {
                            let margin_vs_green = y_sim - green_best;
                            if y_sim >= 0.75 && margin_vs_green >= 0.020 {
                                let conf = (0.60 + ((margin_vs_green - 0.020) / 0.090) * 0.34)
                                    .clamp(0.60, 0.96);
                                matched_group = Some((y_idx, "yellow", conf));
                            }
                        }
                    }

                    if let Some((g_idx, level, conf)) = matched_group {
                        let grp = &INFO_GROUPS[g_idx];
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
                            category: grp.category.to_string(),
                            level: level.to_string(),
                            confidence: conf,
                            reason: grp.reason.to_string(),
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
            detail: "Direkta identitetsbeteckningar kräver normalt särskilt skydd och behörighetsstyrning.".to_string(),
        });
    }
    if card_count > 0 {
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "struktur".to_string(),
            title: format!("{card_count} kortnummer (tjänste-/SITHS- eller betalkort)"),
            detail: "Kortnummer för e-legitimation eller betalning ska inte spridas i öppna ytor.".to_string(),
        });
    }
    if secret_key_count > 0 {
        signals.push(InfoSignal {
            level: "red".to_string(),
            source: "struktur".to_string(),
            title: format!("{secret_key_count} teknisk nyckel / behörighetskod (t.ex. API-nyckel eller HSA-ID)"),
            detail: "Tekniska nycklar, certifikat och behörighetsidentifierare kräver särskilt skydd.".to_string(),
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
                "{} avsnitt med särskilt skyddsvärt innehåll ({})",
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
    } else if has_explicit_draft_markers(text) {
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "struktur".to_string(),
            title: "Markering för utkast / internt arbetsmaterial".to_string(),
            detail: "Texten innehåller ord som 'utkast', 'arbetsmaterial' eller 'minnesanteckning', vilket normalt innebär att den är avsedd för internt bruk.".to_string(),
        });
    }

    if internal_url_count > 0 {
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "struktur".to_string(),
            title: format!("{internal_url_count} intern intranät- eller systemlänk"),
            detail: "Hänvisar till interna sidor eller arbetsytor som inte är avsedda för extern spridning.".to_string(),
        });
    }

    let pii_contact_total = person_count + personal_email_count + personal_phone_count + account_count;
    if pii_contact_total > 0 {
        let mut parts = Vec::new();
        if person_count > 0 {
            parts.push(format!("{person_count} personnamn"));
        }
        if personal_email_count > 0 {
            parts.push(format!("{personal_email_count} e-postadress(er)"));
        }
        if personal_phone_count > 0 {
            parts.push(format!("{personal_phone_count} telefonnummer"));
        }
        if account_count > 0 {
            parts.push(format!("{account_count} konto-/gironummer"));
        }
        signals.push(InfoSignal {
            level: "yellow".to_string(),
            source: "struktur".to_string(),
            title: format!("Person- och kontaktuppgifter ({})", parts.join(", ")),
            detail: "Dokument med namngivna personer eller direkta kontaktuppgifter bör hanteras med omsorg även när de saknar sekretessuppgifter.".to_string(),
        });
    }

    // 4. Avgör total nivå (RÖD > GUL > GRÖN)
    let has_red = signals.iter().any(|s| s.level == "red");
    let has_yellow = signals.iter().any(|s| s.level == "yellow");

    let (level, badge_title, content_headline, content_explanation) = if has_red {
        (
            "red".to_string(),
            "RÖD — Kräver särskilt skydd".to_string(),
            "RÖD: Information som kräver särskilt skydd".to_string(),
            "Dokumentet innehåller uppgifter eller avsnitt som kan orsaka skada för en enskild person, verksamheten eller organisationen om de hamnar i fel händer.".to_string(),
        )
    } else if has_yellow {
        (
            "yellow".to_string(),
            "GUL — Var försiktig (Internt)".to_string(),
            "GUL: Information som vi behöver vara försiktiga med".to_string(),
            "Dokumentet bedöms innehålla intern arbetsinformation, mötesanteckningar, utkast eller personuppgifter som inte är avsedda att spridas fritt utan eftertanke.".to_string(),
        )
    } else {
        signals.push(InfoSignal {
            level: "green".to_string(),
            source: "kontext".to_string(),
            title: "Inga skyddsvärda eller interna markörer upptäcktes".to_string(),
            detail: if semantic_available {
                "Varken känsliga sakavsnitt, interna utkastmarkörer, personnummer eller personkopplade uppgifter hittades i texten.".to_string()
            } else {
                "Inga personnummer, kortnummer, nycklar eller utkastmarkörer hittades (aktivera lokal AI för djup kontextanalys).".to_string()
            },
        });
        (
            "green".to_string(),
            "GRÖN — Kan delas öppet".to_string(),
            "GRÖN: Information som kan delas öppet".to_string(),
            "Dokumentet liknar offentlig eller allmän information utan identifierade skyddsvärda eller interna uppgifter.".to_string(),
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

        // 1. GRÖN: Publicerad tidtabell / offentligt informationsmaterial
        let green_doc = "\
Välkommen till länstrafikens vintertidtabell!\n\n\
Från och med den 15 december gäller nya avgångstider för regionbussarna mellan resecentrum och kransorterna. Bussarna i rusningstrafik avgår nu var tionde minut för att ge pendlare och skolungdomar en smidigare resa.\n\n\
Du kan ladda ner hela tidtabellen som broschyr på vår webbplats eller kontakta kundtjänst om du har allmänna frågor om våra biljetter.";
        let green_rep = classify_information_text(green_doc, &[], Some(&clf), |_, _| {});
        println!(
            "GRÖN test => level={}, signals={:?}",
            green_rep.level,
            green_rep.signals.iter().map(|s| &s.title).collect::<Vec<_>>()
        );
        assert_eq!(
            green_rep.level, "green",
            "Offentlig broschyr/tidtabell ska klassas som GRÖN"
        );

        // 2. GUL: Vanlig APT-minnesanteckning med bemanningsplanering (ej patientdata)
        let apt_doc = "\
**MINNESANTECKNINGAR – ARBETSPLATSTRÄFF (APT)**\n\n\
* **Enhet:** Vårdcentralen Centrum\n\
* **Datum och tid:** 2026-11-11, kl. 13:00–14:30\n\
* **Plats:** Konferensrum Eken / Digitalt via intern videolänk\n\
* **Närvarande:** 18 medarbetare\n\n\
## 1. Verksamhetsläge och bemanning inför jul och nyår\n\n\
Beläggningen i telefonrådgivningen ligger något högre än motsvarande vecka föregående år. Förslag till slutligt julschema ligger i den interna schemamappen och eventuella byten av pass läggs in senast fredag innan schemat låses.";
        let apt_rep = classify_information_text(apt_doc, &[], Some(&clf), |_, _| {});
        println!(
            "GUL APT test => level={}, findings={:?}",
            apt_rep.level,
            apt_rep
                .section_findings
                .iter()
                .map(|f| (&f.category, &f.level, f.confidence))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            apt_rep.level, "yellow",
            "Vanlig APT-minnesanteckning och bemanningsplanering ska klassas som GUL (och varken Grön eller Röd!)"
        );

        // 3. RÖD: Känsligt personalärende (rehabilitering / varning) — ska bli RÖD enligt mötesmodellen
        let hr_red_doc = "\
Anteckning från rehabiliteringsmöte avseende medarbetare på enheten.\n\n\
Medarbetaren har varit sjukskriven på halvtid under hösten på grund av utmattningssymtom och läkarintyg har inkommit. Vi upprättar en individuell plan för arbetsanpassning och uppföljning med företagshälsovården.";
        let hr_rep = classify_information_text(hr_red_doc, &[], Some(&clf), |_, _| {});
        println!(
            "RÖD HR test => level={}, findings={:?}",
            hr_rep.level,
            hr_rep
                .section_findings
                .iter()
                .map(|f| (&f.category, &f.level, f.confidence))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            hr_rep.level, "red",
            "Känsligt personal-/rehabärende ska klassas som RÖD"
        );

        // 4. RÖD: Sekretessbelagd upphandling under pågående utvärdering
        let proc_red_doc = "\
Utvärderingsunderlag inför tilldelningsbeslut — pågående upphandling av medicinteknisk utrustning (omfattas av absolut anbudssekretess).\n\n\
Leverantör A har lämnat ett konfidentiellt anbudspris om 14,2 MSEK med 18 procents volymrabatt, medan Leverantör B ligger på 16,8 MSEK före förhandling.";
        let proc_rep = classify_information_text(proc_red_doc, &[], Some(&clf), |_, _| {});
        assert_eq!(
            proc_rep.level, "red",
            "Pågående upphandling med anbudspriser ska klassas som RÖD"
        );
    }
}
