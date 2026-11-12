use crate::ner::Match;

#[derive(Debug, Clone)]
struct WordSpan<'a> {
    start: usize,
    end: usize,
    end_with_dot: usize,
    lower: String,
    ends_with_hyphen: bool,
    followed_by_dot: bool,
    #[allow(dead_code)]
    raw: &'a str,
}

/// Identifierar svenska yrkestitlar, roller och befattningar (Steg 1A — sammansättnings-
/// och ordlistemotor i Lager 1, 0 MB extra, < 1 ms).
///
/// Hanterar:
/// - Regelbundna svenska befattningsefterled i obestämd/bestämd form, singular/plural och genitiv
///   (t.ex. `informationssäkerhetssamordnaren`, `verksamhetschef`, `biståndshandläggarna`).
/// - Sammansatta roller med bindestreck och samordnade förled
///   (t.ex. `IT-säkerhetsspecialist`, `vård- och omsorgschef`, `barn- och utbildningschefen`).
/// - Förställda befattningsattribut (`biträdande`, `tf.`, `tillförordnad`, `leg.`, `vice`, `klinisk` m.fl.).
/// - Kurerad basordlista för grundyrken som saknar sammansättningsefterled (`läkare`, `jurist`,
///   `barnmorska`, `undersköterska`, `socionom`, `registrator`, `arkivarie` m.fl.).
/// - Undantagsfilter mot falska vänner (`tekniker` i betydelsen metoder, `samtalstekniker`,
///   `halvledare`, `reflektor`, `sektor`, `faktor`, `personuppgiftsansvarig`, `skyddsvärd` m.fl.).
pub fn find_role_matches(text: &str) -> Vec<Match> {
    let words = tokenize_words(text);
    if words.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let mut i = 0usize;

    while i < words.len() {
        // 1. Pröva fasta flerordsuttryck först (längst match vinner)
        if let Some((end_idx, valid)) = match_multiword_phrase(text, &words, i) {
            if valid {
                let first_idx = expand_pre_modifiers(text, &words, i);
                let byte_start = words[first_idx].start;
                let byte_end = words[end_idx].end;
                matches.push(Match {
                    text: text[byte_start..byte_end].to_string(),
                    label: "Roll/Titel".to_string(),
                    start: byte_start,
                    end: byte_end,
                    source: "lexicon".to_string(),
                });
                i = end_idx + 1;
                continue;
            }
        }

        // 2. Pröva enskilt ord (sammansättningsefterled eller kurerat grundyrke)
        let candidate = words[i].lower.trim_end_matches('-');
        if !words[i].ends_with_hyphen && is_role_word(candidate) {
            let first_idx = expand_pre_modifiers(text, &words, i);
            let byte_start = words[first_idx].start;
            let byte_end = words[i].end;
            matches.push(Match {
                text: text[byte_start..byte_end].to_string(),
                label: "Roll/Titel".to_string(),
                start: byte_start,
                end: byte_end,
                source: "lexicon".to_string(),
            });
        }

        i += 1;
    }

    // Slå samman eller deduplicera om en modifierare råkat överlappa en tidigare träff
    let mut deduped: Vec<Match> = Vec::with_capacity(matches.len());
    for m in matches {
        if let Some(last) = deduped.last_mut() {
            if m.start < last.end {
                if m.end > last.end {
                    last.end = m.end;
                    last.text = text[last.start..last.end].to_string();
                }
                continue;
            }
        }
        deduped.push(m);
    }

    deduped
}

fn tokenize_words<'a>(text: &'a str) -> Vec<WordSpan<'a>> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        let ch = text[idx..].chars().next().unwrap();
        let ch_len = ch.len_utf8();

        if ch.is_alphabetic() || ch.is_ascii_digit() {
            let start = idx;
            let mut end = idx + ch_len;
            let mut has_alpha = ch.is_alphabetic();

            while end < bytes.len() {
                let next_ch = text[end..].chars().next().unwrap();
                let next_len = next_ch.len_utf8();
                if next_ch.is_alphabetic() || next_ch.is_ascii_digit() {
                    if next_ch.is_alphabetic() {
                        has_alpha = true;
                    }
                    end += next_len;
                } else if next_ch == ':' {
                    // Tillåt kolon inuti ord för t.ex. "vd:n", "vd:s", "1:e", "2:a"
                    let after_colon = end + next_len;
                    if after_colon < bytes.len() {
                        let c2 = text[after_colon..].chars().next().unwrap();
                        if c2.is_alphabetic() {
                            has_alpha = true;
                            end = after_colon + c2.len_utf8();
                            continue;
                        }
                    }
                    break;
                } else if next_ch == '-' {
                    let after_hyphen = end + next_len;
                    if after_hyphen < bytes.len() {
                        let c2 = text[after_hyphen..].chars().next().unwrap();
                        if c2.is_alphabetic() || c2.is_ascii_digit() {
                            if c2.is_alphabetic() {
                                has_alpha = true;
                            }
                            end = after_hyphen + c2.len_utf8();
                            continue;
                        } else if c2 == ' ' || c2 == '\t' {
                            // Samordnat förled som "vård- och omsorgschef"
                            end = after_hyphen;
                            break;
                        }
                    }
                    break;
                } else {
                    break;
                }
            }

            if has_alpha {
                let raw = &text[start..end];
                let ends_with_hyphen = raw.ends_with('-');
                let followed_by_dot = text[end..].starts_with('.');
                let end_with_dot = if followed_by_dot { end + 1 } else { end };
                out.push(WordSpan {
                    start,
                    end,
                    end_with_dot,
                    lower: raw.to_lowercase(),
                    ends_with_hyphen,
                    followed_by_dot,
                    raw,
                });
            }

            idx = end;
        } else {
            idx += ch_len;
        }
    }

    out
}

#[inline]
fn only_spaces_between(text: &str, left_end: usize, right_start: usize) -> bool {
    if right_start < left_end || right_start - left_end > 4 {
        return false;
    }
    let gap = &text[left_end..right_start];
    !gap.is_empty() && gap.bytes().all(|b| b == b' ' || b == b'\t')
}

/// Kontrollerar fasta flerordsuttryck för befattningar (t.ex. "medicinskt ansvarig sjuksköterska",
/// "personlig assistent", "medicinsk sekreterare", "verkställande direktör").
fn match_multiword_phrase<'a>(
    text: &str,
    words: &[WordSpan<'a>],
    i: usize,
) -> Option<(usize, bool)> {
    // 4-ordsfraser
    if i + 3 < words.len()
        && only_spaces_between(text, words[i].end, words[i + 1].start)
        && only_spaces_between(text, words[i + 1].end, words[i + 2].start)
        && only_spaces_between(text, words[i + 2].end, words[i + 3].start)
    {
        let w0 = words[i].lower.as_str();
        let w1 = words[i + 1].lower.as_str();
        let w2 = words[i + 2].lower.as_str();
        let w3 = strip_genitive(&words[i + 3].lower);
        if w0 == "medicinskt"
            && matches!(w1, "ansvarig" | "ansvarige" | "ansvariga")
            && w2 == "för"
            && w3 == "rehabilitering"
        {
            return Some((i + 3, true));
        }
    }

    // 3-ordsfraser
    if i + 2 < words.len()
        && only_spaces_between(text, words[i].end, words[i + 1].start)
        && only_spaces_between(text, words[i + 1].end, words[i + 2].start)
    {
        let w0 = words[i].lower.as_str();
        let w1 = words[i + 1].lower.as_str();
        let w2 = strip_genitive(&words[i + 2].lower);

        if w0 == "medicinskt"
            && matches!(w1, "ansvarig" | "ansvarige" | "ansvariga")
            && matches!(
                w2,
                "sjuksköterska" | "sjuksköterskan" | "sjuksköterskor" | "sjuksköterskorna"
            )
        {
            return Some((i + 2, true));
        }
    }

    // 2-ordsfraser (där efterledet antingen inte matchar ensamt eller bör hållas ihop)
    if i + 1 < words.len() && only_spaces_between(text, words[i].end, words[i + 1].start) {
        let w0 = words[i].lower.as_str();
        let w1 = strip_genitive(&words[i + 1].lower);

        let is_two_word_role = (matches!(w0, "personligt" | "personliga" | "juridiskt" | "juridiska" | "fackligt" | "fackliga" | "regionalt" | "regionala")
            && matches!(w1, "ombud" | "ombudet" | "ombuden" | "skyddsombud" | "skyddsombudet" | "skyddsombuden"))
            || (matches!(w0, "verkställande")
                && matches!(w1, "direktör" | "direktören" | "direktörer" | "direktörerna"))
            || (matches!(w0, "personlig" | "personliga")
                && matches!(w1, "assistent" | "assistenten" | "assistenter" | "assistenterna"))
            || (matches!(w0, "medicinsk" | "medicinska")
                && matches!(w1, "sekreterare" | "sekreteraren" | "sekreterarna"))
            || (matches!(w0, "biomedicinsk" | "biomedicinska")
                && matches!(w1, "analytiker" | "analytikern" | "analytikerna"));

        if is_two_word_role {
            return Some((i + 1, true));
        }
    }

    None
}

/// Utökar en matchad befattning bakåt över samordnade bindestrecksförled ("vård- och omsorgschef")
/// samt förställda befattningsattribut ("biträdande", "tf.", "vice", "leg." m.fl.).
fn expand_pre_modifiers<'a>(text: &str, words: &[WordSpan<'a>], mut first_idx: usize) -> usize {
    // 1. Samordnat bindestrecksförled: "<förled>- och <roll>", t.ex. "barn- och utbildningschef"
    if first_idx >= 2
        && words[first_idx - 1].lower == "och"
        && words[first_idx - 2].ends_with_hyphen
        && !words[first_idx - 1].followed_by_dot
        && !words[first_idx - 2].followed_by_dot
        && only_spaces_between(text, words[first_idx - 2].end, words[first_idx - 1].start)
        && only_spaces_between(text, words[first_idx - 1].end, words[first_idx].start)
    {
        first_idx -= 2;
    }

    // 2. Upp till två förställda befattningsattribut (t.ex. "tf. biträdande verksamhetschef", "1:e vice ordförande")
    for _ in 0..2 {
        if first_idx == 0 {
            break;
        }
        let prev = &words[first_idx - 1];
        let prev_right_edge = if is_abbreviation_modifier(&prev.lower) && prev.followed_by_dot {
            prev.end_with_dot
        } else if !prev.followed_by_dot {
            prev.end
        } else {
            break;
        };

        if !only_spaces_between(text, prev_right_edge, words[first_idx].start) {
            break;
        }

        let next_lower = words[first_idx].lower.as_str();
        if is_role_pre_modifier(&prev.lower, prev.followed_by_dot, next_lower) {
            first_idx -= 1;
        } else {
            break;
        }
    }

    first_idx
}

#[inline]
fn is_abbreviation_modifier(w: &str) -> bool {
    matches!(w, "tf" | "stf" | "bitr" | "leg")
}

fn is_role_pre_modifier(w: &str, has_dot: bool, next_word: &str) -> bool {
    if is_abbreviation_modifier(w) {
        return true;
    }
    if has_dot {
        return false;
    }
    match w {
        "biträdande"
        | "tillförordnad"
        | "tillförordnade"
        | "ställföreträdande"
        | "vice"
        | "förste"
        | "andre"
        | "legitimerad"
        | "legitimerade"
        | "auktoriserad"
        | "auktoriserade"
        | "klinisk"
        | "kliniska"
        | "jourhavande"
        | "utsedd"
        | "utsedda"
        | "strategisk"
        | "strategiska"
        | "kvalificerad"
        | "kvalificerade"
        | "senior"
        | "seniora"
        | "junior"
        | "juniora"
        | "ansvarig"
        | "ansvarige" => true,
        // Ordningstal som "1:e" / "2:e" / "1:a" / "2:a" enbart framför "vice" eller typisk titel
        "1:e" | "2:e" | "1:a" | "2:a" => {
            next_word == "vice"
                || next_word.contains("socialsekreterare")
                || next_word.contains("ombudsman")
                || next_word.contains("arkivarie")
                || next_word.contains("inspektör")
        }
        _ => false,
    }
}

#[inline]
fn strip_genitive(word: &str) -> &str {
    if let Some(stem) = word.strip_suffix(":s") {
        if stem.chars().count() >= 2 {
            return stem;
        }
    }
    if let Some(stem) = word.strip_suffix('s') {
        if !stem.ends_with('s') && is_role_base_or_compound(stem) {
            return stem;
        }
    }
    word
}

fn is_role_word(lower: &str) -> bool {
    let char_count = lower.chars().count();
    if char_count < 2 {
        return false;
    }
    let normalized = strip_genitive(lower);
    if is_excluded_non_role(normalized) {
        return false;
    }
    is_role_base_or_compound(normalized)
}

/// Filtrerar bort vanliga icke-befattningar och falska vänner som råkar dela ändelse med yrkesord.
fn is_excluded_non_role(w: &str) -> bool {
    // Exakta stoppord (inkl. "tekniker" i obestämd plural som nästan alltid avser metoder/tekniker)
    if matches!(
        w,
        "tekniker"
            | "teknikerna"
            | "ledare"
            | "ledaren"
            | "ledarna"
            | "ombud"
            | "ombudet"
            | "ombuden"
            | "ansvarig"
            | "ansvarige"
            | "ansvariga"
            | "förare"
            | "föraren"
            | "förarna"
            | "mästare"
            | "mästaren"
            | "mästarna"
            | "värd"
            | "värden"
            | "värdar"
            | "värdarna"
            | "polis"
            | "polisen"
            | "poliser"
            | "poliserna"
    ) {
        return true;
    }

    // Ändelser som utgör fysikaliska/tekniska begrepp, juridiska rollbegrepp (GDPR PuA) eller allmänna gruppord
    const EXCLUDED_ENDINGS: &[&str] = &[
        // Falska vänner på -lektor / -rektor
        "reflektor",
        "reflektorn",
        "reflektorer",
        "reflektorerna",
        "selektor",
        "selektorn",
        "selektorer",
        "selektorerna",
        "kollektor",
        "kollektorn",
        "kollektorer",
        "kollektorerna",
        "elektor",
        "elektorn",
        "elektorer",
        "elektorerna",
        "korrektor",
        "korrektorn",
        "detektor",
        "detektorn",
        "detektorer",
        "projektor",
        "projektorn",
        "projektorer",
        "sektor",
        "sektorn",
        "sektorer",
        "sektorerna",
        "faktor",
        "faktorn",
        "faktorer",
        "faktorerna",
        "vektor",
        "vektorn",
        "vektorer",
        "reaktor",
        "reaktorn",
        "reaktorer",
        "traktor",
        "traktorn",
        "traktorer",
        // Fysikaliska ledare
        "halvledare",
        "halvledaren",
        "halvledarna",
        "supraledare",
        "supraledaren",
        "supraledarna",
        "värmeledare",
        "värmeledaren",
        "åskledare",
        "åskledaren",
        "jordledare",
        "jordledaren",
        "fasledare",
        "nolledare",
        "skyddsledare",
        "innerledare",
        "ytterledare",
        "ljusledare",
        "vågledare",
        "elledare",
        // Mjukvaru-/AI-termer på -assistent / -controller
        "ai-assistent",
        "ai-assistenten",
        "ai-assistenter",
        "ai-assistenterna",
        "röstassistent",
        "röstassistenten",
        "röstassistenter",
        "chattassistent",
        "chattassistenten",
        "hemassistent",
        "hemassistenten",
        "mikrocontroller",
        "mikrocontrollern",
        "microcontroller",
        // GDPR- & juridiska adjektiv på -ansvarig
        "personuppgiftsansvarig",
        "personuppgiftsansvarige",
        "personuppgiftsansvariga",
        "oansvarig",
        "oansvarige",
        "oansvariga",
        "medansvarig",
        "medansvarige",
        "medansvariga",
        "ensamansvarig",
        "ensamansvarige",
        "ensamansvariga",
        "huvudansvarig",
        "huvudansvarige",
        "huvudansvariga",
        "skadeståndsansvarig",
        "skadeståndsansvariga",
        "betalningsansvarig",
        "betalningsansvariga",
        "straffansvarig",
        "straffansvariga",
        "yttrandefrihetsansvarig",
        // Kommersiella utlämningsställen på -ombud
        "postombud",
        "postombudet",
        "postombuden",
        "paketombud",
        "paketombudet",
        "paketombuden",
        "spelombud",
        "spelombudet",
        "apoteksombud",
        "apoteksombudet",
        "systembolagsombud",
        // Idrottsmästare m.m.
        "världsmästare",
        "världsmästaren",
        "europamästare",
        "europamästaren",
        "svenskmästare",
        "stormästare",
        "stormästaren",
        "läromästare",
        "läromästaren",
        // Verbalsubstantiv på -förare som inte är yrkesförare
        "medförare",
        "anförare",
        "anföraren",
        "tillförare",
        "överförare",
        "bokförare",
        // Generella icke-identifierande gruppord
        "medarbetare",
        "medarbetaren",
        "medarbetarna",
        "arbetstagare",
        "arbetstagaren",
        "arbetstagarna",
        "arbetsgivare",
        "arbetsgivaren",
        "arbetsgivarna",
        "deltagare",
        "deltagaren",
        "deltagarna",
        "användare",
        "användaren",
        "användarna",
        "invånare",
        "invånaren",
        "invånarna",
        "medborgare",
        "medborgaren",
        "medborgarna",
        "besökare",
        "besökaren",
        "besökarna",
        "mottagare",
        "mottagaren",
        "mottagarna",
        "avsändare",
        "avsändaren",
        "avsändarna",
        "passagerare",
        "passageraren",
        "passagerarna",
        "vårdnadshavare",
        "vårdnadshavaren",
        "vårdnadshavarna",
        "uppdragsgivare",
        "uppdragsgivaren",
        "uppdragsgivarna",
        "beställare",
        "beställaren",
        "beställarna",
        "brukare",
        "brukaren",
        "brukarna",
    ];

    EXCLUDED_ENDINGS.iter().any(|suf| w.ends_with(suf))
}

fn is_role_base_or_compound(w: &str) -> bool {
    if is_excluded_non_role(w) {
        return false;
    }

    // 1. Korta akronymer / oregelbundna befattningsord (exakt matchning)
    if matches!(
        w,
        "vd"
            | "vd:n"
            | "ciso"
            | "ciso:n"
            | "cio"
            | "cfo"
            | "cto"
            | "teknikern"
            | "vårdbiträde"
            | "vårdbiträdet"
            | "vårdbiträden"
            | "vårdbiträdena"
            | "nämndeman"
            | "nämndemannen"
            | "nämndemän"
            | "nämndemännen"
            | "brandman"
            | "brandmannen"
            | "brandmän"
            | "brandmännen"
            | "polisman"
            | "polismannen"
            | "polismän"
            | "polismännen"
            | "rådman"
            | "rådmannen"
            | "rådmän"
            | "rådmännen"
            | "lagman"
            | "lagmannen"
            | "lagmän"
            | "lagmännen"
            | "landshövding"
            | "landshövdingen"
            | "landshövdingar"
            | "landshövdingarna"
            | "kyrkoherde"
            | "kyrkoherden"
            | "kyrkoherdar"
            | "kyrkoherdarna"
            | "diakon"
            | "diakonen"
            | "diakoner"
            | "diakonerna"
            | "präst"
            | "prästen"
            | "präster"
            | "prästerna"
            | "imam"
            | "imamen"
            | "imamer"
            | "imamerna"
            | "flygvärdinna"
            | "flygvärdinnan"
            | "flygvärdinnor"
            | "flygvärdinnorna"
    ) {
        return true;
    }

    // 2. Produktiva befattningsefterled som gäller BÅDE fristående och som sammansättning
    //    (kräver antingen exakt träff eller minst 2 teckens förled vid sammansättning, t.ex. "it-chef", "överläkare")
    const STANDALONE_AND_COMPOUND_FORMS: &[&str] = &[
        // Ledning & styrning
        "chef",
        "chefen",
        "chefer",
        "cheferna",
        "direktör",
        "direktören",
        "direktörer",
        "direktörerna",
        "rektor",
        "rektorn",
        "rektorer",
        "rektorerna",
        "lektor",
        "lektorn",
        "lektorer",
        "lektorerna",
        "ordförande",
        "ordföranden",
        "samordnare",
        "samordnaren",
        "samordnarna",
        "koordinator",
        "koordinatorn",
        "koordinatorer",
        "koordinatorerna",
        "strateg",
        "strategen",
        "strateger",
        "strategerna",
        "controller",
        "controllern",
        "controllers",
        "controllerna",
        // Handläggning, utredning, administration, juridik & ekonomi
        "handläggare",
        "handläggaren",
        "handläggarna",
        "utredare",
        "utredaren",
        "utredarna",
        "sekreterare",
        "sekreteraren",
        "sekreterarna",
        "administratör",
        "administratören",
        "administratörer",
        "administratörerna",
        "assistent",
        "assistenten",
        "assistenter",
        "assistenterna",
        "registrator",
        "registratorn",
        "registratorer",
        "registratorerna",
        "arkivarie",
        "arkivarien",
        "arkivarier",
        "arkivarierna",
        "bibliotekarie",
        "bibliotekarien",
        "bibliotekarier",
        "bibliotekarierna",
        "aktuarie",
        "aktuarien",
        "aktuarier",
        "aktuarierna",
        "jurist",
        "juristen",
        "jurister",
        "juristerna",
        "advokat",
        "advokaten",
        "advokater",
        "advokaterna",
        "åklagare",
        "åklagaren",
        "åklagarna",
        "domare",
        "domaren",
        "domarna",
        "notarie",
        "notarien",
        "notarier",
        "notarierna",
        "ombudsman",
        "ombudsmannen",
        "ombudsmän",
        "ombudsmännen",
        "ekonom",
        "ekonomen",
        "ekonomer",
        "ekonomerna",
        "revisor",
        "revisorn",
        "revisorer",
        "revisorerna",
        "upphandlare",
        "upphandlaren",
        "upphandlarna",
        "inköpare",
        "inköparen",
        "inköparna",
        "rekryterare",
        "rekryteraren",
        "rekryterarna",
        "kommunikatör",
        "kommunikatören",
        "kommunikatörer",
        "kommunikatörerna",
        "informatör",
        "informatören",
        "informatörer",
        "informatörerna",
        "analytiker",
        "analytikern",
        "analytikerna",
        "rådgivare",
        "rådgivaren",
        "rådgivarna",
        // IT, teknik & samhällsbyggnad
        "specialist",
        "specialisten",
        "specialister",
        "specialisterna",
        "utvecklare",
        "utvecklaren",
        "utvecklarna",
        "konsult",
        "konsulten",
        "konsulter",
        "konsulterna",
        "ingenjör",
        "ingenjören",
        "ingenjörer",
        "ingenjörerna",
        "arkitekt",
        "arkitekten",
        "arkitekter",
        "arkitekterna",
        "inspektör",
        "inspektören",
        "inspektörer",
        "inspektörerna",
        // Hälso- och sjukvård, omsorg & tandvård
        "läkare",
        "läkaren",
        "läkarna",
        "sköterska",
        "sköterskan",
        "sköterskor",
        "sköterskorna",
        "barnmorska",
        "barnmorskan",
        "barnmorskor",
        "barnmorskorna",
        "kurator",
        "kuratorn",
        "kuratorer",
        "kuratorerna",
        "psykolog",
        "psykologen",
        "psykologer",
        "psykologerna",
        "terapeut",
        "terapeuten",
        "terapeuter",
        "terapeuterna",
        "logoped",
        "logopeden",
        "logopeder",
        "logopederna",
        "dietist",
        "dietisten",
        "dietister",
        "dietisterna",
        "audionom",
        "audionomen",
        "audionomer",
        "audionomerna",
        "optiker",
        "optikern",
        "optikerna",
        "ortoptist",
        "ortoptisten",
        "ortoptister",
        "ortoptisterna",
        "naprapat",
        "naprapaten",
        "naprapater",
        "naprapaterna",
        "kiropraktor",
        "kiropraktorn",
        "kiropraktorer",
        "kiropraktorerna",
        "apotekare",
        "apotekaren",
        "apotekarna",
        "receptarie",
        "receptarien",
        "receptarier",
        "receptarierna",
        "psykiater",
        "psykiatern",
        "psykiatrer",
        "psykiatriker",
        "psykiatrikern",
        "psykiatrikerna",
        "kirurg",
        "kirurgen",
        "kirurger",
        "kirurgerna",
        "ortoped",
        "ortopeden",
        "ortopeder",
        "ortopederna",
        "tandhygienist",
        "tandhygienisten",
        "tandhygienister",
        "tandhygienisterna",
        "veterinär",
        "veterinären",
        "veterinärer",
        "veterinärerna",
        "socionom",
        "socionomen",
        "socionomer",
        "socionomerna",
        "skötare",
        "skötaren",
        "skötarna",
        "boendestödjare",
        "boendestödjaren",
        "boendestödjarna",
        "lokalvårdare",
        "lokalvårdaren",
        "lokalvårdarna",
        "sjukvårdare",
        "sjukvårdaren",
        "sjukvårdarna",
        "kriminalvårdare",
        "kriminalvårdaren",
        "kriminalvårdarna",
        // Medicinska & vetenskapliga -log/-nom-yrken (specifika stammar så "dialog"/"katalog" aldrig träffas)
        "kardiolog",
        "kardiologen",
        "kardiologer",
        "kardiologerna",
        "neurolog",
        "neurologen",
        "neurologer",
        "neurologerna",
        "onkolog",
        "onkologen",
        "onkologer",
        "onkologerna",
        "radiolog",
        "radiologen",
        "radiologer",
        "radiologerna",
        "patolog",
        "patologen",
        "patologer",
        "patologerna",
        "gynekolog",
        "gynekologen",
        "gynekologer",
        "gynekologerna",
        "urolog",
        "urologen",
        "urologer",
        "urologerna",
        "dermatolog",
        "dermatologen",
        "dermatologer",
        "dermatologerna",
        "reumatolog",
        "reumatologen",
        "reumatologer",
        "reumatologerna",
        "endokrinolog",
        "endokrinologen",
        "endokrinologer",
        "endokrinologerna",
        "anestesiolog",
        "anestesiologen",
        "anestesiologer",
        "anestesiologerna",
        "epidemiolog",
        "epidemiologen",
        "epidemiologer",
        "epidemiologerna",
        "mikrobiolog",
        "mikrobiologen",
        "mikrobiologer",
        "mikrobiologerna",
        "toxikolog",
        "toxikologen",
        "toxikologer",
        "toxikologerna",
        "farmakolog",
        "farmakologen",
        "farmakologer",
        "farmakologerna",
        "sociolog",
        "sociologen",
        "sociologer",
        "sociologerna",
        "kriminolog",
        "kriminologen",
        "kriminologer",
        "kriminologerna",
        "arkeolog",
        "arkeologen",
        "arkeologer",
        "arkeologerna",
        "geolog",
        "geologen",
        "geologer",
        "geologerna",
        "meteorolog",
        "meteorologen",
        "meteorologer",
        "meteorologerna",
        "ekolog",
        "ekologen",
        "ekologer",
        "ekologerna",
        "biolog",
        "biologen",
        "biologer",
        "biologerna",
        "ergonom",
        "ergonomen",
        "ergonomer",
        "ergonomerna",
        "agronom",
        "agronomen",
        "agronomer",
        "agronomerna",
        // Skola, utbildning & forskning
        "lärare",
        "läraren",
        "lärarna",
        "pedagog",
        "pedagogen",
        "pedagoger",
        "pedagogerna",
        "professor",
        "professorn",
        "professorer",
        "professorerna",
        "docent",
        "docenten",
        "docenter",
        "docenterna",
        "doktorand",
        "doktoranden",
        "doktorander",
        "doktoranderna",
        "forskare",
        "forskaren",
        "forskarna",
        // Service, säkerhet, politik & kollektivtrafik
        "poliskommissarie",
        "poliskommissarien",
        "poliskommissarier",
        "poliskommissarierna",
        "ordningsvakt",
        "ordningsvakten",
        "ordningsvakter",
        "ordningsvakterna",
        "skyddsvakt",
        "skyddsvakten",
        "skyddsvakter",
        "skyddsvakterna",
        "väktare",
        "väktaren",
        "väktarna",
        "vaktmästare",
        "vaktmästaren",
        "vaktmästarna",
        "verkmästare",
        "verkmästaren",
        "verkmästarna",
        "byggmästare",
        "byggmästaren",
        "byggmästarna",
        "köksmästare",
        "köksmästaren",
        "köksmästarna",
        "trädgårdsmästare",
        "trädgårdsmästaren",
        "trädgårdsmästarna",
        "brandmästare",
        "brandmästaren",
        "brandmästarna",
        "banmästare",
        "banmästaren",
        "banmästarna",
        "maskinmästare",
        "maskinmästaren",
        "maskinmästarna",
        "borgmästare",
        "borgmästaren",
        "borgmästarna",
        "hovmästare",
        "hovmästaren",
        "hovmästarna",
        "receptionist",
        "receptionisten",
        "receptionister",
        "receptionisterna",
        "telefonist",
        "telefonisten",
        "telefonister",
        "telefonisterna",
        "tolk",
        "tolken",
        "tolkar",
        "tolkarna",
        "kundvärd",
        "kundvärden",
        "kundvärdar",
        "kundvärdarna",
        "tågvärd",
        "tågvärden",
        "tågvärdar",
        "tågvärdarna",
        "servicevärd",
        "servicevärden",
        "servicevärdar",
        "servicevärdarna",
        "trygghetsvärd",
        "trygghetsvärden",
        "trygghetsvärdar",
        "trygghetsvärdarna",
        "stationsvärd",
        "stationsvärden",
        "stationsvärdar",
        "stationsvärdarna",
        "konferensvärd",
        "konferensvärden",
        "konferensvärdar",
        "konferensvärdarna",
        "kvartersvärd",
        "kvartersvärden",
        "kvartersvärdar",
        "kvartersvärdarna",
        "biljettkontrollant",
        "biljettkontrollanten",
        "biljettkontrollanter",
        "biljettkontrollanterna",
        "regionråd",
        "regionrådet",
        "regionråden",
        "kommunalråd",
        "kommunalrådet",
        "kommunalråden",
        "oppositionsråd",
        "oppositionsrådet",
        "oppositionsråden",
        "borgarråd",
        "borgarrådet",
        "borgarråden",
        "statsråd",
        "statsrådet",
        "statsråden",
    ];

    for suf in STANDALONE_AND_COMPOUND_FORMS {
        if w == *suf {
            return true;
        }
        if let Some(prefix) = w.strip_suffix(suf) {
            if valid_compound_prefix(prefix, 2) {
                return true;
            }
        }
    }

    // 3. Befattningsefterled som ENBART gäller i sammansättningar
    //    (-ledare, -ansvarig, -ombud, -förare, -planerare)
    const COMPOUND_ONLY_FORMS: &[(&str, usize)] = &[
        ("ledare", 3),
        ("ledaren", 3),
        ("ledarna", 3),
        ("ansvarig", 4),
        ("ansvarige", 4),
        ("ansvariga", 4),
        ("ombud", 4),
        ("ombudet", 4),
        ("ombuden", 4),
        ("förare", 3),
        ("föraren", 3),
        ("förarna", 3),
        ("planerare", 4),
        ("planeraren", 4),
        ("planerarna", 4),
    ];

    for (suf, min_unhyphenated_chars) in COMPOUND_ONLY_FORMS {
        if let Some(prefix) = w.strip_suffix(suf) {
            if prefix.ends_with('-') {
                // Tillåt kortare akronymförled med bindestreck som "it-ansvarig", "hr-ansvarig", "va-ansvarig"
                if valid_compound_prefix(prefix, 2) {
                    return true;
                }
            } else if valid_compound_prefix(prefix, *min_unhyphenated_chars) {
                return true;
            }
        }
    }

    // 4. Särskild hantering av "-tekniker" (undviker t.ex. "samtalstekniker", "studietekniker", "krypteringstekniker")
    for tech_suf in ["tekniker", "teknikern", "teknikerna"] {
        if let Some(prefix) = w.strip_suffix(tech_suf) {
            if is_human_technician_prefix(prefix) {
                return true;
            }
        }
    }

    false
}

#[inline]
fn valid_compound_prefix(prefix: &str, min_alpha: usize) -> bool {
    let trimmed = prefix.trim_end_matches('-');
    let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    alpha_count >= min_alpha
}

fn is_human_technician_prefix(prefix: &str) -> bool {
    let p = prefix.trim_end_matches('-');
    const TECH_ROLE_PREFIXES: &[&str] = &[
        "it",
        "drift",
        "drifts",
        "system",
        "nätverks",
        "support",
        "servicedesk",
        "helpdesk",
        "medicin",
        "medicinteknisk",
        "fastighets",
        "service",
        "el",
        "fordons",
        "buss",
        "spårvagns",
        "tåg",
        "laboratorie",
        "labb",
        "tand",
        "ortoped",
        "hjälpmedels",
        "brand",
        "larm",
        "ventilations",
        "kyl",
        "värme",
        "automations",
        "underhålls",
        "fält",
        "ljud",
        "ljus",
        "scen",
        "bild",
        "av",
        "bygg",
        "va",
        "gatu",
        "park",
        "kart",
        "mätnings",
        "data",
        "dator",
        "tele",
        "radio",
        "maskin",
        "process",
        "kvalitets",
        "miljö",
        "säkerhets",
    ];
    TECH_ROLE_PREFIXES.iter().any(|allowed| p == *allowed || p.ends_with(allowed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_swedish_compound_and_multiword_roles() {
        let sample = "Mötet leddes av biträdande verksamhetschef Karin Lind, \
                      informationssäkerhetssamordnaren, tf. enhetschef samt vård- och omsorgschefen. \
                      Även IT-säkerhetsspecialisten, medicinskt ansvarig sjuksköterska och personlig assistent deltog.";
        let found: Vec<String> = find_role_matches(sample)
            .into_iter()
            .map(|m| m.text)
            .collect();

        assert_eq!(
            found,
            vec![
                "biträdande verksamhetschef",
                "informationssäkerhetssamordnaren",
                "tf. enhetschef",
                "vård- och omsorgschefen",
                "IT-säkerhetsspecialisten",
                "medicinskt ansvarig sjuksköterska",
                "personlig assistent",
            ]
        );
    }

    #[test]
    fn detects_base_professions_and_genitives() {
        let sample = "Enligt överläkarens bedömning ska undersköterskorna, barnmorskan, \
                      skolkuratorn, kommunjuristen och vd:n informeras av driftteknikern.";
        let found: Vec<String> = find_role_matches(sample)
            .into_iter()
            .map(|m| m.text)
            .collect();

        assert_eq!(
            found,
            vec![
                "överläkarens",
                "undersköterskorna",
                "barnmorskan",
                "skolkuratorn",
                "kommunjuristen",
                "vd:n",
                "driftteknikern",
            ]
        );
    }

    #[test]
    fn ignores_false_friends_and_generic_words() {
        let sample = "Vi diskuterade olika tekniker och samtalstekniker i offentliga sektorn. \
                      Framgångsfaktorerna är att personuppgiftsansvarig skyddar skyddsvärda uppgifter \
                      när medarbetarna och deltagarna använder en AI-assistent eller halvledare med reflektor. \
                      Du är ansvarig för att kontakta polisen eller hämta hos postombudet.";
        let found = find_role_matches(sample);
        assert!(
            found.is_empty(),
            "Förväntade 0 träffar på falska vänner, men fick: {:?}",
            found
        );
    }
}

