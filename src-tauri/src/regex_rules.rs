use std::sync::LazyLock;

use regex::Regex;

use crate::ner::Match;
use crate::roles;

// ── Förkompilerade reguljära uttryck (kompileras en enda gång) ────────────────

static RE_PNR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\d{2})?\d{6}[-+]?\d{4}\b").unwrap());

static RE_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap()
});

static RE_PHONE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        (?:(?:\+46)|\b(?:0046|0))
        [\s\-]?
        [1-9]\d{0,2}
        (?:[\s\-]?\d{2,4}){2,4}\b
        ",
    )
    .unwrap()
});

static RE_ORGNR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\d{2})?\d{6}-\d{4}\b").unwrap());

static RE_IBAN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bSE\d{2}(?:[0-9A-Z]{20}|\s?(?:\d{4}\s?){4}\d{4})\b").unwrap()
});

static RE_BANK_ACCOUNT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?ix)",
        r"(?:\b\d{4}(?:-\d)?\s*,\s*\d{2,3}[\s\-]?\d{3}[\s\-]?\d{3,4}(?:-\d)?\b)",
        r"|(?:\b(?:bankkonto|personkonto|lönekonto|kontonummer)\s*[:=]?\s*\d{4}(?:-\d)?[\s,\-/]+\d[\d\s\-]{5,14}\b)"
    ))
    .unwrap()
});

static RE_BG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{3,4}[\s\-]\d{4}\b").unwrap());

static RE_PG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:\d{5,7}-\d|\d{1,3}(?:\s\d{2}){1,3}-\d|\d{2,7}-\d)\b").unwrap()
});

static RE_GIRO_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:bankgiro(?:nr|nummer)?|plusgiro(?:nr|nummer)?|postgiro|\bbg(?:-?nr)?\b|\bpg(?:-?nr)?\b)\s*[:.=]?\s*$").unwrap()
});

static RE_RANGE_UNIT_SUFFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:%|‰|år\b|års\b|mån(?:ad|ader)?\b|v(?:eckor|ecka)?\b|dag(?:ar)?\b|tim(?:mar|me)?\b|min(?:uter)?\b|sek(?:under)?\b|st\b|kr\b|kronor\b|milj\b|mkr\b|tkr\b|sid(?:or|an|a)?\b|kap(?:itel)?\b|p\b|poäng\b|personer\b|elever\b|barn\b|patienter\b)").unwrap()
});

static RE_IPV4: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b")
        .unwrap()
});

static RE_CARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d(?:[\s\-]?\d){12,18}\b").unwrap());

/// Tjänstekortsnummer / SITHS-kortnummer / certifikatserienummer (16–22 sammanhängande siffror)
static RE_SERVICE_CARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{16,22}\b").unwrap());

// ── Tekniska hemligheter, nycklar, HSA-ID och interna intranätlänkar ─────────
static RE_SECRETS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i:-----BEGIN\s+(?:[A-Z\s]+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:[A-Z\s]+)?PRIVATE\s+KEY-----)",
        r"|\b(?:sk-[a-zA-Z0-9_-]{20,64})\b",
        r"|\b(?:ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{40,90})\b",
        r"|\b(?:xox[baprs]-[0-9a-zA-Z]{10,48})\b",
        r"|\b(?:AKIA[0-9A-Z]{16})\b",
        r"|\b(?:AIza[0-9A-Za-z-_]{35})\b",
        r"|(?:(?i:bearer)\s+eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})",
        r"|\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b",
        r"|(?:(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis):\/\/[^\s:@/]+:[^\s:@/]+@[^\s/]+)",
        r#"|(?:https?:\/\/(?:[a-zA-Z0-9-]+\.)*(?:intranat|intranet|internal|corp|lan|local)(?:\.[a-zA-Z0-9-]+)+(?::\d+)?(?:\/[^\s"'<>)]*)?)"#,
        r"|\bSE\d{10}-[A-Za-z0-9]{3,15}\b",
        r#"|(?:(?i:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|lösenord)\s*[:=]\s*['"]?[A-Za-z0-9_\-.~!@#$%^&*()+=]{8,64}['"]?)"#
    ))
    .unwrap()
});

pub fn merge_regex_and_ner(regex_matches: Vec<Match>, ner_matches: Vec<Match>) -> Vec<Match> {
    let (hard_regex, lexicon_matches): (Vec<Match>, Vec<Match>) = regex_matches
        .into_iter()
        .partition(|m| m.source != "lexicon");

    let mut combined =
        Vec::with_capacity(hard_regex.len() + ner_matches.len() + lexicon_matches.len());

    // 1. Hårda strukturmönster (personnummer, e-post, kortnummer m.m.) vinner över BERT NER
    let mut kept_ner = Vec::with_capacity(ner_matches.len());
    for nm in ner_matches {
        let overlaps_hard = hard_regex
            .iter()
            .any(|rx| !(nm.end <= rx.start || nm.start >= rx.end));
        if !overlaps_hard {
            kept_ner.push(nm);
        }
    }

    // 2. Egennamn från BERT NER (t.ex. "Läkare Utan Gränser") vinner över ordlistebaserad Roll/Titel
    for lx in lexicon_matches {
        let overlaps_higher = hard_regex
            .iter()
            .chain(kept_ner.iter())
            .any(|hi| !(lx.end <= hi.start || lx.start >= hi.end));
        if !overlaps_higher {
            combined.push(lx);
        }
    }

    combined.extend(kept_ner);
    combined.extend(hard_regex);
    combined.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));

    let mut deduped = Vec::with_capacity(combined.len());
    let mut last_end = 0usize;
    for m in combined {
        if m.start >= last_end {
            last_end = m.end;
            deduped.push(m);
        }
    }
    deduped
}

fn valid_swedish_pnr(s: &str) -> bool {
    let digits: Vec<u8> = s
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();

    let offset = match digits.len() {
        10 => 0,
        12 => {
            let century = digits[0] * 10 + digits[1];
            if !matches!(century, 18..=20) {
                return false;
            }
            2
        }
        _ => return false,
    };

    let month = digits[offset + 2] * 10 + digits[offset + 3];
    let day = digits[offset + 4] * 10 + digits[offset + 5];

    (1..=12).contains(&month) && ((1..=31).contains(&day) || (61..=91).contains(&day))
}

fn valid_swedish_orgnr(s: &str) -> bool {
    let digits: Vec<u8> = s
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();

    let offset = match digits.len() {
        10 => 0,
        12 => {
            let prefix = digits[0] * 10 + digits[1];
            if prefix != 16 {
                return false;
            }
            2
        }
        _ => return false,
    };

    let middle = digits[offset + 2] * 10 + digits[offset + 3];
    middle >= 20
}

fn luhn_checksum_ok(s: &str, min_digits: usize, max_digits: usize) -> bool {
    let nums: Vec<u32> = s
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c.to_digit(10).unwrap())
        .collect();
    if !(min_digits..=max_digits).contains(&nums.len()) {
        return false;
    }
    let sum: u32 = nums
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                let x = d * 2;
                if x > 9 {
                    x - 9
                } else {
                    x
                }
            } else {
                d
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

fn luhn_valid(s: &str) -> bool {
    luhn_checksum_ok(s, 13, 19)
}

fn has_preceding_giro_keyword(text: &str, byte_start: usize) -> bool {
    let mut safe_start = byte_start.saturating_sub(28);
    while safe_start > 0 && !text.is_char_boundary(safe_start) {
        safe_start -= 1;
    }
    RE_GIRO_PREFIX.is_match(&text[safe_start..byte_start])
}

fn is_plausible_year_range(s: &str) -> bool {
    if let Some((left, right)) = s.split_once(['-', ' ']) {
        if left.len() == 4 && right.len() == 4 {
            if let (Ok(y1), Ok(y2)) = (left.parse::<u16>(), right.parse::<u16>()) {
                return (1900..=2099).contains(&y1) && (1900..=2099).contains(&y2);
            }
        }
    }
    false
}

fn overlaps_any(matches: &[Match], start: usize, end: usize) -> bool {
    matches.iter().any(|x| !(end <= x.start || start >= x.end))
}

pub fn run_regex(text: &str, requested: &[String]) -> Vec<Match> {
    let has = |name: &str| requested.iter().any(|e| e == name);
    let mut matches = Vec::new();

    // ── Tekniska hemligheter och API-nycklar (högsta prioritet) ──
    if has("Hemlighet/Nyckel") {
        for m in RE_SECRETS.find_iter(text) {
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Hemlighet/Nyckel".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Personnummer / Samordningsnummer ──
    if has("Personnummer") {
        for m in RE_PNR.find_iter(text) {
            let s = m.as_str();
            if !valid_swedish_pnr(s) || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Personnummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Organisationsnummer (mittenpar >= 20) ──
    if has("Org.nummer") {
        for m in RE_ORGNR.find_iter(text) {
            let s = m.as_str();
            if !valid_swedish_orgnr(s) || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Org.nummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── E-post ──
    if has("E-post") {
        for m in RE_EMAIL.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "E-post".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── IBAN & Bankkonto (före telefon/kort så SE-IBAN och clearing+konto inte klipps) ──
    if has("Konto/Giro") {
        for m in RE_IBAN.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
        for m in RE_BANK_ACCOUNT.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Telefonnummer ──
    if has("Telefonnummer") {
        for m in RE_PHONE.find_iter(text) {
            let s = m.as_str().trim_end();
            let end = m.start() + s.len();
            if overlaps_any(&matches, m.start(), end) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Telefonnummer".to_string(),
                start: m.start(),
                end,
                source: "regex".to_string(),
            });
        }
    }

    // ── Bankgiro & Plusgiro (med Luhn-validering och skydd mot intervall som "1-3 år" / "2024-2026") ──
    if has("Konto/Giro") {
        for m in RE_BG.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            let s = m.as_str();
            let has_kw = has_preceding_giro_keyword(text, m.start());
            if !has_kw {
                // Utan "Bankgiro"/"BG"-nyckelord krävs bindestreck, ej årtalsintervall och giltig Luhn-kontrollsiffra
                if !s.contains('-')
                    || is_plausible_year_range(s)
                    || RE_RANGE_UNIT_SUFFIX.is_match(&text[m.end()..])
                    || !luhn_checksum_ok(s, 7, 8)
                {
                    continue;
                }
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
        for m in RE_PG.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            let s = m.as_str();
            let digit_count = s.bytes().filter(|b| b.is_ascii_digit()).count();
            let has_kw = has_preceding_giro_keyword(text, m.start());
            if has_kw {
                // Med "Plusgiro"/"PG" framför tillåts även kortare gironummer (minst 3 siffror totalt, t.ex. 45-0)
                if digit_count < 3 {
                    continue;
                }
            } else {
                // Fristående utan nyckelord: kräv minst 6 siffror totalt (5 före strecket + kontrollsiffra),
                // giltig Luhn (Modulus 10) och att siffran inte följs av enhetsord ("år", "kr", "sid" osv.)
                if digit_count < 6
                    || RE_RANGE_UNIT_SUFFIX.is_match(&text[m.end()..])
                    || !luhn_checksum_ok(s, 6, 8)
                {
                    continue;
                }
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Konto/Giro".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── IPv4-adresser (publika) ──
    if has("IP-adress") {
        for m in RE_IPV4.find_iter(text) {
            let s = m.as_str();
            let is_private = s.starts_with("10.")
                || s.starts_with("127.")
                || s.starts_with("192.168.")
                || s.strip_prefix("172.").is_some_and(|rest| {
                    rest.split('.')
                        .next()
                        .and_then(|n| n.parse::<u32>().ok())
                        .is_some_and(|n| (16..=31).contains(&n))
                });
            if is_private || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "IP-adress".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Kreditkortsnummer (med Luhn-validering) samt tjänstekort/SITHS-kort (16–22 siffror) ──
    if has("Kortnummer") {
        for m in RE_SERVICE_CARD.find_iter(text) {
            if overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: m.as_str().to_string(),
                label: "Kortnummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
        for m in RE_CARD.find_iter(text) {
            let s = m.as_str();
            if !luhn_valid(s) || overlaps_any(&matches, m.start(), m.end()) {
                continue;
            }
            matches.push(Match {
                text: s.to_string(),
                label: "Kortnummer".to_string(),
                start: m.start(),
                end: m.end(),
                source: "regex".to_string(),
            });
        }
    }

    // ── Yrkestitlar, roller och befattningar (Steg 1A — sammansättnings- & ordlistemotor) ──
    if has("Roll/Titel") {
        for rm in roles::find_role_matches(text) {
            if overlaps_any(&matches, rm.start, rm.end) {
                continue;
            }
            matches.push(rm);
        }
    }

    matches.sort_by_key(|m| m.start);
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_regex_labels() -> Vec<String> {
        vec![
            "Personnummer",
            "E-post",
            "Telefonnummer",
            "Org.nummer",
            "Konto/Giro",
            "IP-adress",
            "Kortnummer",
            "Hemlighet/Nyckel",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn separates_personnummer_and_orgnummer() {
        let text = "Sökande 850515-1234 (samordning 701063-2391) vid bolaget 556614-5180.";
        let res = run_regex(text, &all_regex_labels());
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].label, "Personnummer");
        assert_eq!(res[0].text, "850515-1234");
        assert_eq!(res[1].label, "Personnummer");
        assert_eq!(res[1].text, "701063-2391");
        assert_eq!(res[2].label, "Org.nummer");
        assert_eq!(res[2].text, "556614-5180");
    }

    #[test]
    fn detects_api_keys_and_tokens() {
        let text = "OpenAI key: sk-abcdef1234567890abcdef123456, GitHub: ghp_123456789012345678901234567890123456";
        let res = run_regex(text, &all_regex_labels());
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].label, "Hemlighet/Nyckel");
        assert_eq!(res[1].label, "Hemlighet/Nyckel");
    }

    #[test]
    fn detects_siths_card_and_intranet_url() {
        let text = "Ditt SITHS-kort nr. 9752267235710527661 förnyas via https://intranat.rjl.se/service-och-stod-i-arbetet/bestalla/?pageId=49904";
        let res = run_regex(text, &all_regex_labels());
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].label, "Kortnummer");
        assert_eq!(res[0].text, "9752267235710527661");
        assert_eq!(res[1].label, "Hemlighet/Nyckel");
        assert!(res[1].text.starts_with("https://intranat.rjl.se/"));
    }

    #[test]
    fn regex_wins_over_overlapping_ner() {
        let rx = vec![Match {
            text: "anna.svensson@example.se".into(),
            label: "E-post".into(),
            start: 10,
            end: 34,
            source: "regex".into(),
        }];
        let ner = vec![
            Match {
                text: "anna.svensson".into(),
                label: "Person".into(),
                start: 10,
                end: 23,
                source: "kblab".into(),
            },
            Match {
                text: "Stockholm".into(),
                label: "Plats".into(),
                start: 40,
                end: 49,
                source: "kblab".into(),
            },
        ];
        let merged = merge_regex_and_ner(rx, ner);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].label, "E-post");
        assert_eq!(merged[1].label, "Plats");
    }

    #[test]
    fn ner_proper_noun_wins_over_overlapping_role_lexicon() {
        let text = "Överläkare Karin Lind samarbetar med Läkare Utan Gränser.";
        let mut labels = all_regex_labels();
        labels.push("Roll/Titel".into());
        let rx = run_regex(text, &labels);

        let karin_start = text.find("Karin Lind").unwrap();
        let lug_start = text.find("Läkare Utan Gränser").unwrap();
        let ner = vec![
            Match {
                text: "Karin Lind".into(),
                label: "Person".into(),
                start: karin_start,
                end: karin_start + "Karin Lind".len(),
                source: "kblab".into(),
            },
            Match {
                text: "Läkare Utan Gränser".into(),
                label: "Organisation".into(),
                start: lug_start,
                end: lug_start + "Läkare Utan Gränser".len(),
                source: "kblab".into(),
            },
        ];

        let merged = merge_regex_and_ner(rx, ner);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].label, "Roll/Titel");
        assert_eq!(merged[0].text, "Överläkare");
        assert_eq!(merged[1].label, "Person");
        assert_eq!(merged[1].text, "Karin Lind");
        assert_eq!(merged[2].label, "Organisation");
        assert_eq!(merged[2].text, "Läkare Utan Gränser");
    }

    #[test]
    fn konto_giro_ignores_numeric_ranges_and_years_but_catches_real_accounts() {
        let false_positives = "Erfarenhet 1-3 år, väntetid 2-6 månader, kapitel 10-8, \
                               ärende 2025-5 samt strategiperioden 2020-2024 och 2025-2029.";
        let fp_res = run_regex(false_positives, &all_regex_labels());
        assert!(
            fp_res.is_empty(),
            "Intervall som '1-3 år' eller årtal får inte matchas som Konto/Giro, fick: {:?}",
            fp_res
        );

        let valid_accounts = "Betala till Bankgiro 5050-1055, fristående 5050-1055, \
                              eller Plusgiro: 48 15-7 samt 90 19 50-6. \
                              Utbetalning till 8327-9, 123 456 789-0 eller SE45 5000 0000 0583 9825 7466.";
        let ok_res: Vec<String> = run_regex(valid_accounts, &all_regex_labels())
            .into_iter()
            .filter(|m| m.label == "Konto/Giro")
            .map(|m| m.text)
            .collect();

        assert_eq!(
            ok_res,
            vec![
                "5050-1055",
                "5050-1055",
                "48 15-7",
                "90 19 50-6",
                "8327-9, 123 456 789-0",
                "SE45 5000 0000 0583 9825 7466",
            ]
        );
    }
}
