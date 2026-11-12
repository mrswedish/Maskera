import type { Match, TranslationEntry, TranslationTable } from "./types";

// NER- och roll-entiteter får bokstavssuffix: "Person A", "Roll A"…
// Regex- och manuella entiteter får numeriskt suffix: "Personnummer 1", "E-post 2"…
function maskedLabel(label: string, source: string, counter: number): string {
  const baseLabel = label === "Roll/Titel" ? "Roll" : label;
  if (source === "regex" || source === "manuell") {
    return `${baseLabel} ${counter}`;
  }
  // Bokstav A–Z, sedan AA, AB… (upp till 702 unika värden)
  if (counter <= 26) {
    return `${baseLabel} ${String.fromCharCode(64 + counter)}`;
  }
  const first = String.fromCharCode(64 + Math.floor((counter - 1) / 26));
  const second = String.fromCharCode(64 + ((counter - 1) % 26) + 1);
  return `${baseLabel} ${first}${second}`;
}

export function buildTranslationTable(matches: Match[]): TranslationTable {
  const table: TranslationTable = new Map<string, TranslationEntry>();
  const counters = new Map<string, number>();

  for (const m of matches) {
    // För yrkesroller slås meningsinledande versal ("Verksamhetschefen") och gemen
    // ("verksamhetschefen") ihop till samma [ROLL A] så AI:n ser att det är samma roll.
    const lookupKey =
      m.label === "Roll/Titel" ? `role:${m.text.toLowerCase()}` : m.text;

    if (table.has(lookupKey)) {
      const entry = table.get(lookupKey)!;
      entry.positions.push({ start: m.start, end: m.end });
      entry.count += 1;
      if (!entry.variants) {
        entry.variants = [entry.originalText];
      }
      if (!entry.variants.includes(m.text)) {
        entry.variants.push(m.text);
      }
      // Föredra gemen grundform vid avmaskering om ordet inte är en förkortning (t.ex. IT-chef / VD)
      if (m.label === "Roll/Titel" && m.text === m.text.toLowerCase()) {
        entry.originalText = m.text;
      }
    } else {
      const counterKey = `${m.label}|${m.source}`;
      const next = (counters.get(counterKey) ?? 0) + 1;
      counters.set(counterKey, next);

      const entry: TranslationEntry = {
        maskedLabel: maskedLabel(m.label, m.source, next),
        label: m.label,
        originalText: m.text,
        variants: [m.text],
        positions: [{ start: m.start, end: m.end }],
        count: 1,
      };
      table.set(lookupKey, entry);
    }
  }

  return table;
}

export function buildMaskedText(
  sourceText: string,
  activeMatches: Match[],
  table: TranslationTable
): string {
  if (!sourceText || activeMatches.length === 0) return sourceText;

  const posToLabel = new Map<number, string>();
  for (const entry of table.values()) {
    for (const pos of entry.positions) {
      posToLabel.set(pos.start, entry.maskedLabel);
    }
  }

  const sortedDesc = [...activeMatches].sort((a, b) => b.start - a.start);
  const segments: string[] = [];
  let cursor = sourceText.length;

  for (const m of sortedDesc) {
    segments.push(sourceText.slice(m.end, cursor));
    segments.push(`[${(posToLabel.get(m.start) ?? m.label).toUpperCase()}]`);
    cursor = m.start;
  }
  segments.push(sourceText.slice(0, cursor));
  return segments.reverse().join("");
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Återställer maskerade platshållare (t.ex. `[PERSON A]`, `[Person A]` eller fristående `PERSON A`)
 * i ett AI-genererat svar tillbaka till originalvärden lokalt via översättningstabellen.
 */
export function unmaskText(
  aiText: string,
  table: TranslationTable
): { text: string; replacements: number } {
  if (!aiText || table.size === 0) {
    return { text: aiText, replacements: 0 };
  }

  // Sortera längsta platshållare först så att t.ex. "PERSON AA" eller "E-POST 10"
  // matchas före "PERSON A" respektive "E-POST 1".
  const entries = [...table.values()].sort(
    (a, b) => b.maskedLabel.length - a.maskedLabel.length
  );

  let output = aiText;
  let replacements = 0;

  for (const entry of entries) {
    const escaped = escapeRegExp(entry.maskedLabel);
    const escapedUpper = escapeRegExp(entry.maskedLabel.toUpperCase());

    // 1. Hakparenteser (skiftlägesokänsligt): [PERSON A], [Person A], [person a]
    const bracketRe = new RegExp(`\\[\\s*${escaped}\\s*\\]`, "giu");
    output = output.replace(bracketRe, () => {
      replacements += 1;
      return entry.originalText;
    });

    // 2. Fristående platshållare om AI:n tagit bort hakparenteserna (exakt eller VERSALER)
    const bareRe = new RegExp(
      `(?<![\\p{L}\\p{N}])(?:${escapedUpper}|${escaped})(?![\\p{L}\\p{N}])`,
      "gu"
    );
    output = output.replace(bareRe, () => {
      replacements += 1;
      return entry.originalText;
    });
  }

  return { text: output, replacements };
}
