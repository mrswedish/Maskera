import type { Match, TranslationEntry, TranslationTable } from "./types";

// NER-entiteter (kblab) får bokstavssuffix: "Person A", "Person B"…
// Regex-entiteter får numeriskt suffix: "Personnummer 1", "E-post 2"…
function maskedLabel(label: string, source: string, counter: number): string {
  if (source === "regex" || source === "manuell") {
    return `${label} ${counter}`;
  }
  // Bokstav A–Z, sedan AA, AB… (upp till 702 unika värden)
  if (counter <= 26) {
    return `${label} ${String.fromCharCode(64 + counter)}`;
  }
  const first = String.fromCharCode(64 + Math.floor((counter - 1) / 26));
  const second = String.fromCharCode(64 + ((counter - 1) % 26) + 1);
  return `${label} ${first}${second}`;
}

export function buildTranslationTable(matches: Match[]): TranslationTable {
  const table: TranslationTable = new Map<string, TranslationEntry>();
  // Räknare per (label + source)-kombination
  const counters = new Map<string, number>();

  for (const m of matches) {
    if (table.has(m.text)) {
      const entry = table.get(m.text)!;
      entry.positions.push({ start: m.start, end: m.end });
      entry.count += 1;
    } else {
      const counterKey = `${m.label}|${m.source}`;
      const next = (counters.get(counterKey) ?? 0) + 1;
      counters.set(counterKey, next);

      const entry: TranslationEntry = {
        maskedLabel: maskedLabel(m.label, m.source, next),
        label: m.label,
        originalText: m.text,
        positions: [{ start: m.start, end: m.end }],
        count: 1,
      };
      table.set(m.text, entry);
    }
  }

  return table;
}
