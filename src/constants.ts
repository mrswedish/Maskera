import type { InfoSectionFinding, Match, ParagraphFinding } from "./types";

export const REGEX_LABELS = [
  "Personnummer",
  "E-post",
  "Telefonnummer",
  "Org.nummer",
  "Konto/Giro",
  "IP-adress",
  "Kortnummer",
  "Hemlighet/Nyckel",
] as const;

export const ENTITY_LABELS = [
  "Person",
  "Roll/Titel",
  "Organisation",
  "Plats",
  "Övrigt",
  "Tid",
  "Händelse",
] as const;

export const ALL_CATEGORIES = [...REGEX_LABELS, ...ENTITY_LABELS];

export const PRESETS = {
  ai_safe: {
    name: "AI-säker",
    entities: [
      "Personnummer",
      "E-post",
      "Telefonnummer",
      "Org.nummer",
      "Konto/Giro",
      "Kortnummer",
      "Hemlighet/Nyckel",
      "Person",
      "Organisation",
      "Plats",
      "Övrigt",
    ],
  },
  id_only: {
    name: "Endast ID & kontakt",
    entities: [
      "Personnummer",
      "E-post",
      "Telefonnummer",
      "Org.nummer",
      "Konto/Giro",
      "Kortnummer",
      "Hemlighet/Nyckel",
    ],
  },
  strict: {
    name: "Strikt (Allt)",
    entities: [...ALL_CATEGORIES],
  },
} as const;

export type PresetKey = keyof typeof PRESETS;

// WCAG AA-kalibrerade kategorifärger (>= 5.0:1 kontrast mot vit text #ffffff)
export const COLORS: Record<string, string> = {
  Personnummer: "#b91c1c",
  "E-post": "#1d4ed8",
  Telefonnummer: "#b45309",
  "Org.nummer": "#7e22ce",
  "Konto/Giro": "#be185d",
  "IP-adress": "#0369a1",
  Kortnummer: "#991b1b",
  "Hemlighet/Nyckel": "#9f1239",
  Person: "#5b21b6",
  "Roll/Titel": "#3730a3",
  Organisation: "#047857",
  Plats: "#9d174d",
  Övrigt: "#475569",
  Tid: "#0e7490",
  Händelse: "#854d0e",
  Manuell: "#c2410c",
};

export type ColorSchemeSetting = "light dark" | "light" | "dark";

export function fixMojibake(text: string): string {
  try {
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) {
      bytes[i] = text.charCodeAt(i) & 0xff;
    }
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return text;
  }
}

function buildByteToCharMap(
  text: string,
  needed: Set<number>
): Map<number, number> {
  const byteToChar = new Map<number, number>();
  if (needed.size === 0) return byteToChar;

  let u16 = 0;
  let byteIdx = 0;
  const len = text.length;

  while (u16 < len) {
    if (needed.has(byteIdx)) {
      byteToChar.set(byteIdx, u16);
      if (byteToChar.size === needed.size) break;
    }
    const code = text.charCodeAt(u16);
    if (code < 0x80) {
      byteIdx += 1;
      u16 += 1;
    } else if (code < 0x800) {
      byteIdx += 2;
      u16 += 1;
    } else if (code >= 0xd800 && code <= 0xdbff && u16 + 1 < len) {
      byteIdx += 4;
      u16 += 2;
    } else {
      byteIdx += 3;
      u16 += 1;
    }
  }
  if (needed.has(byteIdx)) {
    byteToChar.set(byteIdx, u16);
  }
  return byteToChar;
}

export function fixByteOffsets(text: string, rawMatches: Match[]): Match[] {
  if (rawMatches.length === 0) return rawMatches;

  const needed = new Set<number>();
  for (const m of rawMatches) {
    needed.add(m.start);
    needed.add(m.end);
  }

  const byteToChar = buildByteToCharMap(text, needed);

  return rawMatches.map((m) => ({
    ...m,
    start: byteToChar.get(m.start) ?? m.start,
    end: byteToChar.get(m.end) ?? m.end,
  }));
}

export function fixFindingOffsets(
  text: string,
  rawFindings: ParagraphFinding[]
): ParagraphFinding[] {
  if (rawFindings.length === 0) return rawFindings;

  const needed = new Set<number>();
  for (const f of rawFindings) {
    needed.add(f.start);
    needed.add(f.end);
  }

  const byteToChar = buildByteToCharMap(text, needed);

  return rawFindings.map((f) => ({
    ...f,
    start: Math.min(text.length, byteToChar.get(f.start) ?? f.start),
    end: Math.min(text.length, byteToChar.get(f.end) ?? f.end),
  }));
}

export function fixInfoSectionOffsets(
  text: string,
  rawFindings: InfoSectionFinding[]
): InfoSectionFinding[] {
  if (rawFindings.length === 0) return rawFindings;

  const needed = new Set<number>();
  for (const f of rawFindings) {
    needed.add(f.start);
    needed.add(f.end);
  }

  const byteToChar = buildByteToCharMap(text, needed);

  return rawFindings.map((f) => ({
    ...f,
    start: Math.min(text.length, byteToChar.get(f.start) ?? f.start),
    end: Math.min(text.length, byteToChar.get(f.end) ?? f.end),
  }));
}

export function normalizeInputText(raw: string): string {
  return fixMojibake(raw)
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\u00a0/g, " ")
    .replace(/\u2028/g, "\n")
    .replace(/\u2029/g, "\n")
    .replace(/\ufffd/g, " ");
}
