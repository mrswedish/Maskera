export interface Match {
  text: string;
  label: string;
  start: number;
  end: number;
  source: string;
}

export interface TranslationEntry {
  maskedLabel: string;
  label: string;
  originalText: string;
  positions: Array<{ start: number; end: number }>;
  count: number;
}

export type TranslationTable = Map<string, TranslationEntry>;
