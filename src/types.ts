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

export interface ParagraphFinding {
  paragraph_index: number;
  preview: string;
  start: number;
  end: number;
  category: string;
  axis: "integritet" | "affär" | "säkerhet" | string;
  severity: "high" | "medium" | string;
  confidence: number;
  reason: string;
}

export type SafetyVerdict =
  | "safe"
  | "mask_required"
  | "caution_after_mask"
  | "do_not_share";

export interface ClassificationReport {
  verdict_before_mask: SafetyVerdict;
  verdict_after_mask: SafetyVerdict;
  summary_before: string;
  summary_after: string;
  active_pii_count: number;
  unmasked_pii_count: number;
  secret_count: number;
  semantic_available: boolean;
  paragraphs_scanned: number;
  scan_duration_ms: number;
  findings: ParagraphFinding[];
}

export interface ImportedDocument {
  file_name: string;
  format: "docx" | "xlsx" | "pdf" | "odt" | "text" | string;
  text: string;
  summary: string;
  can_export_docx: boolean;
  can_export_xlsx: boolean;
}

export interface ReplacementPair {
  original: string;
  replacement: string;
}

export interface ModelsStatus {
  nerInstalled: boolean;
  gemmaInstalled: boolean;
  nerReady: boolean;
  gemmaReady: boolean;
}

export interface SetupProgress {
  phase: "downloading" | "copying" | "loading" | "ready";
  step: number;
  totalSteps: number;
  nerDone: boolean;
  gemmaDone: boolean;
  label: string;
  downloadedBytes: number;
  totalBytes: number;
}
