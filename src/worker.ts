import { pipeline, env } from "@huggingface/transformers";
import type { Match } from "./types";

env.allowLocalModels = false;

const MODEL_ID = "psvensk/bert-base-swedish-cased-ner-onnx";

// Hanterar både aggregerade (PER, ORG) och råa B-/I-prefixade taggar (B-PRS, I-LOC)
const NER_TO_UI: Record<string, string> = {
  PER: "Person",  PRS: "Person",
  ORG: "Organisation",
  LOC: "Plats",
  MISC: "Övrigt",
  TME: "Tid",
  EVN: "Händelse",
  MSR: "Övrigt",
};

function normalizeTag(raw: string): string {
  // "B-PRS" → "PRS", "I-ORG" → "ORG", "PER" → "PER"
  return raw.replace(/^[BI]-/, "");
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ner: any = null;

// ── Regex ────────────────────────────────────────────────────────────────────

function findRegexMatches(text: string, requestedLabels: string[]): Match[] {
  const matches: Match[] = [];

  if (requestedLabels.includes("Personnummer")) {
    const pnr = /\b(?:\d{2})?\d{6}[-+]?\d{4}\b/g;
    for (const m of text.matchAll(pnr)) {
      matches.push({ text: m[0], label: "Personnummer", start: m.index!, end: m.index! + m[0].length, source: "regex" });
    }
  }

  if (requestedLabels.includes("E-post")) {
    const email = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g;
    for (const m of text.matchAll(email)) {
      matches.push({ text: m[0], label: "E-post", start: m.index!, end: m.index! + m[0].length, source: "regex" });
    }
  }

  if (requestedLabels.includes("Telefonnummer")) {
    const phone = /(?:(?:0|\+46|0046)\s?(?:[1-9]\d{1,2}\s?(?:[- ]?\d{2,3}){1,3}|\d{2,3}\s?(?:[- ]?\d{2,3}){1,3}))/g;
    for (const m of text.matchAll(phone)) {
      const trimmed = m[0].trimEnd();
      matches.push({ text: trimmed, label: "Telefonnummer", start: m.index!, end: m.index! + trimmed.length, source: "regex" });
    }
  }

  return matches;
}

// ── Chunking ─────────────────────────────────────────────────────────────────

function chunkText(text: string): Array<{ chunk: string; start: number }> {
  const MAX = 1500;
  const chunks: Array<{ chunk: string; start: number }> = [];
  let pos = 0;

  while (pos < text.length) {
    let end = pos + MAX;
    if (end >= text.length) {
      end = text.length;
    } else {
      const sentenceBreak = text.lastIndexOf(". ", end);
      const wordBreak = text.lastIndexOf(" ", end);
      const cut = sentenceBreak > pos ? sentenceBreak + 1 : wordBreak > pos ? wordBreak + 1 : end;
      end = cut;
    }
    chunks.push({ chunk: text.slice(pos, end), start: pos });
    pos = end;
  }

  return chunks;
}

// ── Deduplicate ───────────────────────────────────────────────────────────────

function deduplicateMatches(matches: Match[]): Match[] {
  const sorted = [...matches].sort((a, b) => a.start - b.start);
  const result: Match[] = [];
  let lastEnd = -1;
  for (const m of sorted) {
    if (m.start >= lastEnd) {
      result.push(m);
      lastEnd = m.end;
    }
  }
  return result;
}

// ── Manuell aggregering (Transformers.js saknar aggregation_strategy) ────────

interface RawPrediction { entity: string; score: number; index: number; word: string; }
interface AggregatedEntity { rawTag: string; text: string; start: number; end: number; }

function aggregatePredictions(preds: RawPrediction[], chunk: string): AggregatedEntity[] {
  const sorted = [...preds].sort((a, b) => a.index - b.index);
  const entities: AggregatedEntity[] = [];
  let current: { rawTag: string; start: number; end: number } | null = null;
  let searchPos = 0;

  const flush = () => {
    if (current) {
      entities.push({ rawTag: current.rawTag, text: chunk.slice(current.start, current.end), start: current.start, end: current.end });
      current = null;
    }
  };

  for (const p of sorted) {
    const tag = normalizeTag(p.entity);
    const word = p.word as string;

    if (!NER_TO_UI[tag]) {
      flush();
      if (!word.startsWith("##")) {
        // Advance searchPos only if the word is found close by (within 200 chars).
        // Common words like "och", "på" can appear anywhere — don't let them
        // shoot searchPos past a real entity that comes next in token order.
        const idx = chunk.indexOf(word, searchPos);
        if (idx !== -1 && idx - searchPos <= 200) searchPos = idx + word.length;
      }
      continue;
    }

    if (word.startsWith("##")) {
      const suffix = word.slice(2);
      if (current && chunk.slice(current.end).startsWith(suffix)) {
        current.end += suffix.length;
      }
    } else {
      const idx = chunk.indexOf(word, searchPos);
      if (idx === -1) continue;

      const gap = current ? chunk.slice(current.end, idx) : "";
      if (current && tag === current.rawTag && /^\s*$/.test(gap)) {
        current.end = idx + word.length;
      } else {
        flush();
        current = { rawTag: tag, start: idx, end: idx + word.length };
      }
      searchPos = idx + word.length;
    }
  }
  flush();
  return entities;
}

// ── NER ───────────────────────────────────────────────────────────────────────

async function runNer(text: string, requestedLabels: string[]): Promise<Match[]> {
  if (!ner || requestedLabels.length === 0) return [];

  const requestedNerTags = new Set(
    Object.entries(NER_TO_UI)
      .filter(([, v]) => requestedLabels.includes(v))
      .map(([k]) => k)
  );

  if (requestedNerTags.size === 0) return [];

  const nerMatches: Match[] = [];
  const chunks = chunkText(text);

  for (let i = 0; i < chunks.length; i++) {
    const { chunk, start: chunkStart } = chunks[i];
    if (!chunk.trim()) continue;

    // Yield to runtime every 10 chunks so GC can reclaim WASM memory
    if (i > 0 && i % 10 === 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }

    try {
      const predictions: RawPrediction[] = await (ner as any)(chunk);
      const grouped = aggregatePredictions(predictions, chunk);
      for (const e of grouped) {
        if (!requestedNerTags.has(e.rawTag)) continue;
        nerMatches.push({
          text: e.text,
          label: NER_TO_UI[e.rawTag],
          start: chunkStart + e.start,
          end: chunkStart + e.end,
          source: "kblab",
        });
      }
    } catch (e) {
      console.error(`NER chunk ${i} error:`, e);
    }
  }

  return nerMatches;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === "load") {
    try {
      ner = await pipeline("token-classification", MODEL_ID, {
        progress_callback: (p: any) => {
          if (p.status === "progress" && p.total) {
            self.postMessage({ type: "progress", payload: Math.round((p.loaded / p.total) * 100) });
          }
        },
      });
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", payload: String(err) });
    }
  }

  if (type === "analyze") {
    const { text, entities } = payload as { text: string; entities: string[] };
    try {
      const regexMatches = findRegexMatches(text, entities);
      console.log("[worker] regex:", regexMatches.length, "träffar");
      const nerMatches = await runNer(text, entities);
      console.log("[worker] ner:", nerMatches.length, "träffar");
      const result = deduplicateMatches([...regexMatches, ...nerMatches]);
      console.log("[worker] totalt:", result.length, "träffar");
      self.postMessage({ type: "results", payload: result });
    } catch (err) {
      self.postMessage({ type: "error", payload: String(err) });
    }
  }
};
