import { pipeline, env } from "@huggingface/transformers";
import type { Match } from "./types";

env.allowLocalModels = false;

const MODEL_ID = "psvensk/bert-base-swedish-cased-ner-onnx";

const NER_TO_UI: Record<string, string> = {
  PER: "Person",
  ORG: "Organisation",
  LOC: "Plats",
  MISC: "Övrigt",
};

let ner: Awaited<ReturnType<typeof pipeline>> | null = null;

// ── Regex ────────────────────────────────────────────────────────────────────

function findRegexMatches(text: string): Match[] {
  const matches: Match[] = [];

  const pnr = /\b(?:\d{2})?\d{6}[-+]?\d{4}\b/g;
  for (const m of text.matchAll(pnr)) {
    matches.push({ text: m[0], label: "Personnummer", start: m.index!, end: m.index! + m[0].length, source: "regex" });
  }

  const email = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g;
  for (const m of text.matchAll(email)) {
    matches.push({ text: m[0], label: "E-post", start: m.index!, end: m.index! + m[0].length, source: "regex" });
  }

  const phone = /(?:(?:0|\+46|0046)\s?(?:[1-9]\d{1,2}\s?(?:[- ]?\d{2,3}){1,3}|\d{2,3}\s?(?:[- ]?\d{2,3}){1,3}))/g;
  for (const m of text.matchAll(phone)) {
    const trimmed = m[0].trimEnd();
    matches.push({ text: trimmed, label: "Telefonnummer", start: m.index!, end: m.index! + trimmed.length, source: "regex" });
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

// ── NER ───────────────────────────────────────────────────────────────────────

async function runNer(text: string, requestedLabels: string[]): Promise<Match[]> {
  if (!ner || requestedLabels.length === 0) return [];

  const requestedNerTags = new Set(
    requestedLabels.flatMap((l) => {
      const entry = Object.entries(NER_TO_UI).find(([, v]) => v === l);
      return entry ? [entry[0]] : [];
    })
  );

  const nerMatches: Match[] = [];

  for (const { chunk, start: chunkStart } of chunkText(text)) {
    if (!chunk.trim()) continue;
    try {
      const predictions = await (ner as any)(chunk);
      for (const p of predictions) {
        const group: string = p.entity_group ?? p.entity ?? "";
        const uiLabel = NER_TO_UI[group];
        if (uiLabel && requestedNerTags.has(group)) {
          const word = chunk.slice(p.start, p.end);
          nerMatches.push({
            text: word,
            label: uiLabel,
            start: chunkStart + p.start,
            end: chunkStart + p.end,
            source: "kblab",
          });
        }
      }
    } catch (e) {
      console.error("NER chunk error:", e);
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
        // @ts-expect-error aggregation_strategy is valid at runtime but missing from types
        aggregation_strategy: "simple",
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
      const regexMatches = findRegexMatches(text);
      const nerMatches = await runNer(text, entities);
      const result = deduplicateMatches([...regexMatches, ...nerMatches]);
      self.postMessage({ type: "results", payload: result });
    } catch (err) {
      self.postMessage({ type: "error", payload: String(err) });
    }
  }
};
