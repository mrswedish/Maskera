import React, { useState, useMemo, useCallback, useEffect } from "react";
import { Upload, Shield, Download, FileText, CheckCircle2, Table, PlusCircle, X, Settings } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";
import "./index.css";
import type { Match } from "./types";
import { buildTranslationTable } from "./translationUtils";

const REGEX_LABELS = ["Personnummer", "E-post", "Telefonnummer"] as const;
const ENTITY_LABELS = ["Person", "Organisation", "Plats", "Övrigt", "Tid", "Händelse"] as const;
const DEFAULT_ENTITIES = [...REGEX_LABELS, "Person", "Organisation", "Plats", "Övrigt"];

const COLORS: Record<string, string> = {
  Personnummer: "#ef4444",
  "E-post": "#3b82f6",
  Telefonnummer: "#f59e0b",
  Person: "#8b5cf6",
  Organisation: "#10b981",
  Plats: "#ec4899",
  Övrigt: "#94a3b8",
  Tid: "#06b6d4",
  Händelse: "#eab308",
  Manuell: "#f97316",
};

function matchKey(m: Match): string {
  return `${m.start}-${m.end}`;
}

function fixMojibake(text: string): string {
  try {
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) {
      bytes[i] = text.charCodeAt(i) & 0xFF;
    }
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return text;
  }
}

/**
 * Rust returnerar byte-offset i UTF-8. JS string.slice() vill ha
 * teckenindex (UTF-16 code units). Konverterar alla match-positioner.
 */
function fixByteOffsets(text: string, rawMatches: Match[]): Match[] {
  if (rawMatches.length === 0) return rawMatches;
  const bytes = new TextEncoder().encode(text);
  // Samla alla byte-positioner vi behöver slå upp
  const needed = new Set<number>();
  for (const m of rawMatches) { needed.add(m.start); needed.add(m.end); }
  const byteToChar = new Map<number, number>();
  let charIdx = 0;
  let byteIdx = 0;
  while (byteIdx <= bytes.length) {
    if (needed.has(byteIdx)) byteToChar.set(byteIdx, charIdx);
    if (byteIdx === bytes.length) break;
    const b = bytes[byteIdx];
    byteIdx += b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
    charIdx++;
  }
  return rawMatches.map(m => ({
    ...m,
    start: byteToChar.get(m.start) ?? m.start,
    end:   byteToChar.get(m.end)   ?? m.end,
  }));
}

async function saveTextFile(content: string, defaultName: string): Promise<void> {
  const path = await save({ defaultPath: defaultName, filters: [{ name: "Text", extensions: ["txt"] }] });
  if (path) await invoke("write_file", { path, content });
}

async function saveJsonFile(content: string, defaultName: string): Promise<void> {
  const path = await save({ defaultPath: defaultName, filters: [{ name: "JSON", extensions: ["json"] }] });
  if (path) await invoke("write_file", { path, content });
}

export default function App() {
  const [fileContent, setFileContent] = useState("");
  const [fileName, setFileName] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [matches, setMatches] = useState<Match[]>([]);
  const [ignoredKeys, setIgnoredKeys] = useState<Set<string>>(new Set());
  const [entities, setEntities] = useState<string[]>(DEFAULT_ENTITIES);
  const [showTranslationTable, setShowTranslationTable] = useState(false);
  const [engineReady, setEngineReady] = useState(false);
  const [nerProgress, setNerProgress] = useState<{ chunk: number; total: number } | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<{ file: string; downloaded: number; total: number } | null>(null);
  const [modelStatus, setModelStatus] = useState<string>("Laddar AI-modell…");
  const [showSettings, setShowSettings] = useState(false);
  const [threshold, setThreshold] = useState(0.60);

  useEffect(() => {
    // Registrera lyssnare INNAN invoke — annars kan model_ready emitteras
    // innan lyssnaren är registrerad (race condition).
    const unlisteners: (() => void)[] = [];

    Promise.all([
      listen<void>("model_ready", () => { setEngineReady(true); setDownloadProgress(null); }),
      listen<{ chunk: number; total: number }>("ner_progress", (e) => setNerProgress(e.payload)),
      listen<{ file: string; downloaded: number; total: number }>("download_progress", (e) => setDownloadProgress(e.payload)),
      listen<{ message: string }>("model_status", (e) => setModelStatus(e.payload.message)),
    ]).then((fns) => {
      unlisteners.push(...fns);
      invoke("load_model").catch((e) =>
        alert("Modell-laddning misslyckades: " + String(e))
      );
    });

    return () => { unlisteners.forEach(f => f()); };
  }, []);

  // Manuell tillägg
  const [manualWord, setManualWord] = useState("");
  const [manualLabel, setManualLabel] = useState<string>(ENTITY_LABELS[0]);

  const activeMatches = useMemo(
    () => matches.filter((m) => !ignoredKeys.has(matchKey(m))),
    [matches, ignoredKeys]
  );

  // Bygg bara om översättningstabellen från aktiva matches
  const activeTranslationTable = useMemo(
    () => buildTranslationTable(activeMatches),
    [activeMatches]
  );

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const raw = ev.target?.result as string;
      // Normalisera radbrytningar och problematiska whitespace-tecken
      // så att regex-positioner och NER-positioner alltid stämmer.
      const normalized = fixMojibake(raw)
        .replace(/\r\n/g, "\n")   // Windows → Unix
        .replace(/\r/g, "\n")     // gamla Mac
        .replace(/\u00a0/g, " ")  // non-breaking space
        .replace(/\u2028/g, "\n") // line separator
        .replace(/\u2029/g, "\n") // paragraph separator
        .replace(/\ufffd/g, " "); // replacement char (trasig encoding)
      setFileContent(normalized);
      setMatches([]);
      setIgnoredKeys(new Set());
      setShowTranslationTable(false);
    };
    reader.readAsText(file);
  };

  const handleAnalyze = async () => {
    if (!fileContent || analyzing || !engineReady) return;
    setAnalyzing(true);
    try {
      const raw = await invoke<Match[]>("analyze_text", { text: fileContent, entities, threshold });
      // Konvertera UTF-8 byte-offset → JS char-index (åäö = 2 bytes, 1 char)
      const result = fixByteOffsets(fileContent, raw);
      setMatches(result.sort((a, b) => a.start - b.start));
      setIgnoredKeys(new Set());
    } catch (err) {
      alert("Fel vid analys: " + String(err));
    } finally {
      setNerProgress(null);
      setAnalyzing(false);
    }
  };

  const toggleIgnored = useCallback((key: string) => {
    setIgnoredKeys((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }, []);

  const handleAddManual = useCallback(() => {
    const word = manualWord.trim();
    if (!word || !fileContent) return;

    const newMatches: Match[] = [];
    let searchFrom = 0;
    while (true) {
      const idx = fileContent.indexOf(word, searchFrom);
      if (idx === -1) break;
      newMatches.push({ text: word, label: manualLabel, start: idx, end: idx + word.length, source: "manuell" });
      searchFrom = idx + word.length;
    }

    if (newMatches.length === 0) {
      alert(`"${word}" hittades inte i texten.`);
      return;
    }

    setMatches((prev) => {
      // Slå ihop, sortera och deduplicera mot befintliga matches
      const combined = [...prev, ...newMatches].sort((a, b) => a.start - b.start);
      const deduped: Match[] = [];
      let lastEnd = -1;
      for (const m of combined) {
        if (m.start >= lastEnd) {
          deduped.push(m);
          lastEnd = m.end;
        }
      }
      return deduped;
    });

    setManualWord("");
  }, [manualWord, manualLabel, fileContent]);

  const renderedText = useMemo(() => {
    if (!matches.length) return fileContent;

    const elements: React.ReactNode[] = [];
    let lastIndex = 0;

    matches.forEach((m, idx) => {
      if (m.start > lastIndex) {
        elements.push(<span key={`t-${idx}`}>{fileContent.slice(lastIndex, m.start)}</span>);
      }

      const key = matchKey(m);
      const ignored = ignoredKeys.has(key);
      const masked = activeTranslationTable.get(m.text)?.maskedLabel ?? m.label;

      elements.push(
        <mark
          key={`m-${idx}`}
          className={`highlighted-entity${ignored ? " ignored" : ""}`}
          style={{ backgroundColor: ignored ? "transparent" : (COLORS[m.label] || "#94a3b8") }}
          onClick={() => toggleIgnored(key)}
          title={ignored ? "Klicka för att maskera igen" : "Klicka för att avmaskera"}
        >
          {fileContent.slice(m.start, m.end)}
          {!ignored && <span className="entity-label">[{masked}]</span>}
        </mark>
      );
      lastIndex = m.end;
    });

    if (lastIndex < fileContent.length) {
      elements.push(<span key="tail">{fileContent.slice(lastIndex)}</span>);
    }

    return elements;
  }, [fileContent, matches, ignoredKeys, activeTranslationTable, toggleIgnored]);

  const handleSave = async () => {
    const posToLabel = new Map<number, string>();
    for (const entry of activeTranslationTable.values()) {
      for (const pos of entry.positions) {
        posToLabel.set(pos.start, entry.maskedLabel);
      }
    }
    const sortedDesc = [...activeMatches].sort((a, b) => b.start - a.start);
    const segments: string[] = [];
    let cursor = fileContent.length;
    for (const m of sortedDesc) {
      segments.push(fileContent.slice(m.end, cursor));
      segments.push(`[${(posToLabel.get(m.start) ?? m.label).toUpperCase()}]`);
      cursor = m.start;
    }
    segments.push(fileContent.slice(0, cursor));
    await saveTextFile(segments.reverse().join(""), fileName.replace(".txt", "_maskerad.txt"));
  };

  const handleExportJSON = async () => {
    await saveJsonFile(
      JSON.stringify([...activeTranslationTable.values()], null, 2),
      fileName.replace(".txt", "_oversattning.json")
    );
  };

  const handleExportTXT = async () => {
    const rows = [...activeTranslationTable.values()]
      .map((e) => `${e.maskedLabel}\t${e.originalText}\t${e.count}`)
      .join("\n");
    await saveTextFile(
      "Maskerat label\tOriginalvärde\tFörekomster\n" + rows,
      fileName.replace(".txt", "_oversattning.txt")
    );
  };

  const toggleEntity = (ent: string) => {
    setEntities((prev) =>
      prev.includes(ent) ? prev.filter((e) => e !== ent) : [...prev, ent]
    );
  };

  const sortedTableEntries = useMemo(
    () =>
      [...activeTranslationTable.values()].sort(
        (a, b) => a.label.localeCompare(b.label) || a.maskedLabel.localeCompare(b.maskedLabel)
      ),
    [activeTranslationTable]
  );

  const ignoredCount = ignoredKeys.size;
  const activeCount = activeMatches.length;

  return (
    <div className="app-container">
      <header className="header">
        <Shield size={36} color="#818cf8" />
        <h1>Maskera</h1>
        <button
          onClick={() => setShowSettings(v => !v)}
          title="Inställningar"
          style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", padding: "0.25rem", color: showSettings ? "#818cf8" : "#64748b" }}
        >
          <Settings size={20} />
        </button>
      </header>

      {/* Inställningspanel */}
      {showSettings && (
        <div style={{
          background: "#1e293b", borderBottom: "1px solid #334155",
          padding: "0.75rem 1.5rem", display: "flex", alignItems: "center", gap: "1.5rem",
          fontSize: "0.82rem", color: "#94a3b8",
        }}>
          <span style={{ fontWeight: 600, color: "#cbd5e1" }}>AI-inställningar</span>
          <label style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span>Konfidenströskel</span>
            <input
              type="range" min={0.30} max={0.95} step={0.05}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              style={{ width: "120px", accentColor: "#818cf8" }}
            />
            <span style={{
              minWidth: "3rem", textAlign: "center", fontWeight: 600,
              color: threshold < 0.45 ? "#f59e0b" : threshold > 0.80 ? "#10b981" : "#818cf8",
            }}>
              {Math.round(threshold * 100)}%
            </span>
          </label>
          <span style={{ fontSize: "0.72rem", color: "#475569", maxWidth: "360px" }}>
            Lägre = fler träffar (risk för falska positiv) · Högre = färre men säkrare träffar
          </span>
        </div>
      )}

      <div className="dashboard">
        {/* Vänsterpanel */}
        <div className="panel">
          <div>
            <h2 className="panel-title">
              <FileText size={20} color="#94a3b8" /> Ladda upp text
            </h2>
            <label className="upload-area" style={{ marginTop: "1rem" }}>
              <Upload size={32} color="#94a3b8" />
              <span className="upload-text">Dra och släpp eller klicka för .txt</span>
              <input type="file" accept=".txt" onChange={handleFileUpload} />
            </label>
            {fileName && (
              <p style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "#818cf8", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <CheckCircle2 size={16} /> <span>{fileName}</span>
              </p>
            )}
          </div>

          <div>
            <h2 className="panel-title" style={{ marginBottom: "0.75rem" }}>Vad ska maskeras?</h2>
            <p style={{ fontSize: "0.72rem", color: "#64748b", marginBottom: "0.5rem", marginTop: 0 }}>Regex</p>
            <div className="checkbox-group" style={{ marginBottom: "0.75rem" }}>
              {REGEX_LABELS.map((ent) => (
                <label key={ent} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={entities.includes(ent)}
                    onChange={() => toggleEntity(ent)}
                    className="checkbox-input"
                  />
                  <span>{ent}</span>
                </label>
              ))}
            </div>
            <p style={{ fontSize: "0.72rem", color: "#64748b", marginBottom: "0.5rem", marginTop: 0 }}>AI (BERT)</p>
            <div className="checkbox-group">
              {ENTITY_LABELS.map((ent) => (
                <label key={ent} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={entities.includes(ent)}
                    onChange={() => toggleEntity(ent)}
                    className="checkbox-input"
                  />
                  <span>{ent}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Manuellt tillägg */}
          {fileContent && (
            <div>
              <h2 className="panel-title" style={{ marginBottom: "0.75rem" }}>
                <PlusCircle size={20} color="#94a3b8" /> Lägg till manuellt
              </h2>
              <div className="manual-add">
                <input
                  type="text"
                  className="manual-input"
                  placeholder="Ord eller namn..."
                  value={manualWord}
                  onChange={(e) => setManualWord(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddManual()}
                />
                <select
                  className="manual-select"
                  value={manualLabel}
                  onChange={(e) => setManualLabel(e.target.value)}
                >
                  {ENTITY_LABELS.map((l) => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
                <button
                  className="btn-add"
                  onClick={handleAddManual}
                  disabled={!manualWord.trim()}
                >
                  Lägg till
                </button>
              </div>
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={!fileContent || analyzing || !engineReady}
            className="btn-primary"
          >
            {analyzing ? "Skannar text..." : "Granska Text"}
          </button>

          {/* Progress-bar visas under analys */}
          {analyzing && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <div style={{
                width: "100%", height: "6px", borderRadius: "3px",
                backgroundColor: "#1e293b", overflow: "hidden",
              }}>
                <div style={{
                  height: "100%", borderRadius: "3px",
                  backgroundColor: "#818cf8",
                  width: nerProgress
                    ? `${Math.round((nerProgress.chunk / nerProgress.total) * 100)}%`
                    : "5%",
                  transition: "width 0.3s ease",
                }} />
              </div>
              <p style={{ fontSize: "0.72rem", color: "#64748b", margin: 0, textAlign: "center" }}>
                {nerProgress
                  ? `Block ${nerProgress.chunk} / ${nerProgress.total} (${Math.round((nerProgress.chunk / nerProgress.total) * 100)}%)`
                  : "Förbereder analys..."}
              </p>
            </div>
          )}

          {!engineReady && !analyzing && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              {downloadProgress ? (
                <>
                  <div style={{ width: "100%", height: "6px", borderRadius: "3px", backgroundColor: "#1e293b", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", borderRadius: "3px", backgroundColor: "#06b6d4",
                      width: downloadProgress.total > 0
                        ? `${Math.round((downloadProgress.downloaded / downloadProgress.total) * 100)}%`
                        : "100%",
                      transition: "width 0.2s ease",
                    }} />
                  </div>
                  <p style={{ fontSize: "0.72rem", color: "#64748b", margin: 0, textAlign: "center" }}>
                    {`Laddar ner ${downloadProgress.file} — ${(downloadProgress.downloaded / 1_048_576).toFixed(1)} MB`}
                    {downloadProgress.total > 0 && ` / ${(downloadProgress.total / 1_048_576).toFixed(0)} MB`}
                  </p>
                </>
              ) : (
                <p style={{ fontSize: "0.75rem", color: "#64748b", margin: 0, textAlign: "center" }}>
                  {modelStatus}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Textvisare */}
        <div className="panel panel-fill">
          <h2 className="panel-title" style={{ marginBottom: "1rem" }}>Granskning</h2>
          {matches.length > 0 && (
            <p className="review-hint">
              Klicka på ett markerat ord för att avmaskera det.
            </p>
          )}
          <div className="text-viewer">
            {fileContent
              ? renderedText
              : <p className="empty-state">Ladda upp en fil för att börja...</p>}
          </div>

          {matches.length > 0 && (
            <div className="stats-bar">
              <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
                <strong style={{ color: "#818cf8" }}>{activeCount}</strong> aktiva
                {ignoredCount > 0 && (
                  <span style={{ color: "#64748b" }}> · {ignoredCount} ignorerade</span>
                )}
              </p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button onClick={() => setShowTranslationTable((v) => !v)} className="btn-secondary">
                  <Table size={16} />
                  <span>{showTranslationTable ? "Dölj" : "Visa"} översättningstabell</span>
                </button>
                <button onClick={handleSave} className="btn-secondary" disabled={activeCount === 0}>
                  <Download size={16} />
                  <span>Spara maskerad fil</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Översättningstabell */}
        {showTranslationTable && activeTranslationTable.size > 0 && (
          <div className="panel translation-panel">
            <h2 className="panel-title">
              <Table size={20} color="#94a3b8" /> Översättningstabell
            </h2>
            <table className="translation-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Maskerat label</th>
                  <th>Originalvärde</th>
                  <th>Förekomster</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {sortedTableEntries.map((entry) => (
                  <tr key={entry.originalText}>
                    <td>
                      <span className="entity-dot" style={{ backgroundColor: COLORS[entry.label] || "#94a3b8" }} />
                    </td>
                    <td><code>{entry.maskedLabel}</code></td>
                    <td>{entry.originalText}</td>
                    <td style={{ textAlign: "center" }}>{entry.count}</td>
                    <td>
                      <button
                        className="btn-ignore-all"
                        title="Ignorera alla förekomster"
                        onClick={() => {
                          const keys = matches
                            .filter((m) => m.text === entry.originalText)
                            .map(matchKey);
                          setIgnoredKeys((prev) => {
                            const next = new Set(prev);
                            keys.forEach((k) => next.add(k));
                            return next;
                          });
                        }}
                      >
                        <X size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="translation-export">
              <button onClick={handleExportJSON} className="btn-export">
                <Download size={14} /> Exportera JSON
              </button>
              <button onClick={handleExportTXT} className="btn-export">
                <Download size={14} /> Exportera TXT (tab-separerad)
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
