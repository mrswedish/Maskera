import React, { useState, useMemo } from "react";
import { Upload, Shield, Download, FileText, CheckCircle2, Table } from "lucide-react";
import "./index.css";
import type { Match, TranslationTable } from "./types";
import { buildTranslationTable } from "./translationUtils";

const COLORS: Record<string, string> = {
  Personnummer: "#ef4444",
  "E-post": "#3b82f6",
  Telefonnummer: "#f59e0b",
  Person: "#8b5cf6",
  Organisation: "#10b981",
  Plats: "#ec4899",
  Övrigt: "#94a3b8",
};

export default function App() {
  const [fileContent, setFileContent] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [matches, setMatches] = useState<Match[]>([]);
  const [entities, setEntities] = useState<string[]>(["Person", "Organisation", "Plats", "Övrigt"]);
  const [translationTable, setTranslationTable] = useState<TranslationTable>(new Map());
  const [showTranslationTable, setShowTranslationTable] = useState(false);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target?.result as string);
        setMatches([]);
        setTranslationTable(new Map());
        setShowTranslationTable(false);
      };
      reader.readAsText(file);
    }
  };

  const handleAnalyze = async () => {
    if (!fileContent) return;
    setAnalyzing(true);
    try {
      const res = await fetch("http://127.0.0.1:8594/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: fileContent, entities }),
      });
      const data = await res.json();
      const sorted = (data as Match[]).sort((a, b) => a.start - b.start);
      setMatches(sorted);
      setTranslationTable(buildTranslationTable(sorted));
    } catch (error) {
      console.error(error);
      alert("Kunde inte ansluta till PI-motorn. Se till att den körs.");
    } finally {
      setAnalyzing(false);
    }
  };

  const renderedText = useMemo(() => {
    if (!matches.length) return fileContent;

    const elements: React.ReactNode[] = [];
    let lastIndex = 0;

    matches.forEach((m, idx) => {
      if (m.start > lastIndex) {
        elements.push(<span key={`text-${idx}`}>{fileContent.slice(lastIndex, m.start)}</span>);
      }
      const masked = translationTable.get(m.text)?.maskedLabel ?? m.label;
      elements.push(
        <mark
          key={`mark-${idx}`}
          className="highlighted-entity"
          style={{ backgroundColor: COLORS[m.label] || "#94a3b8" }}
        >
          {fileContent.slice(m.start, m.end)}
          <span className="entity-label">[{masked}]</span>
        </mark>
      );
      lastIndex = m.end;
    });

    if (lastIndex < fileContent.length) {
      elements.push(<span key="tail">{fileContent.slice(lastIndex)}</span>);
    }

    return elements;
  }, [fileContent, matches, translationTable]);

  const handleSave = () => {
    const posToLabel = new Map<number, string>();
    for (const entry of translationTable.values()) {
      for (const pos of entry.positions) {
        posToLabel.set(pos.start, entry.maskedLabel);
      }
    }

    const sortedDesc = [...matches].sort((a, b) => b.start - a.start);
    const segments: string[] = [];
    let cursor = fileContent.length;
    for (const m of sortedDesc) {
      segments.push(fileContent.slice(m.end, cursor));
      const label = posToLabel.get(m.start) ?? m.label;
      segments.push(`[${label.toUpperCase()}]`);
      cursor = m.start;
    }
    segments.push(fileContent.slice(0, cursor));
    const newText = segments.reverse().join('');
    const blob = new Blob([newText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName.replace(".txt", "_maskerad.txt");
    a.click();
  };

  const handleExportJSON = () => {
    const entries = [...translationTable.values()];
    const blob = new Blob([JSON.stringify(entries, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName.replace(".txt", "_oversattning.json");
    a.click();
  };

  const handleExportTXT = () => {
    const header = "Maskerat label\tOriginalvärde\tFörekomster\n";
    const rows = [...translationTable.values()]
      .map(e => `${e.maskedLabel}\t${e.originalText}\t${e.count}`)
      .join('\n');
    const blob = new Blob([header + rows], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName.replace(".txt", "_oversattning.txt");
    a.click();
  };

  const toggleEntity = (ent: string) => {
    setEntities((prev) =>
      prev.includes(ent) ? prev.filter((e) => e !== ent) : [...prev, ent]
    );
  };

  const sortedTableEntries = useMemo(() => {
    return [...translationTable.values()].sort((a, b) =>
      a.label.localeCompare(b.label) || a.maskedLabel.localeCompare(b.maskedLabel)
    );
  }, [translationTable]);

  return (
    <div className="app-container">
      <header className="header">
        <Shield size={36} color="#818cf8" />
        <h1>Maskera</h1>
      </header>

      <div className="dashboard">
        <div className="panel">
          <div>
            <h2 className="panel-title">
              <FileText size={20} color="#94a3b8" /> Ladda upp text
            </h2>
            <label className="upload-area" style={{marginTop: "1rem"}}>
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
            <h2 className="panel-title" style={{marginBottom: "0.5rem"}}>Vad ska maskeras?</h2>
            <p style={{ fontSize: "0.75rem", color: "#94a3b8", marginBottom: "1rem", marginTop: 0 }}>OBS: Personnummer, e-post och telefonnummer söks alltid (Regex).</p>
            <div className="checkbox-group">
              {["Person", "Organisation", "Plats", "Övrigt"].map((ent) => (
                <label key={ent} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={entities.includes(ent)}
                    onChange={() => toggleEntity(ent)}
                    className="checkbox-input"
                  />
                  <span>{ent} (AI)</span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!fileContent || analyzing}
            className="btn-primary"
          >
            {analyzing ? "Skannar text..." : "Granska Text"}
          </button>
        </div>

        <div className="panel relative">
          <h2 className="panel-title" style={{marginBottom: "1rem"}}>Granskning</h2>
          <div className="text-viewer">
            {fileContent ? renderedText : <p className="empty-state">Ladda upp en fil för att börja...</p>}
          </div>

          {matches.length > 0 && (
            <div className="stats-bar">
              <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
                Hittade <strong style={{color: "#818cf8"}}>{matches.length}</strong> känsliga uppgifter.
              </p>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button onClick={() => setShowTranslationTable(v => !v)} className="btn-secondary">
                  <Table size={16} />
                  <span>{showTranslationTable ? "Dölj" : "Visa"} översättningstabell</span>
                </button>
                <button onClick={handleSave} className="btn-secondary">
                  <Download size={16} />
                  <span>Spara maskerad fil</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {showTranslationTable && translationTable.size > 0 && (
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
                </tr>
              </thead>
              <tbody>
                {sortedTableEntries.map((entry) => (
                  <tr key={entry.originalText}>
                    <td>
                      <span
                        className="entity-dot"
                        style={{ backgroundColor: COLORS[entry.label] || "#94a3b8" }}
                      />
                    </td>
                    <td><code>{entry.maskedLabel}</code></td>
                    <td>{entry.originalText}</td>
                    <td style={{ textAlign: "center" }}>{entry.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="translation-export">
              <button onClick={handleExportJSON} className="btn-export">
                <Download size={14} />
                Exportera JSON
              </button>
              <button onClick={handleExportTXT} className="btn-export">
                <Download size={14} />
                Exportera TXT (tab-separerad)
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
