import { useState, useEffect } from "react";
import { Upload, Shield, Download, FileText, CheckCircle2 } from "lucide-react";
import "./index.css";

interface Match {
  text: string;
  label: string;
  start: number;
  end: number;
  source: string;
}

const COLORS: Record<string, string> = {
  Personnummer: "#ef4444",
  "E-post": "#3b82f6",
  Telefonnummer: "#f59e0b",
  Person: "#8b5cf6",
  Organization: "#10b981",
  Location: "#ec4899",
};

export default function App() {
  const [fileContent, setFileContent] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [engineReady, setEngineReady] = useState(false);
  const [engineStatus, setEngineStatus] = useState("Startar AI-Motor...");
  const [matches, setMatches] = useState<Match[]>([]);
  const [entities, setEntities] = useState<string[]>(["Person", "Organization", "Location"]);

  useEffect(() => {
    const startEngine = async () => {
      try {
        const { Command } = await import('@tauri-apps/plugin-shell');
        const cmd = Command.sidecar('bin/masking_engine');
        
        cmd.stdout.on('data', line => {
          console.log(`engine: ${line}`);
          setEngineStatus(line);
          if (line.includes("redo")) {
             setEngineReady(true);
          }
        });
        
        cmd.stderr.on('data', line => {
          console.error(`engine error: ${line}`);
        });

        await cmd.spawn();
        
      } catch (e) {
        console.error("Sidecar boot error: ", e);
        // Only set this if it's not a real crash, or just say it failed
        setEngineStatus("Kritisk motor-krasch. Testa att dubbelklicka på motor-exe:n manuellt!");
      }
    };
    startEngine();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target?.result as string);
        setMatches([]);
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
      setMatches(data);
    } catch (error) {
      console.error(error);
      alert("Kunde inte ansluta till PI-motorn. Se till att den körs.");
    } finally {
      setAnalyzing(false);
    }
  };

  const renderText = () => {
    if (!matches.length) return fileContent;

    const elements = [];
    let lastIndex = 0;

    matches.forEach((m, idx) => {
      if (m.start > lastIndex) {
        elements.push(<span key={`text-${idx}`}>{fileContent.slice(lastIndex, m.start)}</span>);
      }
      elements.push(
        <mark
          key={`mark-${idx}`}
          className="highlighted-entity"
          style={{ backgroundColor: COLORS[m.label] || "#94a3b8" }}
        >
          {fileContent.slice(m.start, m.end)}
          <span className="entity-label">[{m.label}]</span>
        </mark>
      );
      lastIndex = m.end;
    });

    if (lastIndex < fileContent.length) {
      elements.push(<span key="tail">{fileContent.slice(lastIndex)}</span>);
    }

    return elements;
  };

  const handleSave = () => {
      let newText = fileContent;
      const sorted = [...matches].sort((a,b) => b.start - a.start);
      for(const m of sorted){
          newText = newText.slice(0, m.start) + `[${m.label.toUpperCase()}]` + newText.slice(m.end);
      }
      const blob = new Blob([newText], {type: "text/plain"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName.replace(".txt", "_maskerad.txt");
      a.click();
  };

  const toggleEntity = (ent: string) => {
    setEntities((prev) =>
      prev.includes(ent) ? prev.filter((e) => e !== ent) : [...prev, ent]
    );
  };

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
              {["Person", "Organization", "Location", "Date"].map((ent) => (
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
            disabled={!fileContent || analyzing || !engineReady}
            className="btn-primary"
          >
            {!engineReady ? "Startar AI-Motor..." : analyzing ? "Skannar text..." : "Granska Text"}
          </button>
          {!engineReady && <p style={{fontSize: '0.8rem', color: '#f59e0b', textAlign: 'center', marginTop: '-0.5rem'}}>{engineStatus}</p>}
        </div>

        <div className="panel relative">
          <h2 className="panel-title" style={{marginBottom: "1rem"}}>Granskning</h2>
          <div className="text-viewer">
            {fileContent ? renderText() : <p className="empty-state">Ladda upp en fil för att börja...</p>}
          </div>
          
          {matches.length > 0 && (
            <div className="stats-bar">
              <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
                Hittade <strong style={{color: "#818cf8"}}>{matches.length}</strong> känsliga uppgifter.
              </p>
              <button onClick={handleSave} className="btn-secondary">
                <Download size={16} />
                <span>Spara maskerad fil</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
