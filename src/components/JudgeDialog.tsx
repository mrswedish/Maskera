import type React from "react";
import {
  AlertTriangle,
  Download,
  FolderOpen,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { JudgeDownloadProgress, JudgeModelStatus } from "../types";

export interface JudgeDialogProps {
  dialogRef: React.RefObject<HTMLDialogElement | null>;
  mode: "activate" | "manage";
  status: JudgeModelStatus;
  busy: boolean;
  progress: JudgeDownloadProgress | null;
  error: string | null;
  onBackdropClick: (e: React.MouseEvent<HTMLDialogElement>) => void;
  onClose: () => void;
  onDelete: () => void;
  onCancelDownload: () => void;
  onImportLocalDir: () => void;
  onConfirmActivate: () => void;
}

export function JudgeDialog({
  dialogRef,
  mode,
  status,
  busy,
  progress,
  error,
  onBackdropClick,
  onClose,
  onDelete,
  onCancelDownload,
  onImportLocalDir,
  onConfirmActivate,
}: JudgeDialogProps) {
  return (
    /* Native modal <dialog> för aktivt val, nedladdning och borttagning av Lager 4 (MWG dialog & light-dismiss-a-dialog) */
    <dialog
      ref={dialogRef}
      closedby={busy ? "none" : "any"}
      aria-labelledby="judge-dialog-title"
      className="judge-dialog"
      onCancel={(e) => {
        if (busy) e.preventDefault();
      }}
      onClick={onBackdropClick}
    >
      <div className="judge-dialog-inner">
        <header className="judge-dialog-header">
          <div className="setup-icon-badge" aria-hidden="true">
            <Sparkles size={20} />
          </div>
          <div className="judge-dialog-title-wrap">
            <h2 id="judge-dialog-title" className="judge-dialog-title">
              {mode === "manage" && status.installed
                ? "Hantera Lager 4 — Helhetsbedömning (Judge)"
                : "Aktivera Lager 4 — Helhetsbedömning (Judge)"}
            </h2>
            <p className="judge-dialog-subtitle">
              Aktivt tillval enbart för vyn{" "}
              <strong>Klassificera information</strong> · 100 % lokalt på din
              dator
            </p>
          </div>
        </header>

        <div className="judge-dialog-body">
          <div className="judge-comparison-grid">
            <div className="judge-compare-card">
              <strong>Lager 3 (Standard · styckenivå)</strong>
              <p>
                Granskar texten stycke för stycke på under 1 sekund. Kan ibland
                ge för högt utslag (t.ex. Röd på ett kursreferat) när enstaka
                ord liknar känsliga ämnen utan att modellen ser helheten.
              </p>
            </div>
            <div className="judge-compare-card highlight">
              <strong>Lager 4 — Judge (Helhetsbedömning)</strong>
              <p>
                En lokal språkmodell (<em>Gemma 4 E2B</em>) läser dokumentets
                inledning, rubriker, avslutning och flaggade stycken för att
                bedöma <strong>dokumentets helhet och syfte</strong> samt ge en
                kort motivering på svenska.
              </p>
            </div>
          </div>

          <dl className="judge-facts-list">
            <div className="judge-fact-row">
              <dt>Lagringsutrymme</dt>
              <dd>
                {status.installed && status.fileSizeBytes > 0 ? (
                  <>
                    Installerad lokalt på datorn (
                    <strong>
                      {(status.fileSizeBytes / 1_000_000_000)
                        .toFixed(2)
                        .replace(".", ",")}{" "}
                      GB
                    </strong>
                    ). Du kan när som helst ta bort modellen nedan för att
                    frigöra utrymmet.
                  </>
                ) : (
                  <>
                    Kräver ca <strong>2,62 GB</strong> ledigt lagringsutrymme
                    (engångsnedladdning, därefter helt offline). Kan tas bort
                    med ett klick när du vill frigöra utrymmet.
                  </>
                )}
              </dd>
            </div>
            <div className="judge-fact-row">
              <dt>Tidsåtgång</dt>
              <dd>
                Tar lite längre tid vid klassificering — räkna med ca{" "}
                <strong>3–8 sekunder extra</strong> per dokument medan modellen
                tolkar helheten.
              </dd>
            </div>
            <div className="judge-fact-row">
              <dt>Beslutsstöd &amp; säkerhet</dt>
              <dd>
                Ingen text lämnar datorn. Skarpa fynd i Lager 1 (personnummer,
                bankkonton, kortnummer och lösenord/nycklar) har alltid vetorätt
                för <strong>RÖD</strong> nivå — och det är alltid du som fattar
                det slutliga beslutet.
              </dd>
            </div>
          </dl>

          {busy && (
            <div className="setup-progress-section judge-progress-box">
              <div
                id="judge-progress-label"
                className="setup-progress-meta"
                role="status"
                aria-live="polite"
              >
                <span className="setup-progress-status">
                  {progress?.label || "Förbereder Lager 4…"}
                </span>
                {progress?.phase === "downloading" &&
                  progress.totalBytes > 0 && (
                    <span className="setup-progress-bytes">
                      {(progress.downloadedBytes / 1_000_000_000)
                        .toFixed(2)
                        .replace(".", ",")}{" "}
                      av{" "}
                      {(progress.totalBytes / 1_000_000_000)
                        .toFixed(2)
                        .replace(".", ",")}{" "}
                      GB (
                      {Math.min(
                        100,
                        Math.max(
                          1,
                          Math.round(
                            (progress.downloadedBytes / progress.totalBytes) *
                              100
                          )
                        )
                      )}{" "}
                      %)
                    </span>
                  )}
              </div>
              <progress
                className="setup-progress-bar"
                aria-labelledby="judge-progress-label"
                max={100}
                value={
                  progress?.phase === "downloading" && progress.totalBytes > 0
                    ? Math.min(
                        100,
                        Math.max(
                          1,
                          Math.round(
                            (progress.downloadedBytes / progress.totalBytes) *
                              100
                          )
                        )
                      )
                    : undefined
                }
              />
            </div>
          )}

          {error && (
            <div className="setup-error-box" role="alert">
              <div className="setup-error-header">
                <AlertTriangle size={15} aria-hidden="true" />
                <strong>Åtgärden kunde inte slutföras</strong>
              </div>
              <p className="setup-error-detail">{error}</p>
            </div>
          )}
        </div>

        <footer className="judge-dialog-footer">
          {mode === "manage" && status.installed ? (
            <>
              <button
                type="button"
                className="tb-btn"
                disabled={busy}
                onClick={onClose}
              >
                <span>Stäng</span>
              </button>
              <button
                type="button"
                className="tb-btn judge-danger-btn"
                disabled={busy}
                onClick={onDelete}
              >
                <Trash2 size={15} aria-hidden="true" />
                <span>
                  {busy
                    ? "Tar bort modellfil…"
                    : "Ta bort Lager 4 & frigör ~2,6 GB"}
                </span>
              </button>
            </>
          ) : busy ? (
            <>
              <button
                type="button"
                className="tb-btn"
                onClick={onCancelDownload}
              >
                <span>Avbryt nedladdning</span>
              </button>
            </>
          ) : (
            <>
              <button type="button" className="tb-btn" onClick={onClose}>
                <span>Avbryt</span>
              </button>
              <button
                type="button"
                className="tb-btn"
                onClick={onImportLocalDir}
                title="Importera GGUF-filen offline från lokal mapp eller USB"
              >
                <FolderOpen size={15} aria-hidden="true" />
                <span>Välj lokal mapp…</span>
              </button>
              <button
                type="button"
                className="tb-btn-primary"
                onClick={onConfirmActivate}
              >
                <Download size={15} aria-hidden="true" />
                <span>Bekräfta &amp; ladda ner (~2,6 GB)</span>
              </button>
            </>
          )}
        </footer>
      </div>
    </dialog>
  );
}
