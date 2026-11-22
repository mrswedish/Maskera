import {
  AlertTriangle,
  Check,
  FolderOpen,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import type { SetupProgress } from "../types";

export interface FirstTimeSetupProps {
  nerDoneInSetup: boolean;
  gemmaDoneInSetup: boolean;
  setupBusy: boolean;
  setupError: string | null;
  setupProgress: SetupProgress | null;
  setupBytesText: string;
  isDeterminateSetupProgress: boolean;
  setupPercent: number;
  onRetry: () => void;
  onPickLocalModelDir: () => void;
  onSkipOffline: () => void;
}

export function FirstTimeSetup({
  nerDoneInSetup,
  gemmaDoneInSetup,
  setupBusy,
  setupError,
  setupProgress,
  setupBytesText,
  isDeterminateSetupProgress,
  setupPercent,
  onRetry,
  onPickLocalModelDir,
  onSkipOffline,
}: FirstTimeSetupProps) {
  return (
    /* ── Alternativ A: Välkomst- & engångsinstallationsvy (visas endast om modeller saknas vid start) ── */
    <main className="setup-workspace">
      <section className="setup-card" aria-labelledby="setup-heading">
        <div className="setup-card-header">
          <div className="setup-icon-badge" aria-hidden="true">
            <Sparkles size={22} />
          </div>
          <div className="setup-title-group">
            <h2 id="setup-heading" className="setup-title">
              Förbereder lokal AI — engångsnedladdning
            </h2>
            <p className="setup-subtitle">
              Maskera hämtar nu språk- och kontextmodellerna (~318 MB totalt) en
              enda gång till din dator. Därefter körs all avidentifiering och
              säkerhetskontroll <strong>100 % lokalt och offline</strong> —
              ingen text lämnar någonsin din dator.
            </p>
          </div>
        </div>

        <ul className="setup-model-list" role="list">
          <li
            className={`setup-model-item ${
              nerDoneInSetup
                ? "done"
                : setupBusy && !setupError
                ? "active"
                : ""
            }`}
          >
            <div className="setup-model-info">
              <div className="setup-model-name-row">
                <strong>1. Svensk BERT NER</strong>
                <span className="setup-model-size">~121 MB</span>
              </div>
              <p className="setup-model-desc">
                Hittar personnamn, organisationer, platser, tider och händelser
                i svensk löptext (KBLab/Kungliga biblioteket).
              </p>
            </div>
            <span
              className={`setup-step-pill ${
                nerDoneInSetup
                  ? "done"
                  : setupBusy && !setupError
                  ? "active"
                  : "pending"
              }`}
            >
              {nerDoneInSetup ? (
                <>
                  <Check size={13} aria-hidden="true" />
                  <span>Klar</span>
                </>
              ) : setupBusy && !setupError ? (
                <>
                  <RefreshCw size={13} className="spin" aria-hidden="true" />
                  <span>Hämtar…</span>
                </>
              ) : (
                <span>Väntar</span>
              )}
            </span>
          </li>

          <li
            className={`setup-model-item ${
              gemmaDoneInSetup
                ? "done"
                : setupBusy && !setupError && nerDoneInSetup
                ? "active"
                : ""
            }`}
          >
            <div className="setup-model-info">
              <div className="setup-model-name-row">
                <strong>2. EmbeddingGemma 2</strong>
                <span className="setup-model-size">~197 MB</span>
              </div>
              <p className="setup-model-desc">
                Semantisk säkerhetskontroll som varnar för
                patient-/vårduppgifter, företagshemligheter, sårbarheter och
                HR-ärenden.
              </p>
            </div>
            <span
              className={`setup-step-pill ${
                gemmaDoneInSetup
                  ? "done"
                  : setupBusy && !setupError && nerDoneInSetup
                  ? "active"
                  : "pending"
              }`}
            >
              {gemmaDoneInSetup ? (
                <>
                  <Check size={13} aria-hidden="true" />
                  <span>Klar</span>
                </>
              ) : setupBusy && !setupError && nerDoneInSetup ? (
                <>
                  <RefreshCw size={13} className="spin" aria-hidden="true" />
                  <span>Hämtar…</span>
                </>
              ) : (
                <span>Väntar</span>
              )}
            </span>
          </li>
        </ul>

        <div className="setup-progress-section">
          <div
            id="setup-progress-label"
            className="setup-progress-meta"
            role="status"
            aria-live="polite"
          >
            <span className="setup-progress-status">
              {setupError
                ? "Nedladdningen avbröts"
                : setupProgress?.label || "Ansluter till HuggingFace…"}
            </span>
            {setupBytesText && !setupError && (
              <span className="setup-progress-bytes">{setupBytesText}</span>
            )}
          </div>
          <progress
            id="setup-progress-bar"
            className="setup-progress-bar"
            aria-labelledby="setup-progress-label"
            max={100}
            value={isDeterminateSetupProgress ? setupPercent : undefined}
          />
        </div>

        {setupError ? (
          <div className="setup-error-box" role="alert">
            <div className="setup-error-header">
              <AlertTriangle size={16} aria-hidden="true" />
              <strong>Kunde inte hämta AI-modellerna automatiskt</strong>
            </div>
            <p className="setup-error-detail">{setupError}</p>
            <div className="setup-error-actions">
              <button
                type="button"
                className="tb-btn-primary"
                onClick={onRetry}
              >
                <RefreshCw size={15} aria-hidden="true" />
                <span>Försök igen</span>
              </button>
              <button
                type="button"
                className="tb-btn"
                onClick={onPickLocalModelDir}
              >
                <FolderOpen size={15} aria-hidden="true" />
                <span>Välj lokal modellmapp…</span>
              </button>
              <button
                type="button"
                className="tb-btn-subtle"
                onClick={onSkipOffline}
              >
                <span>Öppna ändå (endast mönster/regex offline)</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="setup-offline-footer">
            <span>
              Har du redan modellfilerna på ett USB-minne eller en intern
              nätverksdisk?
            </span>
            <button
              type="button"
              className="tb-btn-subtle"
              onClick={onPickLocalModelDir}
            >
              <FolderOpen size={13} aria-hidden="true" />
              <span>Välj lokal modellmapp…</span>
            </button>
          </div>
        )}
      </section>
    </main>
  );
}
