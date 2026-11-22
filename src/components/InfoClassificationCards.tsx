import {
  AlertTriangle,
  CheckCircle2,
  Shield,
  Sparkles,
  XCircle,
} from "lucide-react";
import type { InfoClassificationReport, JudgeModelStatus } from "../types";

export interface InfoClassificationCardsProps {
  report: InfoClassificationReport;
  judgeEnabled: boolean;
  judgeStatus: JudgeModelStatus;
  infoEditedSinceScan: boolean;
  onRunJudgeClassification: () => void;
}

export function InfoClassificationCards({
  report,
  judgeEnabled,
  judgeStatus,
  infoEditedSinceScan,
  onRunJudgeClassification,
}: InfoClassificationCardsProps) {
  return (
    <section
      aria-label="Resultat av informationsklassning"
      className="info-cards-grid"
    >
      {/* Kort 1 (MWG card: <article> med direkta <header>- och <div>-regioner): Fråga 1 — Vad innehåller informationen? */}
      <article className={`info-card level-${report.level}`}>
        <header className="info-card-header">
          <div className="info-card-header-top">
            <span className={`info-level-badge level-${report.level}`}>
              {report.level === "green" ? (
                <CheckCircle2 size={15} aria-hidden="true" />
              ) : report.level === "yellow" ? (
                <AlertTriangle size={15} aria-hidden="true" />
              ) : (
                <XCircle size={15} aria-hidden="true" />
              )}
              <span>{report.badgeTitle}</span>
            </span>
            <span
              className="info-scan-meta"
              title={
                report.judge
                  ? `Lager 1–3: ${report.scanDurationMs} ms · Lager 4 (Judge): ${report.judge.durationMs} ms`
                  : `Lokal analys med Regex, BERT NER och EmbeddingGemma 2 (${report.scanDurationMs} ms)`
              }
            >
              {report.paragraphsScanned} avsnitt ·{" "}
              {report.judge
                ? `${((report.scanDurationMs + report.judge.durationMs) / 1000)
                    .toFixed(1)
                    .replace(".", ",")} s (inkl. Lager 4)`
                : `${report.scanDurationMs} ms`}
            </span>
          </div>
          <h2 className="info-card-title">1. Vad innehåller informationen?</h2>
        </header>

        <div className="info-card-content">
          <p className="info-lead">
            <strong>{report.contentHeadline}.</strong>{" "}
            {report.contentExplanation}
          </p>

          {report.judge ? (
            <div className="info-judge-box" role="note">
              <div className="info-judge-header">
                <span className="info-judge-badge">
                  <Sparkles size={12} aria-hidden="true" />
                  <span>Lager 4 · Judge (Helhet)</span>
                </span>
                <span className="info-judge-doctype">
                  {report.judge.documentType}
                </span>
                {report.judge.hardVetoApplied ? (
                  <span className="info-judge-delta veto">
                    Lager 1-veto: RÖD behålls
                  </span>
                ) : report.judge.levelAdjusted ? (
                  <span className="info-judge-delta adjusted">
                    Justerad{" "}
                    {report.judge.layer3Level === "red"
                      ? "Röd"
                      : report.judge.layer3Level === "yellow"
                      ? "Gul"
                      : "Grön"}{" "}
                    →{" "}
                    {report.judge.finalLevel === "red"
                      ? "Röd"
                      : report.judge.finalLevel === "yellow"
                      ? "Gul"
                      : "Grön"}
                  </span>
                ) : (
                  <span className="info-judge-delta confirmed">
                    Bekräftar{" "}
                    {report.judge.finalLevel === "red"
                      ? "Röd"
                      : report.judge.finalLevel === "yellow"
                      ? "Gul"
                      : "Grön"}
                  </span>
                )}
              </div>
              <p className="info-judge-rationale">{report.judge.rationale}</p>
            </div>
          ) : (
            judgeEnabled &&
            judgeStatus.installed &&
            !infoEditedSinceScan && (
              <div className="info-judge-rerun-hint">
                <span>
                  Lager 4 (Judge) är nu aktiverat men ingick inte i denna
                  körning.
                </span>
                <button
                  type="button"
                  className="tb-btn-subtle rescan-chip"
                  onClick={onRunJudgeClassification}
                >
                  <Sparkles size={12} aria-hidden="true" />
                  <span>Kör helhetsbedömning (Lager 4)</span>
                </button>
              </div>
            )
          )}

          {report.signals.length > 0 && (
            <ul className="info-signals-list" role="list">
              {report.signals.map((sig, idx) => (
                <li key={`${sig.title}-${idx}`} className="info-signal-item">
                  <span
                    className={`sev-pill ${
                      sig.level === "red"
                        ? "high"
                        : sig.level === "yellow"
                        ? "medium"
                        : "green"
                    }`}
                  >
                    {sig.level === "red" ? (
                      <XCircle size={11} aria-hidden="true" />
                    ) : sig.level === "yellow" ? (
                      <AlertTriangle size={11} aria-hidden="true" />
                    ) : (
                      <CheckCircle2 size={11} aria-hidden="true" />
                    )}
                    <span>
                      {sig.level === "red"
                        ? "Röd"
                        : sig.level === "yellow"
                        ? "Gul"
                        : "Grön"}
                    </span>
                  </span>
                  <div className="info-signal-text">
                    <strong>{sig.title}</strong>
                    <span>{sig.detail}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </article>

      {/* Kort 2 (MWG card): Fråga 2 — Hur får vi hantera informationen? */}
      <article className={`info-card guidance-card level-${report.level}`}>
        <header className="info-card-header">
          <div className="info-card-header-top">
            <span className="info-step-pill">
              Lokalt beslutsstöd · Lagring &amp; delning
            </span>
          </div>
          <h2 className="info-card-title">2. {report.handling.heading}</h2>
        </header>

        <div className="info-card-content">
          <p className="info-lead">{report.handling.summary}</p>
          <dl className="info-guidance-dl">
            <div className="info-guidance-row">
              <dt>Lagring</dt>
              <dd>{report.handling.storageGuidance}</dd>
            </div>
            <div className="info-guidance-row">
              <dt>Delning</dt>
              <dd>{report.handling.sharingGuidance}</dd>
            </div>
            <div className="info-guidance-row">
              <dt>Exempel</dt>
              <dd>{report.handling.examplesNote}</dd>
            </div>
          </dl>
          <div className="info-decision-callout" role="note">
            <Shield
              size={16}
              className="info-callout-icon"
              aria-hidden="true"
            />
            <p className="info-callout-text">
              <strong>
                Lokalt beslutsstöd — det är alltid du som fattar beslutet.
              </strong>{" "}
              {report.handling.disclaimer}
            </p>
          </div>
        </div>
      </article>
    </section>
  );
}
