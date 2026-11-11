import React from "react";
import { Match } from "./types";

export function matchKeyStr(m: { start: number; end: number; text: string }): string {
  return `${m.start}:${m.end}:${m.text}`;
}

/**
 * Bevarar befintliga maskeringar (och användarens undantag i ignoredKeys) när källtexten
 * redigeras, genom att justera start/end-offsets efter redigeringspunkten.
 */
export function reconcileMatchesAfterEdit(
  oldText: string,
  newText: string,
  prevMatches: Match[],
  prevIgnored: Set<string>
): { matches: Match[]; ignoredKeys: Set<string> } {
  if (oldText === newText) {
    return { matches: prevMatches, ignoredKeys: prevIgnored };
  }
  if (!newText || prevMatches.length === 0) {
    return { matches: [], ignoredKeys: new Set() };
  }

  const minLen = Math.min(oldText.length, newText.length);
  let prefixLen = 0;
  while (prefixLen < minLen && oldText.charCodeAt(prefixLen) === newText.charCodeAt(prefixLen)) {
    prefixLen++;
  }

  let suffixLen = 0;
  while (
    suffixLen < minLen - prefixLen &&
    oldText.charCodeAt(oldText.length - 1 - suffixLen) ===
      newText.charCodeAt(newText.length - 1 - suffixLen)
  ) {
    suffixLen++;
  }

  const oldEditEnd = oldText.length - suffixLen;
  const delta = newText.length - oldText.length;

  const nextMatches: Match[] = [];
  const nextIgnored = new Set<string>();

  for (const m of prevMatches) {
    const oldKey = matchKeyStr(m);
    const wasIgnored = prevIgnored.has(oldKey);

    if (m.end <= prefixLen) {
      nextMatches.push(m);
      if (wasIgnored) nextIgnored.add(oldKey);
    } else if (m.start >= oldEditEnd) {
      const shifted: Match = {
        ...m,
        start: m.start + delta,
        end: m.end + delta,
      };
      if (
        shifted.start >= 0 &&
        shifted.end <= newText.length &&
        newText.slice(shifted.start, shifted.end) === shifted.text
      ) {
        nextMatches.push(shifted);
        if (wasIgnored) nextIgnored.add(matchKeyStr(shifted));
      }
    } else {
      // Träffen överlappade det redigerade området — behåll endast om exakt ord står kvar på plats
      if (
        m.start >= 0 &&
        m.end <= newText.length &&
        newText.slice(m.start, m.end) === m.text
      ) {
        nextMatches.push(m);
        if (wasIgnored) nextIgnored.add(oldKey);
      }
    }
  }

  return { matches: nextMatches, ignoredKeys: nextIgnored };
}

/**
 * Renderar inline-formatering (**fetstil**, `kod`) säkert som React-noder utan innerHTML.
 */
export function renderInlineFormattedText(
  text: string,
  keyPrefix: string
): React.ReactNode {
  if (!text) return null;
  if (!text.includes("**") && !text.includes("`") && !text.includes("*")) {
    return text;
  }

  const tokenRe = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\*(?!\s)[^*\n]+(?<!\s)\*)/g;
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;
  let match: RegExpExecArray | null;
  let idx = 0;

  while ((match = tokenRe.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.slice(lastIdx, match.index));
    }
    const tok = match[0];
    if (tok.startsWith("**") && tok.endsWith("**")) {
      parts.push(
        <strong key={`${keyPrefix}-b-${idx++}`}>{tok.slice(2, -2)}</strong>
      );
    } else if (tok.startsWith("`") && tok.endsWith("`")) {
      parts.push(
        <code key={`${keyPrefix}-c-${idx++}`} className="inline-doc-code">
          {tok.slice(1, -1)}
        </code>
      );
    } else if (tok.startsWith("*") && tok.endsWith("*")) {
      parts.push(<em key={`${keyPrefix}-i-${idx++}`}>{tok.slice(1, -1)}</em>);
    } else {
      parts.push(tok);
    }
    lastIdx = tokenRe.lastIndex;
  }

  if (lastIdx < text.length) {
    parts.push(text.slice(lastIdx));
  }

  return parts.length === 1 ? parts[0] : parts;
}

/**
 * Renderar Markdown-text (t.ex. ett AI-svar från ChatGPT/Claude/Copilot/Gemini) som
 * ren dokumenttypografi (rubriker, punktlistor, numrerade listor, citat, fetstil)
 * helt i React JSX utan XSS-risk (ingen dangerouslySetInnerHTML).
 */
export function renderFormattedMarkdown(text: string): React.ReactNode {
  if (!text.trim()) return null;

  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let blockKey = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // Kodblock ```
    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++;
      blocks.push(
        <pre key={`md-pre-${blockKey++}`} className="md-code-block">
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // Horisontell linje ---
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push(<hr key={`md-hr-${blockKey++}`} className="md-hr" />);
      i++;
      continue;
    }

    // Rubriker # .. ####
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const content = renderInlineFormattedText(
        headingMatch[2].trim(),
        `md-h-${blockKey}`
      );
      if (level === 1) {
        blocks.push(
          <h2 key={`md-h1-${blockKey++}`} className="md-h1">
            {content}
          </h2>
        );
      } else if (level === 2) {
        blocks.push(
          <h3 key={`md-h2-${blockKey++}`} className="md-h2">
            {content}
          </h3>
        );
      } else {
        blocks.push(
          <h4 key={`md-h3-${blockKey++}`} className="md-h3">
            {content}
          </h4>
        );
      }
      i++;
      continue;
    }

    // Fristående fetstilt rubrikrad: **RUBRIK**
    if (
      trimmed.startsWith("**") &&
      trimmed.endsWith("**") &&
      trimmed.length > 4 &&
      !trimmed.slice(2, -2).includes("**")
    ) {
      blocks.push(
        <h3 key={`md-bh-${blockKey++}`} className="md-h2">
          {trimmed.slice(2, -2)}
        </h3>
      );
      i++;
      continue;
    }

    // Punktlista (- / * / •)
    if (/^\s*[-*•]\s+/.test(line)) {
      const items: React.ReactNode[] = [];
      let liIdx = 0;
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        const itemText = lines[i].replace(/^\s*[-*•]\s+/, "");
        items.push(
          <li key={`md-ul-li-${blockKey}-${liIdx++}`}>
            {renderInlineFormattedText(itemText, `md-ul-${blockKey}-${liIdx}`)}
          </li>
        );
        i++;
      }
      blocks.push(
        <ul key={`md-ul-${blockKey++}`} className="md-list">
          {items}
        </ul>
      );
      continue;
    }

    // Numrerad lista (1. / 1))
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: React.ReactNode[] = [];
      let liIdx = 0;
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        const itemText = lines[i].replace(/^\s*\d+[.)]\s+/, "");
        items.push(
          <li key={`md-ol-li-${blockKey}-${liIdx++}`}>
            {renderInlineFormattedText(itemText, `md-ol-${blockKey}-${liIdx}`)}
          </li>
        );
        i++;
      }
      blocks.push(
        <ol key={`md-ol-${blockKey++}`} className="md-list md-ordered-list">
          {items}
        </ol>
      );
      continue;
    }

    // Citatblock (> )
    if (trimmed.startsWith(">")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      blocks.push(
        <blockquote key={`md-q-${blockKey++}`} className="md-blockquote">
          {renderInlineFormattedText(quoteLines.join(" "), `md-qi-${blockKey}`)}
        </blockquote>
      );
      continue;
    }

    // Vanligt brödtextstycke (samla intilliggande rader fram till tomrad eller list/rubrik)
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith(">") &&
      !/^\s*[-*•]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^(-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }

    blocks.push(
      <p key={`md-p-${blockKey++}`} className="md-para">
        {paraLines.map((pl, idx) => (
          <React.Fragment key={idx}>
            {idx > 0 && <br />}
            {renderInlineFormattedText(pl, `md-pl-${blockKey}-${idx}`)}
          </React.Fragment>
        ))}
      </p>
    );
  }

  return <div className="formatted-md">{blocks}</div>;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatInlineHtml(text: string): string {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*(?!\s)([^*\n]+)(?<!\s)\*/g, "<em>$1</em>");
}

/**
 * Konverterar Markdown-text till ren, HTML-escapad semantisk HTML för urklipp (`text/html`),
 * så att inklistring i Word, Outlook, Teams eller Google Docs ger riktiga rubriker, fetstil och punktlistor.
 */
export function markdownToCleanHtml(text: string): string {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      i++;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(escapeHtml(lines[i]));
        i++;
      }
      if (i < lines.length) i++;
      out.push(`<pre><code>${codeLines.join("\n")}</code></pre>`);
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      out.push("<hr>");
      i++;
      continue;
    }

    const hm = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hm) {
      const lvl = Math.min(4, hm[1].length + 1);
      out.push(`<h${lvl}>${formatInlineHtml(hm[2].trim())}</h${lvl}>`);
      i++;
      continue;
    }

    if (
      trimmed.startsWith("**") &&
      trimmed.endsWith("**") &&
      trimmed.length > 4 &&
      !trimmed.slice(2, -2).includes("**")
    ) {
      out.push(`<h3>${escapeHtml(trimmed.slice(2, -2))}</h3>`);
      i++;
      continue;
    }

    if (/^\s*[-*•]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        items.push(
          `<li>${formatInlineHtml(lines[i].replace(/^\s*[-*•]\s+/, ""))}</li>`
        );
        i++;
      }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(
          `<li>${formatInlineHtml(lines[i].replace(/^\s*\d+[.)]\s+/, ""))}</li>`
        );
        i++;
      }
      out.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (trimmed.startsWith(">")) {
      const qLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        qLines.push(formatInlineHtml(lines[i].trim().replace(/^>\s?/, "")));
        i++;
      }
      out.push(`<blockquote>${qLines.join(" ")}</blockquote>`);
      continue;
    }

    const pLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith(">") &&
      !/^\s*[-*•]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i])
    ) {
      pLines.push(formatInlineHtml(lines[i]));
      i++;
    }
    out.push(`<p>${pLines.join("<br>")}</p>`);
  }

  return out.join("\n");
}

/**
 * Konverterar Markdown till ren läsbar text utan råa `**` eller `###`-tecken.
 */
export function markdownToCleanPlainText(text: string): string {
  return text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => {
      let l = line;
      l = l.replace(/^\s*#{1,6}\s+/, "");
      l = l.replace(/^\s*[-*]\s+/, "• ");
      l = l.replace(/\*\*([^*\n]+)\*\*/g, "$1");
      l = l.replace(/`([^`\n]+)`/g, "$1");
      l = l.replace(/\*(?!\s)([^*\n]+)(?<!\s)\*/g, "$1");
      return l;
    })
    .join("\n");
}
