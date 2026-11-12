import React, { useState, useMemo, useCallback, useEffect, useRef } from "react";
import {
  Shield,
  Upload,
  Clipboard,
  Copy,
  Download,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  FileText,
  Table,
  RotateCcw,
  Sparkles,
  Sliders,
  ChevronDown,
  Plus,
  RefreshCw,
  Check,
  Sun,
  Moon,
  Monitor,
  Pencil,
  FolderOpen,
  FileSearch,
  Trash2,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import "./index.css";
import type {
  Match,
  ParagraphFinding,
  ClassificationReport,
  SafetyVerdict,
  ImportedDocument,
  ReplacementPair,
  ModelsStatus,
  SetupProgress,
  InfoClassificationReport,
  InfoSectionFinding,
} from "./types";
import {
  buildTranslationTable,
  buildMaskedText,
  unmaskText,
} from "./translationUtils";
import {
  reconcileMatchesAfterEdit,
  renderInlineFormattedText,
  renderFormattedMarkdown,
  markdownToCleanHtml,
  markdownToCleanPlainText,
} from "./markdownUtils";

const REGEX_LABELS = [
  "Personnummer",
  "E-post",
  "Telefonnummer",
  "Org.nummer",
  "Konto/Giro",
  "IP-adress",
  "Kortnummer",
  "Hemlighet/Nyckel",
] as const;

const ENTITY_LABELS = [
  "Person",
  "Organisation",
  "Plats",
  "Övrigt",
  "Tid",
  "Händelse",
] as const;

const ALL_CATEGORIES = [...REGEX_LABELS, ...ENTITY_LABELS];

const PRESETS = {
  ai_safe: {
    name: "AI-säker",
    entities: [
      "Personnummer",
      "E-post",
      "Telefonnummer",
      "Org.nummer",
      "Konto/Giro",
      "Kortnummer",
      "Hemlighet/Nyckel",
      "Person",
      "Organisation",
      "Plats",
      "Övrigt",
    ],
  },
  id_only: {
    name: "Endast ID & kontakt",
    entities: [
      "Personnummer",
      "E-post",
      "Telefonnummer",
      "Org.nummer",
      "Konto/Giro",
      "Kortnummer",
      "Hemlighet/Nyckel",
    ],
  },
  strict: {
    name: "Strikt (Allt)",
    entities: [...ALL_CATEGORIES],
  },
} as const;

type PresetKey = keyof typeof PRESETS;

// WCAG AA-kalibrerade kategorifärger (>= 5.0:1 kontrast mot vit text #ffffff)
const COLORS: Record<string, string> = {
  Personnummer: "#b91c1c",
  "E-post": "#1d4ed8",
  Telefonnummer: "#b45309",
  "Org.nummer": "#7e22ce",
  "Konto/Giro": "#be185d",
  "IP-adress": "#0369a1",
  Kortnummer: "#991b1b",
  "Hemlighet/Nyckel": "#9f1239",
  Person: "#5b21b6",
  Organisation: "#047857",
  Plats: "#9d174d",
  Övrigt: "#475569",
  Tid: "#0e7490",
  Händelse: "#854d0e",
  Manuell: "#c2410c",
};

type ColorSchemeSetting = "light dark" | "light" | "dark";

function matchKey(m: Match): string {
  return `${m.start}-${m.end}`;
}

function fixMojibake(text: string): string {
  try {
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) {
      bytes[i] = text.charCodeAt(i) & 0xff;
    }
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return text;
  }
}

function buildByteToCharMap(text: string, needed: Set<number>): Map<number, number> {
  const byteToChar = new Map<number, number>();
  if (needed.size === 0) return byteToChar;

  let u16 = 0;
  let byteIdx = 0;
  const len = text.length;

  while (u16 < len) {
    if (needed.has(byteIdx)) {
      byteToChar.set(byteIdx, u16);
      if (byteToChar.size === needed.size) break;
    }
    const code = text.charCodeAt(u16);
    if (code < 0x80) {
      byteIdx += 1;
      u16 += 1;
    } else if (code < 0x800) {
      byteIdx += 2;
      u16 += 1;
    } else if (code >= 0xd800 && code <= 0xdbff && u16 + 1 < len) {
      byteIdx += 4;
      u16 += 2;
    } else {
      byteIdx += 3;
      u16 += 1;
    }
  }
  if (needed.has(byteIdx)) {
    byteToChar.set(byteIdx, u16);
  }
  return byteToChar;
}

function fixByteOffsets(text: string, rawMatches: Match[]): Match[] {
  if (rawMatches.length === 0) return rawMatches;

  const needed = new Set<number>();
  for (const m of rawMatches) {
    needed.add(m.start);
    needed.add(m.end);
  }

  const byteToChar = buildByteToCharMap(text, needed);

  return rawMatches.map((m) => ({
    ...m,
    start: byteToChar.get(m.start) ?? m.start,
    end: byteToChar.get(m.end) ?? m.end,
  }));
}

function fixFindingOffsets(
  text: string,
  rawFindings: ParagraphFinding[]
): ParagraphFinding[] {
  if (rawFindings.length === 0) return rawFindings;

  const needed = new Set<number>();
  for (const f of rawFindings) {
    needed.add(f.start);
    needed.add(f.end);
  }

  const byteToChar = buildByteToCharMap(text, needed);

  return rawFindings.map((f) => ({
    ...f,
    start: Math.min(text.length, byteToChar.get(f.start) ?? f.start),
    end: Math.min(text.length, byteToChar.get(f.end) ?? f.end),
  }));
}

function fixInfoSectionOffsets(
  text: string,
  rawFindings: InfoSectionFinding[]
): InfoSectionFinding[] {
  if (rawFindings.length === 0) return rawFindings;

  const needed = new Set<number>();
  for (const f of rawFindings) {
    needed.add(f.start);
    needed.add(f.end);
  }

  const byteToChar = buildByteToCharMap(text, needed);

  return rawFindings.map((f) => ({
    ...f,
    start: Math.min(text.length, byteToChar.get(f.start) ?? f.start),
    end: Math.min(text.length, byteToChar.get(f.end) ?? f.end),
  }));
}

function normalizeInputText(raw: string): string {
  return fixMojibake(raw)
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\u00a0/g, " ")
    .replace(/\u2028/g, "\n")
    .replace(/\u2029/g, "\n")
    .replace(/\ufffd/g, " ");
}

export default function App() {
  // Huvudläge (Alternativ B: två separata system — Avidentifiera för AI vs Klassificera information)
  const [appMode, setAppMode] = useState<"mask" | "classify">("mask");
  const [infoClassifying, setInfoClassifying] = useState(false);
  const [infoReport, setInfoReport] = useState<InfoClassificationReport | null>(
    null
  );
  const [isEditingClassify, setIsEditingClassify] = useState(true);
  const [infoEditedSinceScan, setInfoEditedSinceScan] = useState(false);
  const [showInfoSectionHighlights, setShowInfoSectionHighlights] =
    useState(true);

  // Flikar (i AI-avidentifieringsläget)
  const [activeTab, setActiveTab] = useState<"mask" | "table" | "unmask">("mask");

  // Källtext & dokumentimport (Word .docx/.odt, Excel .xlsx/.xls/.ods, PDF .pdf, Text)
  const [sourceText, setSourceText] = useState("");
  const [fileName, setFileName] = useState("");
  const [docSummary, setDocSummary] = useState<string>("");
  const [originalDocxBytes, setOriginalDocxBytes] = useState<number[] | null>(
    null
  );
  const [originalXlsxBytes, setOriginalXlsxBytes] = useState<number[] | null>(
    null
  );
  const [importingDoc, setImportingDoc] = useState(false);
  const [isEditing, setIsEditing] = useState(true);

  // Inställningar & filter
  const [activePreset, setActivePreset] = useState<PresetKey | null>("ai_safe");
  const [selectedEntities, setSelectedEntities] = useState<string[]>([
    ...PRESETS.ai_safe.entities,
  ]);
  const [threshold, setThreshold] = useState(0.6);
  const [showConfig, setShowConfig] = useState(false);

  // Analys-tillstånd
  const [analyzing, setAnalyzing] = useState(false);
  const [classifying, setClassifying] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [matches, setMatches] = useState<Match[]>([]);
  const [ignoredKeys, setIgnoredKeys] = useState<Set<string>>(new Set());
  const [lastDurationMs, setLastDurationMs] = useState<number | null>(null);
  const [nerProgress, setNerProgress] = useState<{
    chunk: number;
    total: number;
  } | null>(null);

  // AI-gatekeeper (EmbeddingGemma 2 + Secret regex)
  const [report, setReport] = useState<ClassificationReport | null>(null);
  const [rawSemanticFindings, setRawSemanticFindings] = useState<
    ParagraphFinding[]
  >([]);
  const [dismissedFindings, setDismissedFindings] = useState<Set<number>>(
    new Set()
  );
  const [showParaHighlights, setShowParaHighlights] = useState(true);
  const [focusedParaIndex, setFocusedParaIndex] = useState<number | null>(null);
  const [textEditedSinceScan, setTextEditedSinceScan] = useState(false);
  const [editingSlice, setEditingSlice] = useState<{
    start: number;
    end: number;
    draft: string;
  } | null>(null);
  const [nerInstalled, setNerInstalled] = useState(false);
  const [gemmaInstalled, setGemmaInstalled] = useState(false);
  const [gemmaLoading, setGemmaLoading] = useState(false);
  const [gemmaReady, setGemmaReady] = useState(false);
  const [gemmaStatus, setGemmaStatus] = useState<string>("");
  const [gemmaScanProgress, setGemmaScanProgress] = useState<{
    chunk: number;
    total: number;
  } | null>(null);
  const [gemmaDownloadProgress, setGemmaDownloadProgress] = useState<{
    file: string;
    downloaded: number;
    total: number;
  } | null>(null);

  // Modellstatus (BERT NER) + Förstagångsinstallation (Alternativ A)
  const [nerReady, setNerReady] = useState(false);
  const [nerStatus, setNerStatus] = useState<string>("Kontrollerar BERT NER…");
  const [modelsChecked, setModelsChecked] = useState(false);
  const [showFirstTimeSetup, setShowFirstTimeSetup] = useState(false);
  const [setupProgress, setSetupProgress] = useState<SetupProgress | null>(
    null
  );
  const [setupError, setSetupError] = useState<string | null>(null);
  const [setupBusy, setSetupBusy] = useState(false);

  // Manuell entitet
  const [manualWord, setManualWord] = useState("");
  const [manualLabel, setManualLabel] = useState<string>("Person");

  // Avmaskering
  const [aiResponseInput, setAiResponseInput] = useState("");
  const [formattedUnmaskView, setFormattedUnmaskView] = useState(true);
  const [unmaskedResult, setUnmaskedResult] = useState<{
    text: string;
    replacements: number;
  }>({ text: "", replacements: 0 });

  // Kopieringsfeedback (Modern Web Guidance: copy-button live region)
  const [copyStatusMasked, setCopyStatusMasked] = useState<string>("");
  const [copyStatusUnmasked, setCopyStatusUnmasked] = useState<string>("");
  const copyTipMaskedRef = useRef<HTMLSpanElement>(null);
  const copyTipUnmaskedRef = useRef<HTMLSpanElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastCopyMsgRef = useRef<string>("");
  const copyTimerMaskedRef = useRef<number | undefined>(undefined);
  const copyTimerUnmaskedRef = useRef<number | undefined>(undefined);

  // Dra-och-släpp
  const [isDragOver, setIsDragOver] = useState(false);

  // Tema / Färgschema (Modern Web Guidance: dark-mode & dark-mode-toggle)
  const [colorScheme, setColorScheme] = useState<ColorSchemeSetting>(() => {
    try {
      const saved = localStorage.getItem("color-scheme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {}
    return "light dark";
  });
  const [systemPrefersDark, setSystemPrefersDark] = useState<boolean>(() =>
    typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
      : false
  );

  const renderedDark =
    colorScheme === "dark" ||
    (colorScheme === "light dark" && systemPrefersDark);

  const applyColorScheme = useCallback((nextScheme: ColorSchemeSetting) => {
    setColorScheme(nextScheme);
    const meta = document.querySelector<HTMLMetaElement>(
      'meta[name="color-scheme"]'
    );
    if (meta) {
      meta.content = nextScheme;
    }
    try {
      if (nextScheme === "light dark") {
        localStorage.removeItem("color-scheme");
      } else {
        localStorage.setItem("color-scheme", nextScheme);
      }
    } catch {}
  }, []);

  // Tvåläges-snabbknapp i headern enligt MWG dark-mode-toggle-algoritmen
  const handleQuickToggleTheme = useCallback(() => {
    const target: "light" | "dark" = renderedDark ? "light" : "dark";
    const systemScheme: "light" | "dark" = systemPrefersDark ? "dark" : "light";
    if (target === systemScheme) {
      applyColorScheme("light dark");
    } else {
      applyColorScheme(target);
    }
  }, [renderedDark, systemPrefersDark, applyColorScheme]);

  // Lyssna på OS-ändringar (prefers-color-scheme) och storage-ändringar mellan flikar
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onMqChange = (e: MediaQueryListEvent) => {
      setSystemPrefersDark(e.matches);
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key === "color-scheme") {
        const next: ColorSchemeSetting =
          e.newValue === "light" || e.newValue === "dark"
            ? e.newValue
            : "light dark";
        setColorScheme(next);
        const meta = document.querySelector<HTMLMetaElement>(
          'meta[name="color-scheme"]'
        );
        if (meta) meta.content = next;
      }
    };
    mq.addEventListener("change", onMqChange);
    window.addEventListener("storage", onStorage);
    return () => {
      mq.removeEventListener("change", onMqChange);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  // Polyfill focusgroup if not natively supported
  useEffect(() => {
    if (!("focusGroup" in HTMLElement.prototype)) {
      import("@microsoft/focusgroup-polyfill").then(({ polyfill }) => {
        try {
          polyfill(document.body);
        } catch {
          // graceful fallback
        }
      });
    }
  }, []);

  // Kör gemensam installation & laddning av både BERT NER och EmbeddingGemma 2
  const runEnsureAllModels = useCallback(async () => {
    setSetupError(null);
    setSetupBusy(true);
    try {
      const st = await invoke<ModelsStatus>("ensure_all_models");
      setNerInstalled(st.nerInstalled);
      setGemmaInstalled(st.gemmaInstalled);
      setNerReady(st.nerReady);
      setGemmaReady(st.gemmaReady);
      setSetupBusy(false);
      setShowFirstTimeSetup(false);
    } catch (e) {
      setSetupBusy(false);
      setSetupError(String(e));
    }
  }, []);

  // Importera modellfiler från valfri lokal mapp (t.ex. USB-minne / nätverksenhet / offline-paket)
  const handlePickLocalModelDir = useCallback(async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Välj mapp med AI-modeller (BERT NER / EmbeddingGemma 2)",
      });
      if (!selected || typeof selected !== "string") return;

      setSetupError(null);
      setSetupBusy(true);
      const st = await invoke<ModelsStatus>("import_models_from_dir", {
        srcDir: selected,
      });
      setNerInstalled(st.nerInstalled);
      setGemmaInstalled(st.gemmaInstalled);
      setNerReady(st.nerReady);
      setGemmaReady(st.gemmaReady);

      if (st.nerInstalled && st.gemmaInstalled) {
        setSetupBusy(false);
        setShowFirstTimeSetup(false);
      } else {
        // Om mappen bara innehöll en av modellerna, försök hämta den återstående
        await runEnsureAllModels();
      }
    } catch (e) {
      setSetupBusy(false);
      setSetupError(String(e));
    }
  }, [runEnsureAllModels]);

  // Ladda initiala modellstatusar och starta engångsinstallation vid behov (Alternativ A)
  useEffect(() => {
    let active = true;
    const unlisteners: (() => void)[] = [];

    Promise.all([
      listen<SetupProgress>("setup_progress", (e) => {
        if (!active) return;
        setSetupProgress(e.payload);
        if (e.payload.nerDone) setNerInstalled(true);
        if (e.payload.gemmaDone) setGemmaInstalled(true);
      }),
      listen<void>("model_ready", () => {
        if (!active) return;
        setNerReady(true);
        setNerInstalled(true);
      }),
      listen<{ chunk: number; total: number }>("ner_progress", (e) => {
        if (!active) return;
        setNerProgress(e.payload);
      }),
      listen<{ message: string }>("model_status", (e) => {
        if (!active) return;
        setNerStatus(e.payload.message);
      }),
      listen<{ file: string; downloaded: number; total: number }>(
        "download_progress",
        () => {}
      ),
      listen<void>("gemma_ready", () => {
        if (!active) return;
        setGemmaReady(true);
        setGemmaInstalled(true);
        setGemmaLoading(false);
        setGemmaDownloadProgress(null);
      }),
      listen<{ message: string }>("gemma_status", (e) => {
        if (!active) return;
        setGemmaStatus(e.payload.message);
      }),
      listen<{ chunk: number; total: number }>("gemma_scan_progress", (e) => {
        if (!active) return;
        setGemmaScanProgress(e.payload);
      }),
      listen<{ file: string; downloaded: number; total: number }>(
        "gemma_download_progress",
        (e) => {
          if (!active) return;
          setGemmaDownloadProgress(e.payload);
        }
      ),
    ]).then((fns) => {
      unlisteners.push(...fns);
    });

    invoke<ModelsStatus>("check_models_status")
      .then((st) => {
        if (!active) return;
        setNerInstalled(st.nerInstalled);
        setGemmaInstalled(st.gemmaInstalled);
        setNerReady(st.nerReady);
        setGemmaReady(st.gemmaReady);
        setModelsChecked(true);

        if (!st.nerInstalled || !st.gemmaInstalled) {
          // Minst en modell saknas på disk -> Visa förstagångsvyn (Alternativ A) och ladda ner båda direkt
          setShowFirstTimeSetup(true);
          runEnsureAllModels();
        } else {
          // Båda modellerna finns redan på disk -> Öppna editorn direkt och ladda in i RAM i bakgrunden
          setShowFirstTimeSetup(false);
          runEnsureAllModels();
        }
      })
      .catch((e) => {
        if (!active) return;
        setModelsChecked(true);
        setShowFirstTimeSetup(true);
        setSetupError(String(e));
      });

    return () => {
      active = false;
      unlisteners.forEach((f) => f());
    };
  }, [runEnsureAllModels]);

  // Aktiva träffar & översättningstabell
  const activeMatches = useMemo(
    () => matches.filter((m) => !ignoredKeys.has(matchKey(m))),
    [matches, ignoredKeys]
  );

  const activeTranslationTable = useMemo(
    () => buildTranslationTable(activeMatches),
    [activeMatches]
  );

  const maskedText = useMemo(
    () => buildMaskedText(sourceText, activeMatches, activeTranslationTable),
    [sourceText, activeMatches, activeTranslationTable]
  );

  // Kör full semantisk klassificering (Lager 3 EmbeddingGemma 2)
  const runSemanticClassification = useCallback(
    async (
      origText: string,
      currMatches: Match[],
      currIgnored: Set<string>,
      currDismissed: Set<number>,
      reuseCachedFindings: boolean
    ) => {
      if (!origText.trim()) {
        setReport(null);
        setRawSemanticFindings([]);
        return;
      }

      const isSecretOrCard = (label: string) =>
        label === "Hemlighet/Nyckel" || label === "Kortnummer";

      const currActive = currMatches.filter((m) => !currIgnored.has(matchKey(m)));
      const currTable = buildTranslationTable(currActive);
      const currMasked = buildMaskedText(origText, currActive, currTable);

      const activePii = currActive.filter(
        (m) => !isSecretOrCard(m.label)
      ).length;
      const unmaskedPii = currMatches
        .filter((m) => currIgnored.has(matchKey(m)))
        .filter((m) => !isSecretOrCard(m.label)).length;
      const totalSecrets = currMatches.filter((m) =>
        isSecretOrCard(m.label)
      ).length;
      const unmaskedSecrets = currMatches
        .filter((m) => currIgnored.has(matchKey(m)))
        .filter((m) => isSecretOrCard(m.label)).length;

      const canReuse =
        reuseCachedFindings &&
        report !== null &&
        report.semantic_available === gemmaReady;

      if (!canReuse) {
        setClassifying(true);
        setGemmaScanProgress(null);
      }

      const cachedFiltered = canReuse
        ? rawSemanticFindings.filter(
            (f) => !currDismissed.has(f.paragraph_index)
          )
        : null;

      try {
        const rep = await invoke<ClassificationReport>("classify_document", {
          originalText: origText,
          maskedText: currMasked,
          activePiiCount: activePii,
          unmaskedPiiCount: unmaskedPii,
          secretCount: totalSecrets,
          unmaskedSecretCount: unmaskedSecrets,
          cachedFindings: cachedFiltered,
          cachedParagraphsScanned: canReuse ? report.paragraphs_scanned : null,
          cachedDurationMs: canReuse ? report.scan_duration_ms : null,
        });
        if (!canReuse) {
          setRawSemanticFindings(rep.findings);
        }
        setReport(rep);
      } catch (e) {
        console.error("Klassificeringsfel:", e);
      } finally {
        if (!canReuse) {
          setClassifying(false);
          setGemmaScanProgress(null);
        }
      }
    },
    [report, rawSemanticFindings, gemmaReady]
  );

  // Uppdatera trafikljus direkt (< 1 ms med cachade styckefynd) när användaren klickar på ord eller klarmarkerar stycken
  useEffect(() => {
    if (!hasAnalyzed || !sourceText.trim()) return;
    runSemanticClassification(
      sourceText,
      matches,
      ignoredKeys,
      dismissedFindings,
      true
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ignoredKeys, matches, dismissedFindings]);

  // Om EmbeddingGemma 2 blir klar i bakgrunden efter att ett dokument redan analyserats, kör semantisk skanning
  useEffect(() => {
    if (gemmaReady && hasAnalyzed && sourceText.trim() && !report?.semantic_available) {
      runSemanticClassification(
        sourceText,
        matches,
        ignoredKeys,
        dismissedFindings,
        false
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gemmaReady]);

  // Uppdatera avmaskerat svar live
  useEffect(() => {
    const res = unmaskText(aiResponseInput, activeTranslationTable);
    setUnmaskedResult(res);
  }, [aiResponseInput, activeTranslationTable]);

  // Räknare per kategori för filter-chips
  const categoryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const m of activeMatches) {
      counts.set(m.label, (counts.get(m.label) ?? 0) + 1);
    }
    return counts;
  }, [activeMatches]);

  // Hantera preset-byte
  const handleSelectPreset = (key: PresetKey) => {
    setActivePreset(key);
    setSelectedEntities([...PRESETS[key].entities]);
  };

  const toggleCategory = (cat: string) => {
    setActivePreset(null);
    setSelectedEntities((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
    );
  };

  // Sömlös redigering av källtext som bevarar alla befintliga maskeringar genom offset-justering
  const handleSourceTextEdit = useCallback(
    (nextText: string) => {
      if (nextText === sourceText) return;
      const reconciled = reconcileMatchesAfterEdit(
        sourceText,
        nextText,
        matches,
        ignoredKeys
      );
      setSourceText(nextText);
      setMatches(reconciled.matches);
      setIgnoredKeys(reconciled.ignoredKeys);
      if (hasAnalyzed) {
        setTextEditedSinceScan(true);
      }
      if (!nextText.trim()) {
        setInfoReport(null);
        setInfoEditedSinceScan(false);
      } else if (infoReport !== null) {
        setInfoEditedSinceScan(true);
      }
    },
    [sourceText, matches, ignoredKeys, hasAnalyzed, infoReport]
  );

  // Klarmarkera / godkänn ett granskat avsnitt från EmbeddingGemma 2
  const toggleDismissFinding = useCallback((paraIndex: number) => {
    setDismissedFindings((prev) => {
      const next = new Set(prev);
      if (next.has(paraIndex)) {
        next.delete(paraIndex);
      } else {
        next.add(paraIndex);
      }
      return next;
    });
  }, []);

  // Rensa aktuellt dokument
  const handleClearDocument = useCallback(() => {
    setSourceText("");
    setFileName("");
    setDocSummary("");
    setOriginalDocxBytes(null);
    setOriginalXlsxBytes(null);
    setIsEditing(true);
    setIsEditingClassify(true);
    setInfoEditedSinceScan(false);
    setEditingSlice(null);
    setHasAnalyzed(false);
    setTextEditedSinceScan(false);
    setReport(null);
    setInfoReport(null);
    setRawSemanticFindings([]);
    setDismissedFindings(new Set());
    setMatches([]);
    setIgnoredKeys(new Set());
    setLastDurationMs(null);
  }, []);

  // Klistra in från urklipp
  const handlePasteClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (!text.trim()) return;
      setSourceText(normalizeInputText(text));
      setFileName("Urklipp (inklistrad)");
      setDocSummary("Inklistrad text från urklipp");
      setOriginalDocxBytes(null);
      setOriginalXlsxBytes(null);
      setIsEditing(true);
      setIsEditingClassify(true);
      setInfoEditedSinceScan(false);
      setEditingSlice(null);
      setHasAnalyzed(false);
      setTextEditedSinceScan(false);
      setReport(null);
      setInfoReport(null);
      setRawSemanticFindings([]);
      setDismissedFindings(new Set());
      setMatches([]);
      setIgnoredKeys(new Set());
      setLastDurationMs(null);
    } catch (e) {
      alert("Kunde inte läsa från urklipp: " + String(e));
    }
  };

  // Importera fil (Word .docx/.odt, Excel .xlsx/.xls/.ods, PDF .pdf, eller textfil) lokalt via Rust
  const handleImportFile = useCallback(async (file: File) => {
    setImportingDoc(true);
    try {
      const buf = await file.arrayBuffer();
      const byteArr = Array.from(new Uint8Array(buf));
      const imported = await invoke<ImportedDocument>("import_document_bytes", {
        fileName: file.name,
        bytes: byteArr,
      });
      setSourceText(normalizeInputText(imported.text));
      setFileName(imported.file_name);
      setDocSummary(imported.summary);
      setOriginalDocxBytes(imported.can_export_docx ? byteArr : null);
      setOriginalXlsxBytes(imported.can_export_xlsx ? byteArr : null);
      setIsEditing(true);
      setIsEditingClassify(true);
      setInfoEditedSinceScan(false);
      setEditingSlice(null);
      setHasAnalyzed(false);
      setTextEditedSinceScan(false);
      setReport(null);
      setInfoReport(null);
      setRawSemanticFindings([]);
      setDismissedFindings(new Set());
      setMatches([]);
      setIgnoredKeys(new Set());
      setLastDurationMs(null);
    } catch (e) {
      alert("Kunde inte läsa dokumentet: " + String(e));
    } finally {
      setImportingDoc(false);
    }
  }, []);

  // Klassificera dokument enligt signalsystemet (Grön / Gul / Röd) utan att maskera texten
  const handleClassifyInformation = async () => {
    if (!sourceText.trim() || infoClassifying) return;
    setInfoClassifying(true);
    setNerProgress(null);
    setGemmaScanProgress(null);
    try {
      const res = await invoke<InfoClassificationReport>(
        "classify_information_document",
        { text: sourceText }
      );
      setInfoReport(res);
      setInfoEditedSinceScan(false);
      setIsEditingClassify(false);
    } catch (err) {
      alert("Fel vid informationsklassning: " + String(err));
    } finally {
      setInfoClassifying(false);
      setNerProgress(null);
      setGemmaScanProgress(null);
    }
  };

  // Analysera text (Steg 1: Regex + BERT NER, Steg 2: EmbeddingGemma 2)
  const handleAnalyze = async () => {
    if (!sourceText.trim() || analyzing || classifying) return;
    setAnalyzing(true);
    setNerProgress(null);
    setEditingSlice(null);
    const t0 = performance.now();
    try {
      const raw = await invoke<Match[]>("analyze_text", {
        text: sourceText,
        entities: selectedEntities,
        threshold,
      });
      const sorted = fixByteOffsets(sourceText, raw).sort(
        (a, b) => a.start - b.start
      );
      const emptyIgnored = new Set<string>();
      const emptyDismissed = new Set<number>();
      setMatches(sorted);
      setIgnoredKeys(emptyIgnored);
      setDismissedFindings(emptyDismissed);
      setTextEditedSinceScan(false);
      setLastDurationMs(performance.now() - t0);
      setIsEditing(false);
      setHasAnalyzed(true);
      setAnalyzing(false);
      setNerProgress(null);

      // Steg 2: Kör semantisk kontextanalys med EmbeddingGemma 2
      await runSemanticClassification(
        sourceText,
        sorted,
        emptyIgnored,
        emptyDismissed,
        false
      );
    } catch (err) {
      alert("Fel vid analys: " + String(err));
      setAnalyzing(false);
      setNerProgress(null);
    }
  };

  // Klicka på entitet i granskningsvyn för att toggla avmaskering
  const toggleIgnored = useCallback((key: string) => {
    setIgnoredKeys((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }, []);

  // Lägg till manuellt ord
  const handleAddManual = useCallback(() => {
    const word = manualWord.trim();
    if (!word || !sourceText) return;

    const newMatches: Match[] = [];
    let searchFrom = 0;
    while (true) {
      const idx = sourceText.indexOf(word, searchFrom);
      if (idx === -1) break;
      newMatches.push({
        text: word,
        label: manualLabel,
        start: idx,
        end: idx + word.length,
        source: "manuell",
      });
      searchFrom = idx + word.length;
    }

    if (newMatches.length === 0) {
      alert(`"${word}" hittades inte i källtexten.`);
      return;
    }

    setMatches((prev) => {
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
  }, [manualWord, manualLabel, sourceText]);

  // Kopiera maskerad text med Modern Web Guidance copy-button standard
  const handleCopyMasked = async () => {
    if (!maskedText) return;
    try {
      await navigator.clipboard.writeText(maskedText);
      reportCopy(setCopyStatusMasked, copyTimerMaskedRef, true);
    } catch {
      reportCopy(setCopyStatusMasked, copyTimerMaskedRef, false);
    }
  };

  // Kopiera avmaskerat svar som både formaterad HTML (för Word/Outlook/Teams) och ren text
  const handleCopyUnmasked = async () => {
    if (!unmaskedResult.text) return;
    try {
      if (formattedUnmaskView && typeof ClipboardItem !== "undefined" && navigator.clipboard.write) {
        const html = markdownToCleanHtml(unmaskedResult.text);
        const plain = markdownToCleanPlainText(unmaskedResult.text);
        const item = new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([plain], { type: "text/plain" }),
        });
        await navigator.clipboard.write([item]);
      } else {
        await navigator.clipboard.writeText(unmaskedResult.text);
      }
      reportCopy(setCopyStatusUnmasked, copyTimerUnmaskedRef, true);
    } catch {
      try {
        await navigator.clipboard.writeText(
          formattedUnmaskView
            ? markdownToCleanPlainText(unmaskedResult.text)
            : unmaskedResult.text
        );
        reportCopy(setCopyStatusUnmasked, copyTimerUnmaskedRef, true);
      } catch {
        reportCopy(setCopyStatusUnmasked, copyTimerUnmaskedRef, false);
      }
    }
  };

  const reportCopy = (
    setter: React.Dispatch<React.SetStateAction<string>>,
    timerRef: React.MutableRefObject<number | undefined>,
    ok: boolean
  ) => {
    window.clearTimeout(timerRef.current);
    let msg = ok ? "Kopierad!" : "Kopiering misslyckades";
    if (msg === lastCopyMsgRef.current) msg += "\u00A0";
    lastCopyMsgRef.current = msg;
    setter("");
    requestAnimationFrame(() => {
      setter(msg);
    });
    timerRef.current = window.setTimeout(() => {
      setter("");
    }, 2200);
  };

  // Spara maskerad fil (.txt)
  const handleSaveMaskedFile = async () => {
    const defaultName = fileName
      ? fileName.replace(/\.[^/.]+$/, "") + "_maskerad.txt"
      : "maskerad_text.txt";
    const path = await save({
      defaultPath: defaultName,
      filters: [{ name: "Text", extensions: ["txt"] }],
    });
    if (path) {
      await invoke("write_file", { path, content: maskedText });
    }
  };

  // Spara maskerat Word-dokument (.docx) med bibehållen formatering
  const handleSaveMaskedDocx = async () => {
    if (!originalDocxBytes) return;
    const defaultName = fileName
      ? fileName.replace(/\.[^/.]+$/, "") + "_maskerad.docx"
      : "maskerat_dokument.docx";
    const path = await save({
      defaultPath: defaultName,
      filters: [{ name: "Word-dokument", extensions: ["docx"] }],
    });
    if (!path) return;

    const replacements: ReplacementPair[] = [
      ...activeTranslationTable.values(),
    ].map((entry) => ({
      original: entry.originalText,
      replacement: `[${entry.maskedLabel.toUpperCase()}]`,
    }));

    try {
      await invoke("export_masked_docx", {
        path,
        originalBytes: originalDocxBytes,
        replacements,
      });
    } catch (e) {
      alert("Kunde inte exportera maskerad .docx: " + String(e));
    }
  };

  // Spara maskerad Excel-arbetsbok (.xlsx) med bibehållen formatering
  const handleSaveMaskedXlsx = async () => {
    if (!originalXlsxBytes) return;
    const defaultName = fileName
      ? fileName.replace(/\.[^/.]+$/, "") + "_maskerad.xlsx"
      : "maskerad_arbetsbok.xlsx";
    const path = await save({
      defaultPath: defaultName,
      filters: [{ name: "Excel-arbetsbok", extensions: ["xlsx"] }],
    });
    if (!path) return;

    const replacements: ReplacementPair[] = [
      ...activeTranslationTable.values(),
    ].map((entry) => ({
      original: entry.originalText,
      replacement: `[${entry.maskedLabel.toUpperCase()}]`,
    }));

    try {
      await invoke("export_masked_xlsx", {
        path,
        originalBytes: originalXlsxBytes,
        replacements,
      });
    } catch (e) {
      alert("Kunde inte exportera maskerad .xlsx: " + String(e));
    }
  };

  // Avmaskera ett Word- (.docx) eller Excel-dokument (.xlsx) tillbaka till originalvärden
  const handleUnmaskDocumentFile = async (file: File) => {
    if (activeTranslationTable.size === 0) {
      alert("Det finns inga aktiva mappningar i översättningstabellen att avmaskera med.");
      return;
    }
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    const isDocx = ext === "docx";
    const isXlsx = ext === "xlsx" || ext === "xlsm";
    if (!isDocx && !isXlsx) {
      alert("Välj en .docx- eller .xlsx-fil för formateringsbevarande avmaskering.");
      return;
    }

    const defaultName =
      file.name.replace(/\.[^/.]+$/, "") +
      (isDocx ? "_avmaskerad.docx" : "_avmaskerad.xlsx");
    const path = await save({
      defaultPath: defaultName,
      filters: [
        isDocx
          ? { name: "Word-dokument", extensions: ["docx"] }
          : { name: "Excel-arbetsbok", extensions: ["xlsx"] },
      ],
    });
    if (!path) return;

    // Bygg omvända ersättningspar (inkl. versaler, gemener och Titel-case för platshållare)
    const replacements: ReplacementPair[] = [];
    for (const entry of activeTranslationTable.values()) {
      const upper = `[${entry.maskedLabel.toUpperCase()}]`;
      const exact = `[${entry.maskedLabel}]`;
      const lower = `[${entry.maskedLabel.toLowerCase()}]`;
      replacements.push({ original: upper, replacement: entry.originalText });
      if (exact !== upper) {
        replacements.push({ original: exact, replacement: entry.originalText });
      }
      if (lower !== upper && lower !== exact) {
        replacements.push({ original: lower, replacement: entry.originalText });
      }
    }

    try {
      const buf = await file.arrayBuffer();
      const originalBytes = Array.from(new Uint8Array(buf));
      await invoke(isDocx ? "export_masked_docx" : "export_masked_xlsx", {
        path,
        originalBytes,
        replacements,
      });
    } catch (e) {
      alert("Kunde inte avmaskera dokumentet: " + String(e));
    }
  };

  // Konvertera byte-offsets i flaggade stycken till UTF-16-positioner i sourceText
  const charFindings = useMemo(() => {
    if (!report?.findings || !sourceText) return [];
    return fixFindingOffsets(sourceText, report.findings)
      .filter((f) => f.end > f.start)
      .sort((a, b) => a.start - b.start);
  }, [report?.findings, sourceText]);

  // Hoppa till och markera ett flaggat avsnitt i granskningsvyn
  const handleJumpToParagraph = useCallback((paraIndex: number) => {
    setActiveTab("mask");
    setIsEditing(false);
    setShowParaHighlights(true);
    setFocusedParaIndex(paraIndex);
    setTimeout(() => {
      const el = document.getElementById(`finding-para-${paraIndex}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 50);
    setTimeout(() => {
      setFocusedParaIndex((prev) => (prev === paraIndex ? null : prev));
    }, 2600);
  }, []);

  // Sparar en inline-redigerad delsträng [start..end] utan att tappa övriga maskeringar
  const commitInlineSliceEdit = useCallback(() => {
    if (!editingSlice) return;
    const nextFull =
      sourceText.slice(0, editingSlice.start) +
      editingSlice.draft +
      sourceText.slice(editingSlice.end);
    handleSourceTextEdit(nextFull);
    setEditingSlice(null);
  }, [editingSlice, sourceText, handleSourceTextEdit]);

  // Hjälpfunktion för att rendera inline-text + <mark>-chips inom [rangeStart, rangeEnd]
  const renderInlineRangeWithMatches = useCallback(
    (rangeStart: number, rangeEnd: number, keyPrefix: string) => {
      if (rangeEnd <= rangeStart) return null;
      const sliceMatches = matches.filter(
        (m) => m.end > rangeStart && m.start < rangeEnd
      );
      if (sliceMatches.length === 0) {
        return (
          <span key={`${keyPrefix}-plain`}>
            {renderInlineFormattedText(
              sourceText.slice(rangeStart, rangeEnd),
              `${keyPrefix}-fmt`
            )}
          </span>
        );
      }

      const nodes: React.ReactNode[] = [];
      let cursor = rangeStart;

      sliceMatches.forEach((m, idx) => {
        const mStart = Math.max(m.start, rangeStart);
        const mEnd = Math.min(m.end, rangeEnd);

        if (mStart > cursor) {
          nodes.push(
            <span key={`${keyPrefix}-t-${idx}`}>
              {renderInlineFormattedText(
                sourceText.slice(cursor, mStart),
                `${keyPrefix}-tf-${idx}`
              )}
            </span>
          );
        }

        const key = matchKey(m);
        const ignored = ignoredKeys.has(key);
        const masked =
          activeTranslationTable.get(m.text)?.maskedLabel ?? m.label;

        nodes.push(
          <mark
            key={`${keyPrefix}-m-${idx}`}
            role="button"
            tabIndex={0}
            aria-pressed={!ignored}
            className={`highlight-chip${ignored ? " ignored" : ""}`}
            style={{
              backgroundColor: ignored
                ? "transparent"
                : COLORS[m.label] || "#475569",
            }}
            onClick={(e) => {
              e.stopPropagation();
              toggleIgnored(key);
            }}
            onDoubleClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                toggleIgnored(key);
              } else if (e.key === " ") {
                e.preventDefault();
              }
            }}
            onKeyUp={(e) => {
              if (e.key === " ") {
                e.preventDefault();
                toggleIgnored(key);
              }
            }}
            title={
              ignored
                ? "Klicka eller tryck Retur för att maskera igen"
                : "Klicka eller tryck Retur för att undanta/avmaskera"
            }
          >
            {sourceText.slice(mStart, mEnd)}
            {!ignored && <span className="entity-tag">[{masked}]</span>}
          </mark>
        );
        cursor = mEnd;
      });

      if (cursor < rangeEnd) {
        nodes.push(
          <span key={`${keyPrefix}-tail`}>
            {renderInlineFormattedText(
              sourceText.slice(cursor, rangeEnd),
              `${keyPrefix}-tailf`
            )}
          </span>
        );
      }

      return nodes;
    },
    [sourceText, matches, ignoredKeys, activeTranslationTable, toggleIgnored]
  );

  // Renderar en enskild rad [lineStart, lineEnd] med dokumenttypografi (rubrik, punktlista, brödtext)
  const renderFormattedDocLine = useCallback(
    (lineStart: number, lineEnd: number, keyPrefix: string) => {
      const rawLine = sourceText.slice(lineStart, lineEnd);
      if (!rawLine.trim()) {
        return <div key={`${keyPrefix}-empty`} className="doc-empty-line" />;
      }

      // Markdown-rubrik (# .. ####) om ingen träff börjar inne i själva #-prefixet
      const headingMatch = rawLine.match(/^(\s*#{1,4}\s+)/);
      if (headingMatch) {
        const prefixLen = headingMatch[1].length;
        const level = headingMatch[1].trim().length;
        const hasMatchInPrefix = matches.some(
          (m) => m.start < lineStart + prefixLen && m.end > lineStart
        );
        if (!hasMatchInPrefix) {
          return (
            <div
              key={`${keyPrefix}-h`}
              className={`doc-heading doc-heading-${Math.min(level, 3)}`}
            >
              {renderInlineRangeWithMatches(
                lineStart + prefixLen,
                lineEnd,
                `${keyPrefix}-hc`
              )}
            </div>
          );
        }
      }

      // Fristående fetstilt rubrikrad: **RUBRIK**
      const trimmed = rawLine.trim();
      if (
        trimmed.startsWith("**") &&
        trimmed.endsWith("**") &&
        trimmed.length > 4 &&
        !trimmed.slice(2, -2).includes("**")
      ) {
        const leadWs = rawLine.length - rawLine.trimStart().length;
        const innerStart = lineStart + leadWs + 2;
        const innerEnd = lineStart + leadWs + trimmed.length - 2;
        const hasMatchAtEdges = matches.some(
          (m) =>
            (m.start < innerStart && m.end > lineStart) ||
            (m.start < lineEnd && m.end > innerEnd)
        );
        if (!hasMatchAtEdges) {
          return (
            <div key={`${keyPrefix}-bh`} className="doc-heading doc-heading-2">
              {renderInlineRangeWithMatches(
                innerStart,
                innerEnd,
                `${keyPrefix}-bhc`
              )}
            </div>
          );
        }
      }

      // Punktlista (* / - / •)
      const bulletMatch = rawLine.match(/^(\s*[-*•]\s+)/);
      if (bulletMatch) {
        const prefixLen = bulletMatch[1].length;
        const hasMatchInPrefix = matches.some(
          (m) => m.start < lineStart + prefixLen && m.end > lineStart
        );
        if (!hasMatchInPrefix) {
          return (
            <div key={`${keyPrefix}-li`} className="doc-bullet-item">
              <span className="doc-bullet-dot" aria-hidden="true">
                •
              </span>
              <span className="doc-bullet-content">
                {renderInlineRangeWithMatches(
                  lineStart + prefixLen,
                  lineEnd,
                  `${keyPrefix}-lic`
                )}
              </span>
            </div>
          );
        }
      }

      return (
        <div key={`${keyPrefix}-p`} className="doc-line">
          {renderInlineRangeWithMatches(lineStart, lineEnd, `${keyPrefix}-lc`)}
        </div>
      );
    },
    [sourceText, matches, renderInlineRangeWithMatches]
  );

  // Renderar ett intervall [sliceStart, sliceEnd] uppdelat i dokumentstycken med direktredigering på plats
  const renderSliceWithMatches = useCallback(
    (
      sliceStart: number,
      sliceEnd: number,
      keyPrefix: string,
      insideFlaggedCard = false
    ) => {
      if (sliceEnd <= sliceStart) return null;

      if (
        insideFlaggedCard &&
        editingSlice &&
        editingSlice.start === sliceStart &&
        editingSlice.end === sliceEnd
      ) {
        return (
          <div className="inline-slice-editor" key={`${keyPrefix}-iedit`}>
            <textarea
              className="inline-slice-textarea"
              value={editingSlice.draft}
              rows={Math.max(3, editingSlice.draft.split("\n").length + 1)}
              autoFocus
              onChange={(e) =>
                setEditingSlice((prev) =>
                  prev ? { ...prev, draft: e.target.value } : null
                )
              }
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  commitInlineSliceEdit();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  setEditingSlice(null);
                }
              }}
            />
            <div className="inline-slice-actions">
              <span className="inline-slice-hint">
                Övriga maskeringar i dokumentet bevaras automatiskt (Ctrl+Retur sparar)
              </span>
              <button
                type="button"
                className="tb-btn-subtle"
                onClick={() => setEditingSlice(null)}
              >
                Avbryt
              </button>
              <button
                type="button"
                className="tb-btn-primary inline-save-btn"
                onClick={commitInlineSliceEdit}
              >
                <Check size={14} aria-hidden="true" />
                <span>Spara ändring</span>
              </button>
            </div>
          </div>
        );
      }

      // Dela upp i stycken längs tomrader (\n\n+) med exakta teckenpositioner
      const blocks: Array<{ start: number; end: number }> = [];
      let cursor = sliceStart;
      while (cursor < sliceEnd) {
        while (
          cursor < sliceEnd &&
          (sourceText[cursor] === "\n" || sourceText[cursor] === "\r")
        ) {
          cursor++;
        }
        if (cursor >= sliceEnd) break;
        let bEnd = cursor;
        while (bEnd < sliceEnd) {
          if (
            sourceText[bEnd] === "\n" &&
            bEnd + 1 < sliceEnd &&
            sourceText[bEnd + 1] === "\n"
          ) {
            break;
          }
          bEnd++;
        }
        blocks.push({ start: cursor, end: bEnd });
        cursor = bEnd;
      }

      return blocks.map((blk, bIdx) => {
        const isEditingThisBlock =
          editingSlice !== null &&
          editingSlice.start === blk.start &&
          editingSlice.end === blk.end;

        if (isEditingThisBlock && editingSlice) {
          return (
            <div
              key={`${keyPrefix}-blk-edit-${bIdx}`}
              className="inline-slice-editor"
              onBlur={(e) => {
                if (!document.hasFocus()) return;
                if (e.currentTarget.contains(e.relatedTarget as Node | null)) {
                  return;
                }
                commitInlineSliceEdit();
              }}
            >
              <textarea
                className="inline-slice-textarea"
                value={editingSlice.draft}
                rows={Math.max(2, editingSlice.draft.split("\n").length + 1)}
                autoFocus
                onChange={(e) =>
                  setEditingSlice((prev) =>
                    prev ? { ...prev, draft: e.target.value } : null
                  )
                }
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                    e.preventDefault();
                    commitInlineSliceEdit();
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    setEditingSlice(null);
                  }
                }}
              />
              <div className="inline-slice-actions">
                <span className="inline-slice-hint">
                  Klicka utanför eller Ctrl+Retur sparar · Esc avbryter
                </span>
                <button
                  type="button"
                  className="tb-btn-subtle"
                  onClick={() => setEditingSlice(null)}
                >
                  Avbryt
                </button>
                <button
                  type="button"
                  className="tb-btn-primary inline-save-btn"
                  onClick={commitInlineSliceEdit}
                >
                  <Check size={14} aria-hidden="true" />
                  <span>Klar</span>
                </button>
              </div>
            </div>
          );
        }

        // Dela upp blocket i rader med exakta teckenpositioner
        const lines: React.ReactNode[] = [];
        let lStart = blk.start;
        let lIdx = 0;
        while (lStart <= blk.end) {
          let lEnd = sourceText.indexOf("\n", lStart);
          if (lEnd === -1 || lEnd > blk.end) {
            lEnd = blk.end;
          }
          lines.push(
            renderFormattedDocLine(
              lStart,
              lEnd,
              `${keyPrefix}-b${bIdx}-l${lIdx++}`
            )
          );
          if (lEnd === blk.end) break;
          lStart = lEnd + 1;
        }

        return (
          <div
            key={`${keyPrefix}-blk-${bIdx}`}
            className={`doc-para ${insideFlaggedCard ? "doc-para-in-card" : "doc-para-editable"}`}
            onClick={() => {
              const sel = window.getSelection();
              if (sel && !sel.isCollapsed) return;
              setEditingSlice({
                start: blk.start,
                end: blk.end,
                draft: sourceText.slice(blk.start, blk.end),
              });
            }}
            title="Klicka i stycket för att redigera på plats (klicka utanför för att spara)"
          >
            {lines}
          </div>
        );
      });
    },
    [
      sourceText,
      editingSlice,
      commitInlineSliceEdit,
      renderFormattedDocLine,
    ]
  );

  // Highlight-rendering i läsläge (kombinerar Lager 3-styckemarkeringar och Lager 1/2-ordmarkeringar)
  const renderedReviewText = useMemo(() => {
    if (!sourceText) return null;
    if (!showParaHighlights || charFindings.length === 0) {
      return renderSliceWithMatches(0, sourceText.length, "full", false);
    }

    const elements: React.ReactNode[] = [];
    let paraCursor = 0;

    charFindings.forEach((f, idx) => {
      const fStart = Math.max(f.start, paraCursor);
      const fEnd = Math.min(f.end, sourceText.length);
      if (fEnd <= fStart) return;

      if (fStart > paraCursor) {
        elements.push(
          renderSliceWithMatches(paraCursor, fStart, `before-p-${idx}`, false)
        );
      }

      const isFocused = focusedParaIndex === f.paragraph_index;
      const isEditingWholeFlagged =
        editingSlice !== null &&
        editingSlice.start === fStart &&
        editingSlice.end === fEnd;

      elements.push(
        <div
          key={`para-${f.paragraph_index}`}
          id={`finding-para-${f.paragraph_index}`}
          className={`flagged-para-block severity-${f.severity}${
            isFocused ? " focused" : ""
          }`}
        >
          <div className="flagged-para-header">
            <div className="flagged-para-top-row">
              <span className="flagged-para-title">
                <span className={`sev-pill ${f.severity}`}>
                  {f.severity === "high" ? (
                    <XCircle size={12} aria-hidden="true" />
                  ) : (
                    <AlertTriangle size={12} aria-hidden="true" />
                  )}
                  <span>
                    {f.severity === "high" ? "Hög känslighet" : "Gränsfall"}
                  </span>
                </span>
                <span>
                  Avsnitt {f.paragraph_index} · {f.category}
                </span>
              </span>
              <span className="flagged-para-meta">
                <button
                  type="button"
                  className="flagged-action-btn"
                  onClick={() =>
                    setEditingSlice({
                      start: fStart,
                      end: fEnd,
                      draft: sourceText.slice(fStart, fEnd),
                    })
                  }
                  title="Redigera detta avsnitt direkt på plats"
                >
                  <Pencil size={12} aria-hidden="true" />
                  <span>
                    Redigera
                    <span className="visually-hidden">
                      {" "}
                      avsnitt {f.paragraph_index}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  className="flagged-action-btn approve"
                  onClick={() => toggleDismissFinding(f.paragraph_index)}
                  title="Markera att du har granskat detta avsnitt och bedömer det säkert att dela"
                >
                  <Check size={12} aria-hidden="true" />
                  <span>
                    Klarmarkera
                    <span className="visually-hidden">
                      {" "}
                      avsnitt {f.paragraph_index}
                    </span>
                  </span>
                </button>
              </span>
            </div>
            <p className="flagged-para-explain">
              <strong>
                {f.severity === "high"
                  ? `Röd (${Math.round(f.confidence * 100)}% träffstyrka): `
                  : `Gul (${Math.round(f.confidence * 100)}% träffstyrka): `}
              </strong>
              {f.reason}{" "}
              {f.severity === "high"
                ? "Stark koppling — kontrollera att inga känsliga sakdetaljer röjs, eller klarmarkera efter granskning."
                : "Svagare gränsfallsträff — ögna igenom och klarmarkera om innehållet är ofarligt."}
            </p>
          </div>
          <div className="flagged-para-body">
            {isEditingWholeFlagged && editingSlice ? (
              <div
                className="inline-slice-editor"
                onBlur={(e) => {
                  if (!document.hasFocus()) return;
                  if (e.currentTarget.contains(e.relatedTarget as Node | null)) {
                    return;
                  }
                  commitInlineSliceEdit();
                }}
              >
                <textarea
                  className="inline-slice-textarea"
                  value={editingSlice.draft}
                  rows={Math.max(2, editingSlice.draft.split("\n").length + 1)}
                  autoFocus
                  onChange={(e) =>
                    setEditingSlice((prev) =>
                      prev ? { ...prev, draft: e.target.value } : null
                    )
                  }
                  onKeyDown={(e) => {
                    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                      e.preventDefault();
                      commitInlineSliceEdit();
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      setEditingSlice(null);
                    }
                  }}
                />
                <div className="inline-slice-actions">
                  <span className="inline-slice-hint">
                    Klicka utanför eller Ctrl+Retur sparar · Esc avbryter
                  </span>
                  <button
                    type="button"
                    className="tb-btn-subtle"
                    onClick={() => setEditingSlice(null)}
                  >
                    Avbryt
                  </button>
                  <button
                    type="button"
                    className="tb-btn-primary inline-save-btn"
                    onClick={commitInlineSliceEdit}
                  >
                    <Check size={14} aria-hidden="true" />
                    <span>Klar</span>
                  </button>
                </div>
              </div>
            ) : (
              renderSliceWithMatches(fStart, fEnd, `in-p-${idx}`, true)
            )}
          </div>
        </div>
      );

      paraCursor = fEnd;
    });

    if (paraCursor < sourceText.length) {
      elements.push(
        renderSliceWithMatches(
          paraCursor,
          sourceText.length,
          "after-all",
          false
        )
      );
    }

    return elements;
  }, [
    sourceText,
    showParaHighlights,
    charFindings,
    focusedParaIndex,
    editingSlice,
    commitInlineSliceEdit,
    renderSliceWithMatches,
    toggleDismissFinding,
  ]);

  // ── Läsvy för Informationsklassning (helt omaskerad originaltext + pedagogiska avsnittsmarkeringar) ──
  const charInfoFindings = useMemo(
    () =>
      infoReport
        ? fixInfoSectionOffsets(sourceText, infoReport.sectionFindings)
            .filter((f) => f.end > f.start)
            .sort((a, b) => a.start - b.start)
        : [],
    [infoReport, sourceText]
  );

  const renderPlainDocLine = useCallback(
    (lineStart: number, lineEnd: number, keyPrefix: string) => {
      const rawLine = sourceText.slice(lineStart, lineEnd);
      if (!rawLine.trim()) {
        return <div key={`${keyPrefix}-empty`} className="doc-empty-line" />;
      }

      const headingMatch = rawLine.match(/^(\s*#{1,4}\s+)/);
      if (headingMatch) {
        const prefixLen = headingMatch[1].length;
        const level = headingMatch[1].trim().length;
        return (
          <div
            key={`${keyPrefix}-h`}
            className={`doc-heading doc-heading-${Math.min(level, 3)}`}
          >
            {renderInlineFormattedText(
              sourceText.slice(lineStart + prefixLen, lineEnd),
              `${keyPrefix}-hc`
            )}
          </div>
        );
      }

      const trimmed = rawLine.trim();
      if (
        trimmed.startsWith("**") &&
        trimmed.endsWith("**") &&
        trimmed.length > 4 &&
        !trimmed.slice(2, -2).includes("**")
      ) {
        return (
          <div key={`${keyPrefix}-bh`} className="doc-heading doc-heading-2">
            {renderInlineFormattedText(trimmed.slice(2, -2), `${keyPrefix}-bhc`)}
          </div>
        );
      }

      const bulletMatch = rawLine.match(/^(\s*[-*•]\s+)/);
      if (bulletMatch) {
        const prefixLen = bulletMatch[1].length;
        return (
          <div key={`${keyPrefix}-li`} className="doc-bullet-item">
            <span className="doc-bullet-dot" aria-hidden="true">
              •
            </span>
            <span className="doc-bullet-content">
              {renderInlineFormattedText(
                sourceText.slice(lineStart + prefixLen, lineEnd),
                `${keyPrefix}-lic`
              )}
            </span>
          </div>
        );
      }

      return (
        <div key={`${keyPrefix}-p`} className="doc-line">
          {renderInlineFormattedText(
            sourceText.slice(lineStart, lineEnd),
            `${keyPrefix}-lc`
          )}
        </div>
      );
    },
    [sourceText]
  );

  const renderPlainDocSlice = useCallback(
    (sliceStart: number, sliceEnd: number, keyPrefix: string) => {
      if (sliceEnd <= sliceStart) return null;

      const blocks: Array<{ start: number; end: number }> = [];
      let cursor = sliceStart;
      while (cursor < sliceEnd) {
        while (
          cursor < sliceEnd &&
          (sourceText[cursor] === "\n" || sourceText[cursor] === "\r")
        ) {
          cursor++;
        }
        if (cursor >= sliceEnd) break;
        let bEnd = cursor;
        while (bEnd < sliceEnd) {
          if (
            sourceText[bEnd] === "\n" &&
            bEnd + 1 < sliceEnd &&
            sourceText[bEnd + 1] === "\n"
          ) {
            break;
          }
          bEnd++;
        }
        blocks.push({ start: cursor, end: bEnd });
        cursor = bEnd;
      }

      return blocks.map((blk, bIdx) => {
        const lines: React.ReactNode[] = [];
        let lStart = blk.start;
        let lIdx = 0;
        while (lStart <= blk.end) {
          let lEnd = sourceText.indexOf("\n", lStart);
          if (lEnd === -1 || lEnd > blk.end) {
            lEnd = blk.end;
          }
          lines.push(
            renderPlainDocLine(
              lStart,
              lEnd,
              `${keyPrefix}-b${bIdx}-l${lIdx++}`
            )
          );
          if (lEnd === blk.end) break;
          lStart = lEnd + 1;
        }
        return (
          <div key={`${keyPrefix}-blk-${bIdx}`} className="doc-para">
            {lines}
          </div>
        );
      });
    },
    [sourceText, renderPlainDocLine]
  );

  const renderedClassifyDocument = useMemo(() => {
    if (!sourceText) return null;
    if (
      !showInfoSectionHighlights ||
      infoEditedSinceScan ||
      charInfoFindings.length === 0
    ) {
      return renderPlainDocSlice(0, sourceText.length, "plain-full");
    }

    const elements: React.ReactNode[] = [];
    let cursor = 0;

    charInfoFindings.forEach((f, idx) => {
      const fStart = Math.max(f.start, cursor);
      const fEnd = Math.min(f.end, sourceText.length);
      if (fEnd <= fStart) return;

      if (fStart > cursor) {
        elements.push(
          renderPlainDocSlice(cursor, fStart, `info-before-${idx}`)
        );
      }

      const isRed = f.level === "red";
      elements.push(
        <div
          key={`info-para-${f.paragraphIndex}-${idx}`}
          id={`info-para-${f.paragraphIndex}`}
          className={`flagged-para-block ${
            isRed ? "severity-high" : "severity-medium"
          }`}
        >
          <div className="flagged-para-header">
            <div className="flagged-para-top-row">
              <span className="flagged-para-title">
                <span className={`sev-pill ${isRed ? "high" : "medium"}`}>
                  {isRed ? (
                    <XCircle size={12} aria-hidden="true" />
                  ) : (
                    <AlertTriangle size={12} aria-hidden="true" />
                  )}
                  <span>
                    {isRed
                      ? "Röd · Kräver särskilt skydd"
                      : "Gul · Var försiktig (Internt)"}
                  </span>
                </span>
                <span>
                  Avsnitt {f.paragraphIndex} · {f.category}
                </span>
              </span>
              <span className="flagged-para-meta">
                <span className="flagged-para-conf">
                  {Math.round(f.confidence * 100)}% träffstyrka
                </span>
              </span>
            </div>
            <p className="flagged-para-explain">{f.reason}</p>
          </div>
          <div className="flagged-para-body">
            {renderPlainDocSlice(fStart, fEnd, `info-in-${idx}`)}
          </div>
        </div>
      );

      cursor = fEnd;
    });

    if (cursor < sourceText.length) {
      elements.push(
        renderPlainDocSlice(cursor, sourceText.length, "info-after-all")
      );
    }

    return elements;
  }, [
    sourceText,
    showInfoSectionHighlights,
    infoEditedSinceScan,
    charInfoFindings,
    renderPlainDocSlice,
  ]);

  // Traffic light helpers
  const verdictBadge = (verdict: SafetyVerdict | undefined) => {
    switch (verdict) {
      case "safe":
        return {
          icon: <CheckCircle2 size={16} />,
          text: "Redo för AI",
          className: "badge-safe",
        };
      case "mask_required":
        return {
          icon: <AlertTriangle size={16} />,
          text: "Maskering krävs",
          className: "badge-warn",
        };
      case "caution_after_mask":
        return {
          icon: <AlertTriangle size={16} />,
          text: "Granska före delning",
          className: "badge-caution",
        };
      case "do_not_share":
        return {
          icon: <XCircle size={16} />,
          text: "Dela ej i öppen AI",
          className: "badge-danger",
        };
      default:
        return {
          icon: <Shield size={16} />,
          text: "Ej analyserad",
          className: "badge-neutral",
        };
    }
  };

  const beforeBadge = verdictBadge(report?.verdict_before_mask);
  const afterBadge = verdictBadge(report?.verdict_after_mask);

  const aiBusy =
    analyzing ||
    classifying ||
    infoClassifying ||
    gemmaLoading ||
    setupBusy ||
    (!nerReady && !setupError);
  const aiStatusText = showFirstTimeSetup
    ? setupError
      ? "Väntar på modellinstallation"
      : "Förbereder lokal AI…"
    : infoClassifying
    ? gemmaScanProgress && gemmaScanProgress.total > 0
      ? `Klassificerar (${gemmaScanProgress.chunk}/${gemmaScanProgress.total})…`
      : "Klassificerar information…"
    : analyzing
    ? nerProgress && nerProgress.total > 0
      ? `Skannar (${nerProgress.chunk}/${nerProgress.total})…`
      : "Skannar text…"
    : classifying
    ? gemmaScanProgress && gemmaScanProgress.total > 0
      ? `Analyserar kontext (${gemmaScanProgress.chunk}/${gemmaScanProgress.total})…`
      : "Analyserar kontext…"
    : gemmaLoading
    ? gemmaDownloadProgress && gemmaDownloadProgress.total > 0
      ? `Laddar ner modell (${Math.round(
          (gemmaDownloadProgress.downloaded / gemmaDownloadProgress.total) * 100
        )}%)…`
      : "Laddar kontextmodell…"
    : !nerReady
    ? "Startar lokal AI…"
    : "Lokal AI redo";

  const nerDoneInSetup = nerInstalled || Boolean(setupProgress?.nerDone);
  const gemmaDoneInSetup = gemmaInstalled || Boolean(setupProgress?.gemmaDone);
  const isDeterminateSetupProgress =
    setupProgress !== null &&
    setupProgress.phase === "downloading" &&
    setupProgress.totalBytes > 0;
  const setupPercent = isDeterminateSetupProgress
    ? Math.min(
        100,
        Math.max(
          1,
          Math.round(
            (setupProgress.downloadedBytes / setupProgress.totalBytes) * 100
          )
        )
      )
    : 0;
  const setupBytesText = isDeterminateSetupProgress
    ? `${(setupProgress.downloadedBytes / 1_000_000).toFixed(0)} av ${(
        setupProgress.totalBytes / 1_000_000
      ).toFixed(0)} MB (${setupPercent} %)`
    : setupProgress?.phase === "loading"
    ? "Laddar in i RAM…"
    : setupProgress?.phase === "copying"
    ? "Kopierar modellfiler…"
    : "";

  const showUnifiedDocHeader =
    Boolean(sourceText) &&
    (hasAnalyzed || matches.length > 0 || charFindings.length > 0);

  const latencyTooltip = [
    lastDurationMs !== null ? `BERT + Regex: ${Math.round(lastDurationMs)} ms` : "",
    report?.semantic_available
      ? `EmbeddingGemma 2: ${report.paragraphs_scanned} avsnitt på ${report.scan_duration_ms} ms`
      : "",
  ]
    .filter(Boolean)
    .join(" · ");

  const highFindingsCount =
    report?.findings.filter((f) => f.severity === "high").length ?? 0;
  const medFindingsCount =
    (report?.findings.length ?? 0) - highFindingsCount;

  return (
    <div className="maskera-root">
      {/* ── 1. Kompakt header (Modern Web Guidance: < 12% vh) ── */}
      <header className="top-header">
        <div className="header-brand">
          <div className="brand-logo" aria-hidden="true">
            {appMode === "classify" ? (
              <FileSearch size={20} />
            ) : (
              <Shield size={20} />
            )}
          </div>
          <div className="brand-titles">
            <h1 className="brand-title">Maskera</h1>
            <span className="brand-tagline">
              {appMode === "classify"
                ? "Informationsklassning (Grön · Gul · Röd)"
                : "Lokal avidentifiering för AI"}
            </span>
          </div>
        </div>

        {modelsChecked && !showFirstTimeSetup && (
          <div
            role="group"
            aria-label="Arbetsläge"
            className="app-mode-switcher"
          >
            <button
              type="button"
              className="mode-switch-btn"
              aria-pressed={appMode === "mask"}
              onClick={() => setAppMode("mask")}
            >
              <Shield size={15} aria-hidden="true" />
              <span>Avidentifiera för AI</span>
            </button>
            <button
              type="button"
              className="mode-switch-btn"
              aria-pressed={appMode === "classify"}
              onClick={() => setAppMode("classify")}
            >
              <FileSearch size={15} aria-hidden="true" />
              <span>Klassificera information</span>
            </button>
          </div>
        )}

        <div className="header-status-pills">
          <span
            className={`status-pill ${
              analyzing || classifying || infoClassifying
                ? "working"
                : gemmaLoading || setupBusy || !nerReady
                ? "loading"
                : "ready"
            }`}
            title={`BERT NER: ${nerReady ? "Aktiv" : nerStatus} · EmbeddingGemma 2: ${
              gemmaReady ? "Aktiv" : gemmaStatus || "Klar vid start"
            }`}
          >
            {aiBusy && <RefreshCw size={12} className="spin" aria-hidden="true" />}
            <span>{aiStatusText}</span>
          </span>

          <button
            type="button"
            className="theme-toggle-btn"
            onClick={handleQuickToggleTheme}
            title={
              renderedDark ? "Byt till ljust tema" : "Byt till mörkt tema"
            }
          >
            {renderedDark ? (
              <>
                <Sun size={14} aria-hidden="true" />
                <span>Ljust</span>
              </>
            ) : (
              <>
                <Moon size={14} aria-hidden="true" />
                <span>Mörkt</span>
              </>
            )}
          </button>
        </div>
      </header>

      {!modelsChecked ? (
        <main className="main-workspace" aria-busy="true" />
      ) : showFirstTimeSetup ? (
        /* ── Alternativ A: Välkomst- & engångsinstallationsvy (visas endast om modeller saknas vid start) ── */
        <main className="setup-workspace">
          <section
            className="setup-card"
            aria-labelledby="setup-heading"
          >
            <div className="setup-card-header">
              <div className="setup-icon-badge" aria-hidden="true">
                <Sparkles size={22} />
              </div>
              <div className="setup-title-group">
                <h2 id="setup-heading" className="setup-title">
                  Förbereder lokal AI — engångsnedladdning
                </h2>
                <p className="setup-subtitle">
                  Maskera hämtar nu språk- och kontextmodellerna (~318 MB totalt)
                  en enda gång till din dator. Därefter körs all avidentifiering
                  och säkerhetskontroll <strong>100 % lokalt och offline</strong>{" "}
                  — ingen text lämnar någonsin din dator.
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
                    Hittar personnamn, organisationer, platser, tider och
                    händelser i svensk löptext (KBLab/Kungliga biblioteket).
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
                    Semantisk säkerhetskontroll som varnar för patient-/vårduppgifter,
                    företagshemligheter, sårbarheter och HR-ärenden.
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
                    onClick={runEnsureAllModels}
                  >
                    <RefreshCw size={15} aria-hidden="true" />
                    <span>Försök igen</span>
                  </button>
                  <button
                    type="button"
                    className="tb-btn"
                    onClick={handlePickLocalModelDir}
                  >
                    <FolderOpen size={15} aria-hidden="true" />
                    <span>Välj lokal modellmapp…</span>
                  </button>
                  <button
                    type="button"
                    className="tb-btn-subtle"
                    onClick={() => setShowFirstTimeSetup(false)}
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
                  onClick={handlePickLocalModelDir}
                >
                  <FolderOpen size={13} aria-hidden="true" />
                  <span>Välj lokal modellmapp…</span>
                </button>
              </div>
            )}
          </section>
        </main>
      ) : (
        <>

      {/* Dold filväljare utanför role="toolbar" så att verktygsraden enbart har riktiga <button> */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.log,.md,.csv,.tsv,.json,.xml,.docx,.odt,.xlsx,.xls,.xlsm,.xlsb,.ods,.pdf"
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) {
            handleImportFile(f);
            e.target.value = "";
          }
        }}
      />

      {appMode === "classify" ? (
        /* ── Alternativ B: Fristående läge för Informationsklassning (Grön / Gul / Röd) utan maskering ── */
        <>
          <div
            role="toolbar"
            focusgroup="toolbar wrap"
            aria-label="Dokument och informationsklassning"
            className="action-toolbar"
          >
            <div role="group" aria-label="Inmatning" className="toolbar-group">
              <button
                type="button"
                className="tb-btn"
                onClick={handlePasteClipboard}
                title="Klistra in text från urklipp"
              >
                <Clipboard size={15} aria-hidden="true" />
                <span>Klistra in</span>
              </button>
              <button
                type="button"
                className="tb-btn"
                onClick={() => fileInputRef.current?.click()}
                title="Öppna Word (.docx/.odt), Excel (.xlsx/.xls/.ods), PDF (.pdf) eller textfil"
              >
                {importingDoc ? (
                  <RefreshCw size={15} className="spin" aria-hidden="true" />
                ) : (
                  <Upload size={15} aria-hidden="true" />
                )}
                <span>
                  {importingDoc ? "Läser dokument…" : "Öppna dokument"}
                </span>
              </button>
              <button
                type="button"
                className="tb-btn"
                aria-disabled={!sourceText ? "true" : undefined}
                onClick={() => {
                  if (sourceText) handleClearDocument();
                }}
                title="Rensa aktuellt dokument och bedömning"
              >
                <Trash2 size={15} aria-hidden="true" />
                <span>Rensa</span>
              </button>
            </div>

            <span
              role="separator"
              aria-orientation="vertical"
              className="tb-sep"
            />

            <div
              role="group"
              aria-label="Klassificering"
              className="toolbar-group toolbar-group-end"
            >
              <button
                type="button"
                className="tb-btn-primary"
                aria-disabled={
                  !sourceText.trim() || infoClassifying ? "true" : undefined
                }
                onClick={handleClassifyInformation}
              >
                {infoClassifying ? (
                  <RefreshCw size={15} className="spin" aria-hidden="true" />
                ) : (
                  <FileSearch size={15} aria-hidden="true" />
                )}
                <span>
                  {infoClassifying
                    ? "Klassificerar…"
                    : "Klassificera (Grön / Gul / Röd)"}
                </span>
              </button>
            </div>
          </div>

          {sourceText && infoClassifying && (
            <section
              aria-label="Status för informationsklassning"
              className="safety-banner working-banner"
            >
              <div className="safety-main-row">
                <RefreshCw size={15} className="spin" aria-hidden="true" />
                <span>
                  {gemmaScanProgress && gemmaScanProgress.total > 0
                    ? `Analyserar innehåll mot Grön, Gul och Röd informationsnivå (avsnitt ${gemmaScanProgress.chunk}/${gemmaScanProgress.total})…`
                    : "Analyserar dokumentets innehåll mot Grön, Gul och Röd informationsnivå…"}
                </span>
              </div>
            </section>
          )}

          {sourceText && !infoClassifying && infoReport && (
            <section
              aria-label="Resultat av informationsklassning"
              className="info-cards-grid"
            >
              {/* Kort 1 (MWG card: <article> med direkta <header>- och <div>-regioner): Fråga 1 — Vad innehåller informationen? */}
              <article
                className={`info-card level-${infoReport.level}`}
              >
                <header className="info-card-header">
                  <div className="info-card-header-top">
                    <span
                      className={`info-level-badge level-${infoReport.level}`}
                    >
                      {infoReport.level === "green" ? (
                        <CheckCircle2 size={15} aria-hidden="true" />
                      ) : infoReport.level === "yellow" ? (
                        <AlertTriangle size={15} aria-hidden="true" />
                      ) : (
                        <XCircle size={15} aria-hidden="true" />
                      )}
                      <span>{infoReport.badgeTitle}</span>
                    </span>
                    <span
                      className="info-scan-meta"
                      title={`Lokal analys med Regex, BERT NER och EmbeddingGemma 2 (${infoReport.scanDurationMs} ms)`}
                    >
                      {infoReport.paragraphsScanned} avsnitt ·{" "}
                      {infoReport.scanDurationMs} ms
                    </span>
                  </div>
                  <h2 className="info-card-title">
                    1. Vad innehåller informationen?
                  </h2>
                </header>

                <div className="info-card-content">
                  <p className="info-lead">
                    <strong>{infoReport.contentHeadline}.</strong>{" "}
                    {infoReport.contentExplanation}
                  </p>

                  {infoReport.signals.length > 0 && (
                    <ul className="info-signals-list" role="list">
                      {infoReport.signals.map((sig, idx) => (
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
              <article
                className={`info-card guidance-card level-${infoReport.level}`}
              >
                <header className="info-card-header">
                  <div className="info-card-header-top">
                    <span className="info-step-pill">
                      Vägledande beslutsstöd · Lagring &amp; delning
                    </span>
                  </div>
                  <h2 className="info-card-title">
                    2. {infoReport.handling.heading}
                  </h2>
                </header>

                <div className="info-card-content">
                  <p className="info-lead">{infoReport.handling.summary}</p>
                  <dl className="info-guidance-dl">
                    <div className="info-guidance-row">
                      <dt>Lagring</dt>
                      <dd>{infoReport.handling.storageGuidance}</dd>
                    </div>
                    <div className="info-guidance-row">
                      <dt>Delning</dt>
                      <dd>{infoReport.handling.sharingGuidance}</dd>
                    </div>
                    <div className="info-guidance-row">
                      <dt>Exempel</dt>
                      <dd>{infoReport.handling.examplesNote}</dd>
                    </div>
                  </dl>
                  <p className="info-disclaimer">
                    {infoReport.handling.disclaimer}
                  </p>
                </div>
              </article>
            </section>
          )}

          <main className="main-workspace classify-workspace">
            <div className="panel-inner classify-panel-inner">
              {!infoReport || !sourceText ? (
                <div
                  className={`textarea-wrapper ${isDragOver ? "dragover" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragOver(true);
                  }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragOver(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) {
                      handleImportFile(f);
                    }
                  }}
                >
                  <label
                    htmlFor="classify-editor-textarea"
                    className="visually-hidden"
                  >
                    Text eller dokument att klassificera (Grön, Gul eller Röd)
                  </label>
                  <textarea
                    id="classify-editor-textarea"
                    className="editor-textarea"
                    placeholder={
                      sourceText
                        ? "Skriv eller klistra in text att klassificera…"
                        : undefined
                    }
                    value={sourceText}
                    onChange={(e) => handleSourceTextEdit(e.target.value)}
                  />
                  {!sourceText && (
                    <div className="editor-empty-hint">
                      <FileSearch size={32} aria-hidden="true" />
                      <p>
                        Klistra in text (Ctrl+V) eller dra och släpp Word
                        (.docx), Excel (.xlsx), PDF (.pdf) eller .txt här för
                        att få hjälp att bedöma om informationen är{" "}
                        <strong>Grön</strong>, <strong>Gul</strong> eller{" "}
                        <strong>Röd</strong> (ingen text maskeras).
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div
                  className={`review-viewer-wrapper ${
                    isDragOver ? "dragover" : ""
                  }`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragOver(true);
                  }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragOver(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) {
                      handleImportFile(f);
                    }
                  }}
                >
                  <div className="review-viewer-header">
                    <span className="review-viewer-hint">
                      {isEditingClassify
                        ? "Redigerar text — kör ”Klassificera” igen när du vill uppdatera bedömningen."
                        : infoEditedSinceScan
                        ? "Texten har ändrats sedan senaste klassificeringen — uppdatera för ny bedömning."
                        : charInfoFindings.length > 0
                        ? "Originaltext (inga ord maskeras) — markerade avsnitt visar vad som gav utslag i klassningen."
                        : "Originaltext (inga ord maskeras) — inga gula eller röda avsnitt identifierades."}
                    </span>
                    <div className="review-header-actions">
                      {infoEditedSinceScan && (
                        <button
                          type="button"
                          className="tb-btn-subtle rescan-chip"
                          onClick={handleClassifyInformation}
                          title="Kör om informationsklassningen på den ändrade texten"
                        >
                          <RefreshCw size={12} aria-hidden="true" />
                          <span>Uppdatera klassning</span>
                        </button>
                      )}
                      {charInfoFindings.length > 0 &&
                        !isEditingClassify &&
                        !infoEditedSinceScan && (
                          <button
                            type="button"
                            className={`tb-preset-btn para-toggle-btn ${
                              charInfoFindings.some((f) => f.level === "red")
                                ? "has-high"
                                : "only-medium"
                            }`}
                            aria-pressed={showInfoSectionHighlights}
                            onClick={() =>
                              setShowInfoSectionHighlights((v) => !v)
                            }
                            title="Visa eller dölj utmärkning av avsnitt som påverkade klassningen"
                          >
                            <AlertTriangle size={13} aria-hidden="true" />
                            <span>
                              Markerade avsnitt ({charInfoFindings.length})
                            </span>
                          </button>
                        )}
                      <button
                        type="button"
                        className="edit-source-btn"
                        aria-pressed={isEditingClassify}
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => setIsEditingClassify((v) => !v)}
                        title={
                          isEditingClassify
                            ? "Avsluta redigering och visa dokumentvyn (Esc)"
                            : "Redigera texten"
                        }
                      >
                        <Pencil size={13} aria-hidden="true" />
                        <span>Ändra text</span>
                      </button>
                    </div>
                  </div>

                  {isEditingClassify ? (
                    <textarea
                      className="editor-textarea in-review-card"
                      value={sourceText}
                      autoFocus
                      onChange={(e) => handleSourceTextEdit(e.target.value)}
                      onBlur={() => {
                        if (!document.hasFocus()) return;
                        if (infoReport && sourceText.trim()) {
                          setIsEditingClassify(false);
                        }
                      }}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Escape" &&
                          infoReport &&
                          sourceText.trim()
                        ) {
                          e.preventDefault();
                          setIsEditingClassify(false);
                        }
                      }}
                    />
                  ) : (
                    <div className="review-viewer-content">
                      {renderedClassifyDocument}
                    </div>
                  )}
                </div>
              )}

              <div className="tab-footer">
                <span className="footer-stats">
                  {sourceText.length > 0 && (
                    <>
                      {fileName && (
                        <>
                          <strong>{fileName}</strong>
                          {docSummary ? ` (${docSummary})` : ""} ·{" "}
                        </>
                      )}
                      {sourceText.length} tecken ·{" "}
                      {sourceText.split(/\s+/).filter(Boolean).length} ord
                      {infoReport && !infoEditedSinceScan && (
                        <>
                          {" "}
                          · Klassning:{" "}
                          <strong>{infoReport.badgeTitle}</strong>
                        </>
                      )}
                    </>
                  )}
                </span>
                <span className="footer-stats">
                  100 % lokal analys — inget skickas externt
                </span>
              </div>
            </div>
          </main>
        </>
      ) : (
        <>
      {/* ── 2. Kompakt verktygsrad (Modern Web Guidance: role="toolbar") ── */}
      <div
        role="toolbar"
        focusgroup="toolbar wrap"
        aria-label="Dokument och filter"
        className="action-toolbar"
      >
        <div role="group" aria-label="Inmatning" className="toolbar-group">
          <button
            type="button"
            className="tb-btn"
            onClick={handlePasteClipboard}
            title="Klistra in text från urklipp"
          >
            <Clipboard size={15} aria-hidden="true" />
            <span>Klistra in</span>
          </button>
          <button
            type="button"
            className="tb-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Öppna Word (.docx/.odt), Excel (.xlsx/.xls/.ods), PDF (.pdf) eller textfil"
          >
            {importingDoc ? (
              <RefreshCw size={15} className="spin" aria-hidden="true" />
            ) : (
              <Upload size={15} aria-hidden="true" />
            )}
            <span>{importingDoc ? "Läser dokument…" : "Öppna dokument"}</span>
          </button>
        </div>

        <span role="separator" aria-orientation="vertical" className="tb-sep" />

        {/* Snabbprofiler + Filterknapp (aria-pressed) */}
        <div role="group" aria-label="Profiler och filter" className="toolbar-group">
          {(Object.keys(PRESETS) as PresetKey[]).map((key) => (
            <button
              key={key}
              type="button"
              className="tb-preset-btn"
              aria-pressed={activePreset === key}
              onClick={() => handleSelectPreset(key)}
            >
              {PRESETS[key].name}
            </button>
          ))}
          <button
            type="button"
            className="tb-preset-btn tb-filter-btn"
            aria-pressed={showConfig}
            onClick={() => setShowConfig((v) => !v)}
            title="Anpassa kategorier, känslighet, tema och manuella ord"
          >
            <Sliders size={14} aria-hidden="true" />
            <span>Filter</span>
          </button>
        </div>

        <span role="separator" aria-orientation="vertical" className="tb-sep" />

        {/* Primär Granska-knapp (aria-disabled behåller fokus i focusgroup) */}
        <div role="group" aria-label="Analys" className="toolbar-group toolbar-group-end">
          <button
            type="button"
            className="tb-btn-primary"
            onClick={handleAnalyze}
            aria-disabled={
              !sourceText.trim() || analyzing || classifying ? "true" : undefined
            }
            title="Kör lokal regex-, BERT NER- och EmbeddingGemma 2-analys"
          >
            {analyzing || classifying ? (
              <RefreshCw size={16} className="spin" aria-hidden="true" />
            ) : (
              <Sparkles size={16} aria-hidden="true" />
            )}
            <span>
              {analyzing
                ? "Steg 1/2: Maskerar…"
                : classifying
                ? "Steg 2/2: Klassificerar…"
                : "Granska & Maskera"}
            </span>
          </button>
        </div>
      </div>

      {/* ── 3. Filter- och konfigurationspanel ── */}
      {showConfig && (
        <section aria-label="Filterinställningar" className="filter-drawer">
          <div className="filter-drawer-content">
            <div className="filter-chips-row">
              <span className="filter-section-title">Kategorier:</span>
              {ALL_CATEGORIES.map((cat) => {
                const isSelected = selectedEntities.includes(cat);
                const count = categoryCounts.get(cat) ?? 0;
                return (
                  <button
                    key={cat}
                    type="button"
                    className="filter-chip"
                    aria-pressed={isSelected}
                    onClick={() => toggleCategory(cat)}
                    style={{
                      borderColor: isSelected ? COLORS[cat] || "#4338ca" : undefined,
                    }}
                  >
                    <span
                      className="filter-dot"
                      style={{ backgroundColor: COLORS[cat] || "#475569" }}
                      aria-hidden="true"
                    />
                    <span>{cat}</span>
                    {count > 0 && <span className="chip-count">{count}</span>}
                  </button>
                );
              })}
            </div>

            <div className="filter-extra-row">
              <label className="threshold-control" htmlFor="bert-threshold-range">
                <span>BERT Känslighet:</span>
                <input
                  id="bert-threshold-range"
                  type="range"
                  min={0.3}
                  max={0.95}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                />
                <span className="threshold-val">{Math.round(threshold * 100)}%</span>
              </label>

              {/* Treläges temaväljare i inställningspanelen (MWG: dark-mode-toggle) */}
              <div
                role="group"
                aria-label="Färgtema"
                className="theme-choice-group"
              >
                <span className="filter-section-title">Tema:</span>
                <button
                  type="button"
                  className="tb-preset-btn"
                  aria-pressed={colorScheme === "light dark"}
                  onClick={() => applyColorScheme("light dark")}
                >
                  <Monitor size={13} aria-hidden="true" />
                  <span>
                    System ({systemPrefersDark ? "Mörkt" : "Ljust"})
                  </span>
                </button>
                <button
                  type="button"
                  className="tb-preset-btn"
                  aria-pressed={colorScheme === "light"}
                  onClick={() => applyColorScheme("light")}
                >
                  <Sun size={13} aria-hidden="true" />
                  <span>Ljust</span>
                </button>
                <button
                  type="button"
                  className="tb-preset-btn"
                  aria-pressed={colorScheme === "dark"}
                  onClick={() => applyColorScheme("dark")}
                >
                  <Moon size={13} aria-hidden="true" />
                  <span>Mörkt</span>
                </button>
              </div>

              {/* Manuell markering */}
              <div className="manual-add-inline">
                <label htmlFor="manual-word-input" className="visually-hidden">
                  Ord eller namn att maskera manuellt
                </label>
                <input
                  id="manual-word-input"
                  type="text"
                  placeholder="Lägg till ord/namn manuellt…"
                  value={manualWord}
                  onChange={(e) => setManualWord(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddManual()}
                />
                <label htmlFor="manual-label-select" className="visually-hidden">
                  Kategori för manuell maskering
                </label>
                <select
                  id="manual-label-select"
                  value={manualLabel}
                  onChange={(e) => setManualLabel(e.target.value)}
                >
                  {ENTITY_LABELS.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="tb-btn"
                  onClick={handleAddManual}
                  disabled={!manualWord.trim()}
                >
                  <Plus size={14} aria-hidden="true" />
                  <span>Lägg till</span>
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── 4. AI-Gatekeeper & Säkerhetsstatus (Lager 3) ── */}
      {sourceText && (analyzing || classifying || report !== null) && (
        <section
          aria-label="AI-säkerhetsbedömning"
          className={`gatekeeper-banner ${
            analyzing || classifying ? "badge-working" : afterBadge.className
          }`}
        >
          {analyzing || classifying ? (
            <div className="gatekeeper-working" role="status" aria-live="polite">
              <div className="gatekeeper-working-row">
                <RefreshCw size={16} className="spin" />
                <strong>
                  {analyzing
                    ? "Steg 1/2: Identifierar personuppgifter, kortnummer & interna länkar…"
                    : "Steg 2/2: Analyserar dokumentets kontext & säkerhetsklassning…"}
                </strong>
                <span className="gatekeeper-step-count">
                  {analyzing && nerProgress && nerProgress.total > 0
                    ? `Del ${nerProgress.chunk} av ${nerProgress.total}`
                    : classifying &&
                      gemmaScanProgress &&
                      gemmaScanProgress.total > 0
                    ? `Avsnitt ${gemmaScanProgress.chunk} av ${gemmaScanProgress.total}`
                    : "Arbetar lokalt…"}
                </span>
              </div>
              <div className="gatekeeper-progress-track">
                <div
                  className="gatekeeper-progress-fill"
                  style={{
                    width:
                      analyzing && nerProgress && nerProgress.total > 0
                        ? `${Math.max(
                            12,
                            Math.round((nerProgress.chunk / nerProgress.total) * 100)
                          )}%`
                        : classifying &&
                          gemmaScanProgress &&
                          gemmaScanProgress.total > 0
                        ? `${Math.max(
                            15,
                            Math.round(
                              (gemmaScanProgress.chunk / gemmaScanProgress.total) *
                                100
                            )
                          )}%`
                        : "45%",
                  }}
                />
              </div>
            </div>
          ) : (
            <>
              <div className="gatekeeper-header">
                <div className="gatekeeper-badge" title={latencyTooltip || undefined}>
                  {afterBadge.icon}
                  <strong>Efter maskering: {afterBadge.text}</strong>
                  <span className="gatekeeper-summary-inline">
                    — {report?.summary_after}
                  </span>
                </div>

                <div className="gatekeeper-meta">
                  <span className="before-status">
                    Före: <em>{beforeBadge.text}</em>
                  </span>
                  {dismissedFindings.size > 0 && (
                    <button
                      type="button"
                      className="tb-btn-subtle"
                      onClick={() => setDismissedFindings(new Set())}
                      title="Visa alla klarmarkerade avsnitt igen"
                    >
                      Återställ {dismissedFindings.size} klarmarkerade
                    </button>
                  )}
                </div>
              </div>

              {/* Sekundär hopp-lista hopfälld i <details> (MWG: content-first-noise-reduction) */}
              {report?.findings && report.findings.length > 0 && (
                <details className="gatekeeper-findings-details">
                  <summary className="gatekeeper-findings-summary">
                    <span>
                      Visa {report.findings.length} flaggade avsnitt i listvy
                    </span>
                    <span className="summary-sev-pills">
                      {highFindingsCount > 0 && (
                        <span className="sev-pill high">
                          <XCircle size={11} aria-hidden="true" />
                          <span>{highFindingsCount} röd · Hög känslighet</span>
                        </span>
                      )}
                      {medFindingsCount > 0 && (
                        <span className="sev-pill medium">
                          <AlertTriangle size={11} aria-hidden="true" />
                          <span>{medFindingsCount} gul · Gränsfall</span>
                        </span>
                      )}
                    </span>
                    <ChevronDown size={14} aria-hidden="true" className="details-chevron" />
                  </summary>
                  <ul className="findings-list" role="list">
                    {report.findings.map((f, i) => (
                      <li key={i} className={`finding-item ${f.severity}`}>
                        <div className="finding-item-row">
                          <button
                            type="button"
                            className="finding-item-btn"
                            onClick={() => handleJumpToParagraph(f.paragraph_index)}
                            title={f.reason}
                          >
                            <span className={`sev-pill ${f.severity}`}>
                              {f.severity === "high" ? (
                                <XCircle size={11} aria-hidden="true" />
                              ) : (
                                <AlertTriangle size={11} aria-hidden="true" />
                              )}
                              <span>
                                {f.severity === "high"
                                  ? "Hög känslighet"
                                  : "Gränsfall"}
                              </span>
                            </span>
                            <span className="finding-category">
                              Avsnitt {f.paragraph_index} · {f.category}
                            </span>
                            <span className="finding-preview">”{f.preview}”</span>
                            <span className="finding-jump-link">Visa i text →</span>
                          </button>
                          <button
                            type="button"
                            className="finding-dismiss-btn"
                            onClick={() => toggleDismissFinding(f.paragraph_index)}
                            title="Jag har granskat detta avsnitt och bedömer det säkert efter maskering"
                          >
                            <Check size={12} aria-hidden="true" />
                            <span>
                              Klarmarkera
                              <span className="visually-hidden">
                                {" "}
                                avsnitt {f.paragraph_index}
                              </span>
                            </span>
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </section>
      )}

      {/* ── 5. Arbetsyta: Dokumentflikar till vänster, Kopiera + Avmaskera AI-svar till höger ── */}
      <main className="main-workspace">
        <div className="tabs-nav">
          <div
            focusgroup="tablist nomemory"
            aria-label="Dokumentvyer"
            className="tabs-list"
          >
            <button
              type="button"
              id="tab-mask"
              aria-selected={activeTab === "mask"}
              focusgroupstart={activeTab === "table" ? undefined : ""}
              onClick={() => setActiveTab("mask")}
              className="tab-trigger"
            >
              <FileText size={16} aria-hidden="true" />
              <span>Text &amp; Maskering</span>
              {activeMatches.length > 0 && (
                <span className="tab-badge">{activeMatches.length}</span>
              )}
            </button>
            <button
              type="button"
              id="tab-table"
              aria-selected={activeTab === "table"}
              focusgroupstart={activeTab === "table" ? "" : undefined}
              onClick={() => setActiveTab("table")}
              className="tab-trigger"
            >
              <Table size={16} aria-hidden="true" />
              <span>Översättningstabell</span>
              {activeTranslationTable.size > 0 && (
                <span className="tab-badge">{activeTranslationTable.size}</span>
              )}
            </button>
          </div>

          {/* Åtgärder mot extern AI utanför tablist (MWG: tabs): Kopiera maskerad precis till vänster om Avmaskera AI-svar */}
          <div className="workspace-ai-actions">
            <span className="copy-wrapper">
              <button
                type="button"
                className="tb-btn-copy"
                onClick={handleCopyMasked}
                disabled={!sourceText}
                data-copied={copyStatusMasked.startsWith("Kopierad") ? "" : undefined}
                title="Kopiera maskerad text (redo att klistras in i valfri AI-tjänst)"
              >
                {copyStatusMasked.startsWith("Kopierad") ? (
                  <Check size={15} aria-hidden="true" />
                ) : (
                  <Copy size={15} aria-hidden="true" />
                )}
                <span>Kopiera maskerad</span>
              </button>
              <span
                ref={copyTipMaskedRef}
                className={`copy-bubble ${copyStatusMasked ? "show" : ""}`}
                role="status"
              >
                {copyStatusMasked}
              </span>
            </span>

            <button
              type="button"
              id="tab-unmask"
              aria-pressed={activeTab === "unmask"}
              onClick={() =>
                setActiveTab((prev) => (prev === "unmask" ? "mask" : "unmask"))
              }
              className="unmask-nav-btn"
              title="Klistra in svaret från AI-tjänsten för att återställa alla originalnamn lokalt"
            >
              <RotateCcw size={15} aria-hidden="true" />
              <span>Avmaskera AI-svar</span>
              {unmaskedResult.replacements > 0 && (
                <span className="tab-badge">{unmaskedResult.replacements}</span>
              )}
            </button>
          </div>
        </div>

        {/* Tab 1: Text & Maskering */}
        <div
          role="tabpanel"
          aria-labelledby="tab-mask"
          hidden={activeTab !== "mask"}
          className="tab-panel"
        >
          <div className="panel-inner">
            {!showUnifiedDocHeader ? (
              <div
                className={`textarea-wrapper ${isDragOver ? "dragover" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragOver(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) {
                    handleImportFile(f);
                  }
                }}
              >
                <label
                  htmlFor="source-editor-textarea"
                  className="visually-hidden"
                >
                  Källtext att granska och maskera
                </label>
                <textarea
                  id="source-editor-textarea"
                  className="editor-textarea"
                  placeholder={
                    sourceText
                      ? "Skriv eller klistra in text här…"
                      : undefined
                  }
                  value={sourceText}
                  onChange={(e) => handleSourceTextEdit(e.target.value)}
                />
                {!sourceText && (
                  <div className="editor-empty-hint">
                    <Clipboard size={32} aria-hidden="true" />
                    <p>
                      Klistra in text (Ctrl+V) eller dra och släpp Word (.docx), Excel (.xlsx), PDF (.pdf) eller .txt här
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div
                className={`review-viewer-wrapper ${isDragOver ? "dragover" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragOver(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) {
                    handleImportFile(f);
                  }
                }}
              >
                <div className="review-viewer-header">
                  <span className="review-viewer-hint">
                    {isEditing
                      ? "Redigerar källtext — befintliga maskeringar bevaras. Klicka utanför när du är klar."
                      : "Klicka på en markering för att undanta, eller klicka i ett stycke för att redigera."}
                  </span>
                  <div className="review-header-actions">
                    {textEditedSinceScan && !isEditing && (
                      <button
                        type="button"
                        className="tb-btn-subtle rescan-chip"
                        onClick={handleAnalyze}
                        title="Texten har redigerats — klicka för att skanna eventuella nya ord"
                      >
                        <RefreshCw size={12} aria-hidden="true" />
                        <span>Uppdatera analys</span>
                      </button>
                    )}
                    {charFindings.length > 0 && !isEditing && (
                      <button
                        type="button"
                        className={`tb-preset-btn para-toggle-btn ${
                          highFindingsCount > 0 ? "has-high" : "only-medium"
                        }`}
                        aria-pressed={showParaHighlights}
                        onClick={() => setShowParaHighlights((v) => !v)}
                        title="Visa eller dölj utmärkning av känsliga avsnitt i texten"
                      >
                        <AlertTriangle size={13} aria-hidden="true" />
                        <span>Känsliga avsnitt ({charFindings.length})</span>
                      </button>
                    )}
                    <button
                      type="button"
                      className="edit-source-btn"
                      aria-pressed={isEditing}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => {
                        setEditingSlice(null);
                        setIsEditing((v) => !v);
                      }}
                      title={
                        isEditing
                          ? "Avsluta redigering och återgå till granskningsvyn (Esc)"
                          : "Ändra hela källtexten med bibehållna maskeringar"
                      }
                    >
                      <Pencil size={13} aria-hidden="true" />
                      <span>Ändra källtext</span>
                    </button>
                  </div>
                </div>

                {isEditing ? (
                  <textarea
                    className="editor-textarea in-review-card"
                    value={sourceText}
                    autoFocus
                    onChange={(e) => handleSourceTextEdit(e.target.value)}
                    onBlur={() => {
                      if (!document.hasFocus()) return;
                      if (hasAnalyzed && sourceText.trim()) {
                        setIsEditing(false);
                      }
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Escape" && hasAnalyzed && sourceText.trim()) {
                        e.preventDefault();
                        setIsEditing(false);
                      }
                    }}
                  />
                ) : (
                  <div className="review-viewer-content">{renderedReviewText}</div>
                )}
              </div>
            )}

            {/* Fot-rad för tab 1 */}
            <div className="tab-footer">
              <span className="footer-stats">
                {sourceText.length > 0 && (
                  <>
                    {fileName && (
                      <>
                        <strong>{fileName}</strong>
                        {docSummary ? ` (${docSummary})` : ""} ·{" "}
                      </>
                    )}
                    {sourceText.length} tecken ·{" "}
                    {sourceText.split(/\s+/).filter(Boolean).length} ord
                    {activeMatches.length > 0 && (
                      <>
                        {" "}
                        · <strong>{activeMatches.length}</strong> maskerade
                        uppgifter
                      </>
                    )}
                    {ignoredKeys.size > 0 && (
                      <>
                        {" "}
                        · <em>{ignoredKeys.size}</em> undantagna
                      </>
                    )}
                  </>
                )}
              </span>

              <div className="footer-actions">
                {originalDocxBytes && (
                  <button
                    type="button"
                    className="tb-btn"
                    onClick={handleSaveMaskedDocx}
                    disabled={!sourceText}
                    title="Spara en maskerad kopia av Word-dokumentet (.docx) med bibehållen formatering"
                  >
                    <Download size={14} aria-hidden="true" />
                    <span>Spara maskerad .docx</span>
                  </button>
                )}
                {originalXlsxBytes && (
                  <button
                    type="button"
                    className="tb-btn"
                    onClick={handleSaveMaskedXlsx}
                    disabled={!sourceText}
                    title="Spara en maskerad kopia av Excel-arbetsboken (.xlsx) med bibehållen formatering och formler"
                  >
                    <Download size={14} aria-hidden="true" />
                    <span>Spara maskerad .xlsx</span>
                  </button>
                )}
                <button
                  type="button"
                  className="tb-btn"
                  onClick={handleSaveMaskedFile}
                  disabled={!sourceText}
                >
                  <Download size={14} aria-hidden="true" />
                  <span>Spara maskerad .txt</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Tab 2: Översättningstabell */}
        <div
          role="tabpanel"
          aria-labelledby="tab-table"
          hidden={activeTab !== "table"}
          tabIndex={
            activeTab === "table" && activeTranslationTable.size === 0
              ? 0
              : undefined
          }
          className="tab-panel"
        >
          <div className="panel-inner">
            {activeTranslationTable.size === 0 ? (
              <div className="empty-panel-hint">
                <Table size={40} />
                <p>Inga aktiva maskeringar att visa ännu.</p>
                <p className="sub">Klistra in en text och kör ”Granska &amp; Maskera” först.</p>
              </div>
            ) : (
              <>
                <div className="table-wrapper">
                  <table className="custom-table">
                    <caption className="visually-hidden">
                      Översättningstabell mellan maskerade platshållare och originalvärden
                    </caption>
                    <thead>
                      <tr>
                        <th scope="col">Typ</th>
                        <th scope="col">Platshållare (Maskerad)</th>
                        <th scope="col">Originalvärde i text</th>
                        <th scope="col">Förekomster</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...activeTranslationTable.values()].map((entry) => (
                        <tr key={entry.originalText}>
                          <td>
                            <span
                              className="entity-dot"
                              style={{
                                backgroundColor: COLORS[entry.label] || "#94a3b8",
                              }}
                            />
                            <span className="entity-type-name">{entry.label}</span>
                          </td>
                          <td>
                            <code className="placeholder-tag">
                              [{entry.maskedLabel.toUpperCase()}]
                            </code>
                          </td>
                          <td className="orig-cell">{entry.originalText}</td>
                          <td className="count-cell">{entry.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="tab-footer">
                  <span className="footer-stats">
                    {activeTranslationTable.size} unika värden mappade
                  </span>
                  <div className="footer-actions">
                    <button
                      type="button"
                      className="tb-btn"
                      onClick={async () => {
                        const path = await save({
                          defaultPath: "oversattning.json",
                          filters: [{ name: "JSON", extensions: ["json"] }],
                        });
                        if (path) {
                          await invoke("write_file", {
                            path,
                            content: JSON.stringify(
                              [...activeTranslationTable.values()],
                              null,
                              2
                            ),
                          });
                        }
                      }}
                    >
                      <Download size={14} />
                      <span>Exportera JSON</span>
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Vy 3: Avmaskera AI-svar */}
        <div
          role="region"
          aria-labelledby="tab-unmask"
          hidden={activeTab !== "unmask"}
          className="tab-panel"
        >
          <div className="panel-inner unmask-grid">
            <div className="unmask-col">
              <label className="unmask-label">
                <span>1. Klistra in svaret du fick från AI-tjänsten:</span>
                <textarea
                  className="editor-textarea"
                  placeholder="Klistra in AI-svaret här (t.ex. 'Enligt sammanfattningen ska [PERSON A] kontakta [ORGANISATION A]...')..."
                  value={aiResponseInput}
                  onChange={(e) => setAiResponseInput(e.target.value)}
                />
              </label>
            </div>

            <div className="unmask-col">
              <div className="unmask-header-row">
                <span className="unmask-label-text">
                  2. Avmaskerat svar (återställt med originalvärden):
                </span>
                <div className="unmask-header-actions">
                  <button
                    type="button"
                    className="tb-preset-btn unmask-format-toggle"
                    aria-pressed={formattedUnmaskView}
                    onClick={() => setFormattedUnmaskView((v) => !v)}
                    title="Växla mellan snyggt formaterat dokumentläge och rå Markdown-text"
                  >
                    <FileText size={13} aria-hidden="true" />
                    <span>{formattedUnmaskView ? "Formaterad vy" : "Råtext"}</span>
                  </button>
                  <span className="copy-wrapper">
                    <button
                      type="button"
                      className="tb-btn-copy"
                      onClick={handleCopyUnmasked}
                      disabled={!unmaskedResult.text}
                      data-copied={copyStatusUnmasked.startsWith("Kopierad") ? "" : undefined}
                      title="Kopierar som formaterad text (för Word/Outlook/Teams) samt ren text"
                    >
                      {copyStatusUnmasked.startsWith("Kopierad") ? (
                        <Check size={14} aria-hidden="true" />
                      ) : (
                        <Copy size={14} aria-hidden="true" />
                      )}
                      <span>Kopiera svar</span>
                    </button>
                    <span
                      ref={copyTipUnmaskedRef}
                      className={`copy-bubble ${copyStatusUnmasked ? "show" : ""}`}
                      role="status"
                    >
                      {copyStatusUnmasked}
                    </span>
                  </span>
                </div>
              </div>
              <div
                className={`unmasked-preview-box ${
                  formattedUnmaskView ? "formatted-mode" : "raw-mode"
                }`}
              >
                {unmaskedResult.text ? (
                  formattedUnmaskView ? (
                    renderFormattedMarkdown(unmaskedResult.text)
                  ) : (
                    unmaskedResult.text
                  )
                ) : (
                  <span className="empty-italic">
                    Det återställda svaret visas här automatiskt så snart du klistrar in ett svar…
                  </span>
                )}
              </div>
              <div className="unmask-status-bar">
                <span>
                  {unmaskedResult.replacements > 0 ? (
                    <>
                      <Check size={14} aria-hidden="true" />
                      <strong>{unmaskedResult.replacements}</strong> platshållare återställda lokalt
                    </>
                  ) : (
                    "Inga platshållare hittades i svaret än."
                  )}
                </span>
                <label
                  className="tb-btn"
                  title="Välj ett maskerat Word- (.docx) eller Excel-dokument (.xlsx) för att spara en återställd (avmaskerad) kopia med bibehållen formatering"
                >
                  <Upload size={14} aria-hidden="true" />
                  <span>Avmaskera .docx / .xlsx-fil</span>
                  <input
                    type="file"
                    accept=".docx,.xlsx,.xlsm"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) {
                        handleUnmaskDocumentFile(f);
                        e.target.value = "";
                      }
                    }}
                  />
                </label>
              </div>
            </div>
          </div>
        </div>
      </main>
        </>
      )}
        </>
      )}
    </div>
  );
}
