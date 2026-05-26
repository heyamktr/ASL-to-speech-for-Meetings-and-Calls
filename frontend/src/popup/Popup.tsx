import { useEffect, useRef, useState } from "react";
import { getSettings, setSettings, getRuntimeState } from "../lib/storage";
import { sendToActiveTab } from "../lib/messaging";
import type { DetectionState } from "../lib/storage";

interface LastPrediction {
  word: string;
  confidence: number;
}

interface TranscriptEntry {
  id: number;
  word: string;
  confidence: number;
  demo?: boolean;
}

const DEMO_WORDS = [
  "hello", "thank you", "yes", "please", "help",
  "water", "good morning", "more", "stop", "go",
  "sorry", "no", "finished", "name", "nice to meet you",
];

let entryId = 0;

export default function Popup() {
  const [enabled, setEnabled]               = useState(false);
  const [loading, setLoading]               = useState(true);
  const [wsConnected, setWsConnected]       = useState(false);
  const [detectionState, setDetectionState] = useState<DetectionState>("idle");
  const [lastPrediction, setLastPrediction] = useState<LastPrediction | null>(null);
  const [transcript, setTranscript]         = useState<TranscriptEntry[]>([]);
  const [demoActive, setDemoActive]         = useState(false);

  const scrollRef  = useRef<HTMLDivElement>(null);
  const demoTimer  = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-scroll transcript to bottom on new entry
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  // Load initial state + listen for live updates
  useEffect(() => {
    Promise.all([getSettings(), getRuntimeState()]).then(([settings, state]) => {
      setEnabled(settings.enabled);
      setWsConnected(state.wsConnected);
      setDetectionState(state.detectionState);
      setLastPrediction(state.lastPrediction ?? null);
      setLoading(false);
    });

    const listener = (changes: Record<string, chrome.storage.StorageChange>) => {
      if (changes.enabled)         setEnabled(changes.enabled.newValue);
      if (changes.wsConnected)     setWsConnected(changes.wsConnected.newValue);
      if (changes.detectionState)  setDetectionState(changes.detectionState.newValue);
      if (changes.lastPrediction) {
        const p = changes.lastPrediction.newValue as LastPrediction | undefined;
        setLastPrediction(p ?? null);
        if (p) {
          setTranscript((prev) => [
            ...prev,
            { id: entryId++, word: p.word, confidence: p.confidence },
          ]);
        }
      }
    };

    chrome.storage.onChanged.addListener(listener);
    return () => chrome.storage.onChanged.removeListener(listener);
  }, []);

  // Demo timer cleanup
  useEffect(() => () => { if (demoTimer.current) clearInterval(demoTimer.current); }, []);

  async function toggle() {
    const next = !enabled;
    setEnabled(next);
    await setSettings({ enabled: next });
    sendToActiveTab({ type: "TOGGLE", enabled: next });
  }

  function toggleDemo() {
    if (demoActive) {
      if (demoTimer.current) clearInterval(demoTimer.current);
      demoTimer.current = null;
      setDemoActive(false);
    } else {
      setDemoActive(true);
      demoTimer.current = setInterval(() => {
        const word = DEMO_WORDS[Math.floor(Math.random() * DEMO_WORDS.length)];
        const confidence = parseFloat((0.68 + Math.random() * 0.31).toFixed(2));
        setTranscript((prev) => [
          ...prev,
          { id: entryId++, word, confidence, demo: true },
        ]);
        setLastPrediction({ word, confidence });
        setDetectionState("predicted");
      }, 1800);
    }
  }

  function clearTranscript() {
    setTranscript([]);
  }

  if (loading) {
    return (
      <div className="w-full h-screen bg-blue-600 flex items-center justify-center">
        <div className="w-6 h-6 rounded-full border-2 border-blue-300 border-t-white animate-spin" />
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-blue-50 font-sans flex flex-col overflow-hidden">

      {/* ── Header ─────────────────────────────────── */}
      <div className="bg-blue-600 px-5 pt-6 pb-5 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-blue-500 border-2 border-blue-400 flex items-center justify-center shadow-lg">
            <span className="text-xl leading-none">🤟</span>
          </div>
          <div>
            <h1 className="text-base font-bold text-white leading-tight tracking-wide">ASL Interpreter</h1>
            <p className="text-xs text-blue-200">Sign language → voice</p>
          </div>
        </div>
      </div>

      {/* ── Controls ───────────────────────────────── */}
      <div className="px-4 pt-4 pb-3 flex-shrink-0 flex flex-col gap-3">

        {/* Toggle */}
        <button
          onClick={toggle}
          className={`w-full py-3 rounded-2xl text-sm font-bold tracking-wide transition-all duration-150 active:scale-95 shadow-md ${
            enabled
              ? "bg-blue-600 text-white shadow-blue-300 hover:bg-blue-700"
              : "bg-white text-blue-600 border-2 border-blue-400 hover:bg-blue-50 shadow-blue-100"
          }`}
        >
          {enabled ? "■  Disable" : "▶  Enable"}
        </button>

        {/* Status row */}
        <div className="flex items-center gap-2">
          {/* Backend */}
          <div className="flex-1 flex items-center justify-between bg-white rounded-xl px-3 py-2 border border-blue-200 shadow-sm">
            <span className="text-xs text-blue-500 font-semibold">Backend</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? "bg-green-500 shadow-sm shadow-green-300" : "bg-blue-300"}`} />
              <span className={`text-xs font-semibold ${wsConnected ? "text-green-600" : "text-blue-400"}`}>
                {wsConnected ? "connected" : "offline"}
              </span>
            </div>
          </div>
          {/* Detection */}
          <div className="flex-1 flex items-center justify-between bg-white rounded-xl px-3 py-2 border border-blue-200 shadow-sm">
            <span className="text-xs text-blue-500 font-semibold">Camera</span>
            <span className={`text-xs font-semibold ${
              detectionState === "no_hand" ? "text-amber-500"
              : detectionState === "thinking" ? "text-blue-500"
              : detectionState === "predicted" ? "text-green-600"
              : "text-blue-300"
            }`}>
              {detectionState === "idle"      ? "idle"
               : detectionState === "no_hand"  ? "no hand"
               : detectionState === "thinking" ? "reading…"
               : "detected"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Current prediction card ─────────────────── */}
      <div className="px-4 pb-3 flex-shrink-0">
        <div className="bg-blue-600 rounded-2xl px-5 py-4 shadow-lg shadow-blue-200 min-h-20 flex items-center justify-center">
          <DetectionDisplay enabled={enabled} state={detectionState} prediction={lastPrediction} />
        </div>
      </div>

      {/* ── Transcript ─────────────────────────────── */}
      <div className="flex-1 flex flex-col px-4 pb-4 min-h-0">
        {/* Transcript header */}
        <div className="flex items-center justify-between mb-2 flex-shrink-0">
          <span className="text-xs font-bold text-blue-700 uppercase tracking-widest">Transcript</span>
          <div className="flex items-center gap-2">
            <button
              onClick={clearTranscript}
              className="text-xs text-blue-400 hover:text-blue-600 font-medium transition-colors"
            >
              Clear
            </button>
            <button
              onClick={toggleDemo}
              className={`text-xs font-bold px-2.5 py-1 rounded-lg transition-all ${
                demoActive
                  ? "bg-amber-400 text-white hover:bg-amber-500"
                  : "bg-blue-200 text-blue-700 hover:bg-blue-300"
              }`}
            >
              {demoActive ? "⏹ Stop Demo" : "▶ Demo"}
            </button>
          </div>
        </div>

        {/* Scrollable list */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto flex flex-col gap-1.5 pr-0.5"
          style={{ scrollBehavior: "smooth" }}
        >
          {transcript.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-6">
              <span className="text-3xl mb-2 opacity-20">📝</span>
              <p className="text-xs text-blue-300">No words yet.<br />Interpreted signs will appear here.</p>
            </div>
          ) : (
            transcript.map((entry, i) => (
              <TranscriptRow key={entry.id} entry={entry} index={i} />
            ))
          )}
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────── */}
      <div className="px-4 pb-4 flex-shrink-0">
        <p className="text-center text-xs text-blue-400 leading-relaxed">
          Camera is used locally only. No video is sent to any server.
        </p>
      </div>
    </div>
  );
}

function TranscriptRow({ entry, index }: { entry: TranscriptEntry; index: number }) {
  const pct = Math.round(entry.confidence * 100);
  const barColor = pct >= 85 ? "bg-green-400" : pct >= 65 ? "bg-blue-400" : "bg-amber-400";

  return (
    <div className="flex items-center gap-3 bg-white rounded-xl px-3 py-2.5 border border-blue-100 shadow-sm">
      <span className="text-xs text-blue-300 w-5 text-right flex-shrink-0 font-mono">{index + 1}</span>
      <span className="flex-1 text-sm font-semibold text-blue-800 capitalize">{entry.word}</span>
      <div className="flex items-center gap-1.5 flex-shrink-0">
        <div className="w-12 h-1.5 rounded-full bg-blue-100 overflow-hidden">
          <div className={`h-full rounded-full transition-all duration-300 ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
        <span className="text-xs text-blue-400 font-medium w-8 text-right">{pct}%</span>
      </div>
      {entry.demo && (
        <span className="text-xs text-amber-400 font-bold flex-shrink-0">DEMO</span>
      )}
    </div>
  );
}

function DetectionDisplay({
  enabled,
  state,
  prediction,
}: {
  enabled: boolean;
  state: DetectionState;
  prediction: LastPrediction | null;
}) {
  if (!enabled || state === "idle") {
    return (
      <div className="text-center">
        <div className="text-2xl mb-1.5 opacity-40">🤟</div>
        <p className="text-xs text-blue-200 leading-relaxed">Enable and open Google Meet<br />to start capturing.</p>
      </div>
    );
  }

  if (state === "no_hand") {
    return (
      <div className="text-center">
        <div className="text-2xl mb-1.5">✋</div>
        <p className="text-sm font-semibold text-white">No hand detected</p>
        <p className="text-xs text-blue-200 mt-0.5">Show your hand to the camera</p>
      </div>
    );
  }

  if (state === "thinking") {
    return (
      <div className="text-center">
        <div className="flex gap-1.5 justify-center mb-2">
          {[0, 150, 300].map((delay) => (
            <span key={delay} className="w-2 h-2 rounded-full bg-blue-300 animate-bounce" style={{ animationDelay: `${delay}ms` }} />
          ))}
        </div>
        <p className="text-sm font-semibold text-white">Thinking…</p>
      </div>
    );
  }

  if (state === "predicted" && prediction) {
    return (
      <div className="text-center">
        <p className="text-2xl font-bold text-white tracking-wide mb-1.5 capitalize">{prediction.word}</p>
        <div className="flex items-center justify-center gap-2">
          <div className="h-2 rounded-full bg-blue-500 w-20 overflow-hidden">
            <div
              className="h-full rounded-full bg-white transition-all duration-300"
              style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
            />
          </div>
          <span className="text-xs text-blue-200 font-semibold">{Math.round(prediction.confidence * 100)}%</span>
        </div>
      </div>
    );
  }

  return null;
}
