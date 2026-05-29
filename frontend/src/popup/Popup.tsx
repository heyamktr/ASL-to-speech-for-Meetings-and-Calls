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
  timestamp: Date;
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

  // Demo timer cleanup
  useEffect(() => () => { if (demoTimer.current) clearInterval(demoTimer.current); }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

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
            { id: entryId++, word: p.word, confidence: p.confidence, timestamp: new Date() },
          ]);
        }
      }
    };

    chrome.storage.onChanged.addListener(listener);
    return () => chrome.storage.onChanged.removeListener(listener);
  }, []);

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
          { id: entryId++, word, confidence, timestamp: new Date(), demo: true },
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
      <div className="w-full h-screen bg-violet-100 flex items-center justify-center">
        <div className="w-6 h-6 rounded-full border-2 border-violet-300 border-t-violet-600 animate-spin" />
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-violet-50 font-sans flex flex-col overflow-hidden">

      {/* ── Header ─────────────────────────────────── */}
      <div className="bg-violet-200 px-5 pt-6 pb-5 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-white border-2 border-violet-300 flex items-center justify-center shadow-sm">
            <span className="text-xl leading-none">🤟</span>
          </div>
          <div>
            <h1 className="text-base font-bold text-violet-900 leading-tight tracking-wide">ASL Interpreter</h1>
            <p className="text-xs text-violet-600">Sign language → voice</p>
          </div>
        </div>
      </div>

      {/* ── Controls ───────────────────────────────── */}
      <div className="px-4 pt-4 pb-3 flex-shrink-0 flex flex-col gap-3">

        {/* Toggle */}
        <button
          onClick={toggle}
          className={`w-full py-3 rounded-2xl text-sm font-bold tracking-wide transition-all duration-150 active:scale-95 shadow-sm ${
            enabled
              ? "bg-violet-500 text-white hover:bg-violet-600"
              : "bg-white text-violet-700 border-2 border-violet-300 hover:bg-violet-50"
          }`}
        >
          {enabled ? "■  Disable" : "▶  Enable"}
        </button>

        {/* Status row */}
        <div className="flex items-center gap-2">
          <div className="flex-1 flex items-center justify-between bg-white rounded-xl px-3 py-2 border border-violet-200 shadow-sm">
            <span className="text-xs text-violet-500 font-semibold">Backend</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? "bg-green-400" : "bg-violet-200"}`} />
              <span className={`text-xs font-semibold ${wsConnected ? "text-green-600" : "text-violet-400"}`}>
                {wsConnected ? "connected" : "offline"}
              </span>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-between bg-white rounded-xl px-3 py-2 border border-violet-200 shadow-sm">
            <span className="text-xs text-violet-500 font-semibold">Camera</span>
            <span className={`text-xs font-semibold ${
              detectionState === "no_hand"  ? "text-amber-500"
              : detectionState === "thinking"  ? "text-violet-500"
              : detectionState === "predicted" ? "text-green-600"
              : "text-violet-400"
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
        <div className="bg-violet-100 rounded-2xl px-5 py-4 border border-violet-200 min-h-20 flex items-center justify-center">
          <DetectionDisplay enabled={enabled} state={detectionState} prediction={lastPrediction} />
        </div>
      </div>

      {/* ── Transcript ─────────────────────────────── */}
      <div className="flex-1 flex flex-col px-4 pb-4 min-h-0">
        <div className="flex items-center justify-between mb-2 flex-shrink-0">
          <span className="text-xs font-bold text-violet-600 uppercase tracking-widest">Transcript</span>
          <div className="flex items-center gap-2">
            <button
              onClick={clearTranscript}
              className="text-xs text-violet-400 hover:text-violet-700 font-medium transition-colors"
            >
              Clear
            </button>
            <button
              onClick={toggleDemo}
              className={`text-xs font-bold px-2.5 py-1 rounded-lg transition-all ${
                demoActive
                  ? "bg-amber-300 text-amber-900 hover:bg-amber-400"
                  : "bg-violet-200 text-violet-700 hover:bg-violet-300"
              }`}
            >
              {demoActive ? "⏹ Stop Demo" : "▶ Demo"}
            </button>
          </div>
        </div>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto flex flex-col gap-1.5 pr-0.5"
          style={{ scrollBehavior: "smooth" }}
        >
          {transcript.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-6">
              <span className="text-3xl mb-2 opacity-20">📝</span>
              <p className="text-xs text-violet-400">No words yet.<br />Interpreted signs will appear here.</p>
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
        <p className="text-center text-xs text-violet-400 leading-relaxed">
          Camera is used locally only. No video is sent to any server.
        </p>
      </div>
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function TranscriptRow({ entry, index }: { entry: TranscriptEntry; index: number }) {
  const pct = Math.round(entry.confidence * 100);
  const pctColor = pct >= 85 ? "text-green-600" : pct >= 65 ? "text-violet-600" : "text-amber-500";

  return (
    <div className="flex items-center gap-3 bg-white rounded-xl px-3 py-2.5 border border-violet-100 shadow-sm">
      <span className="text-xs text-violet-300 w-5 text-right flex-shrink-0 font-mono">{index + 1}</span>
      <span className="flex-1 text-sm font-semibold text-violet-800 capitalize">{entry.word}</span>
      <span className="text-xs text-violet-400 font-mono flex-shrink-0">{formatTime(entry.timestamp)}</span>
      <span className={`text-xs font-bold flex-shrink-0 w-8 text-right ${pctColor}`}>{pct}%</span>
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
        <div className="text-2xl mb-1.5 opacity-30">🤟</div>
        <p className="text-xs text-violet-500 leading-relaxed">Enable and open Google Meet<br />to start capturing.</p>
      </div>
    );
  }

  if (state === "no_hand") {
    return (
      <div className="text-center">
        <div className="text-2xl mb-1.5">✋</div>
        <p className="text-sm font-semibold text-violet-800">No hand detected</p>
        <p className="text-xs text-violet-500 mt-0.5">Show your hand to the camera</p>
      </div>
    );
  }

  if (state === "thinking") {
    return (
      <div className="text-center">
        <div className="flex gap-1.5 justify-center mb-2">
          {[0, 150, 300].map((delay) => (
            <span key={delay} className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: `${delay}ms` }} />
          ))}
        </div>
        <p className="text-sm font-semibold text-violet-700">Thinking…</p>
      </div>
    );
  }

  if (state === "predicted" && prediction) {
    return (
      <div className="text-center">
        <p className="text-2xl font-bold text-violet-900 tracking-wide mb-1 capitalize">{prediction.word}</p>
        <span className="text-xs font-semibold text-violet-500">{Math.round(prediction.confidence * 100)}% confidence</span>
      </div>
    );
  }

  return null;
}
