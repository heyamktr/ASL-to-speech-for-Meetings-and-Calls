import { useEffect, useRef, useState } from "react";
import { getSettings, getRuntimeState, setSettings } from "../lib/storage";
import { sendToActiveTab } from "../lib/messaging";
import type { DetectionState, Theme } from "../lib/storage";

// ── Types ─────────────────────────────────────────────────────────────────────

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

// ── Theme CSS variables ───────────────────────────────────────────────────────

const THEME_VARS: Record<Theme, React.CSSProperties> = {
  light: {
    '--pb':   '#f5f3ff',          // page background
    '--ph':   '#ede9fe',          // header background
    '--pc':   '#ffffff',          // card background
    '--ps':   '#ede9fe',          // sentence card background
    '--pt':   '#1a1a2e',          // primary text
    '--pm':   '#7c6fa0',          // muted text
    '--pa':   '#7c3aed',          // accent
    '--pbo':  'rgba(109,78,200,0.18)', // border
    '--pbtn': '#7c3aed',          // active button bg
    '--pbtnt':'#ffffff',          // active button text
  } as React.CSSProperties,
  dark: {
    '--pb':   '#0d0920',
    '--ph':   '#180f38',
    '--pc':   '#1c1440',
    '--ps':   '#160d30',
    '--pt':   '#e8e8ff',
    '--pm':   '#9b8fc0',
    '--pa':   '#a78bfa',
    '--pbo':  'rgba(167,139,250,0.18)',
    '--pbtn': '#7c3aed',
    '--pbtnt':'#ffffff',
  } as React.CSSProperties,
  'high-contrast': {
    '--pb':   '#000000',
    '--ph':   '#111111',
    '--pc':   '#0a0a0a',
    '--ps':   '#0a0a0a',
    '--pt':   '#ffffff',
    '--pm':   '#bbbbbb',
    '--pa':   '#ffff00',
    '--pbo':  'rgba(255,255,255,0.4)',
    '--pbtn': '#ffff00',
    '--pbtnt':'#000000',
  } as React.CSSProperties,
};

// ── Constants ─────────────────────────────────────────────────────────────────

const DEMO_WORDS = [
  "hello", "thank you", "yes", "please", "help",
  "water", "good morning", "more", "stop", "go",
  "sorry", "no", "finished", "name", "nice to meet you",
];

const THEME_OPTIONS: { value: Theme; label: string; icon: string }[] = [
  { value: 'light',         label: 'Light',  icon: '☀️' },
  { value: 'dark',          label: 'Dark',   icon: '🌙' },
  { value: 'high-contrast', label: 'Hi-Con', icon: '◑'  },
];

let entryId = 0;

// Font size is controlled from the overlay; here we only read it. Coerce to a
// valid number so a legacy string/NaN value can't break the --fs CSS variable.
function toFontSize(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(32, Math.max(8, Math.round(n))) : 20;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Popup() {
  const [enabled, setEnabled]               = useState(false);
  const [loading, setLoading]               = useState(true);
  const [wsConnected, setWsConnected]       = useState(false);
  const [detectionState, setDetectionState] = useState<DetectionState>("idle");
  const [lastPrediction, setLastPrediction] = useState<LastPrediction | null>(null);
  const [transcript, setTranscript]         = useState<TranscriptEntry[]>([]);
  const [demoActive, setDemoActive]         = useState(false);
  const [theme, setThemeState]              = useState<Theme>('dark');
  const [fontSize, setFontSizeState]        = useState<number>(20);
  const [privacyOpen, setPrivacyOpen]       = useState(false);
  const [currentSentence, setCurrentSentence] = useState('');

  const scrollRef   = useRef<HTMLDivElement>(null);
  const demoTimer   = useRef<ReturnType<typeof setInterval> | null>(null);
  const sentenceRef = useRef('');

  useEffect(() => () => { if (demoTimer.current) clearInterval(demoTimer.current); }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  useEffect(() => {
    Promise.all([getSettings(), getRuntimeState()]).then(([settings, state]) => {
      setEnabled(settings.enabled);
      setThemeState(settings.theme);
      setFontSizeState(toFontSize(settings.fontSize));
      setWsConnected(state.wsConnected);
      setDetectionState(state.detectionState);
      setLastPrediction(state.lastPrediction ?? null);
      setLoading(false);
    });

    const listener = (changes: Record<string, chrome.storage.StorageChange>) => {
      if (changes.enabled)        setEnabled(changes.enabled.newValue);
      if (changes.wsConnected)    setWsConnected(changes.wsConnected.newValue);
      if (changes.detectionState) setDetectionState(changes.detectionState.newValue);
      if (changes.theme)          setThemeState(changes.theme.newValue as Theme);
      if (changes.fontSize)       setFontSizeState(toFontSize(changes.fontSize.newValue));
      if (changes.lastPrediction) {
        const p = changes.lastPrediction.newValue as LastPrediction | undefined;
        setLastPrediction(p ?? null);
        if (p) {
          const next = sentenceRef.current
            ? sentenceRef.current + ' ' + p.word
            : p.word;
          sentenceRef.current = next;
          setCurrentSentence(next);
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

  // ── Actions ──────────────────────────────────────────────────────────────────

  async function toggle() {
    const next = !enabled;
    setEnabled(next);
    await setSettings({ enabled: next });
    sendToActiveTab({ type: "TOGGLE", enabled: next });
  }

  async function saveTheme(t: Theme) {
    setThemeState(t);
    await setSettings({ theme: t });
  }

  function clearAll() {
    sentenceRef.current = '';
    setCurrentSentence('');
    setTranscript([]);
    sendToActiveTab({ type: "CLEAR_SENTENCE" });
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
        const next = sentenceRef.current ? sentenceRef.current + ' ' + word : word;
        sentenceRef.current = next;
        setCurrentSentence(next);
        setTranscript((prev) => [
          ...prev,
          { id: entryId++, word, confidence, timestamp: new Date(), demo: true },
        ]);
        setLastPrediction({ word, confidence });
        setDetectionState("predicted");
      }, 1800);
    }
  }

  // ── Loading ───────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="w-full h-screen flex items-center justify-center"
           style={{ background: 'var(--pb)', ...THEME_VARS[theme] }}>
        <div className="w-6 h-6 rounded-full border-2 border-t-transparent animate-spin"
             style={{ borderColor: 'var(--pa)', borderTopColor: 'transparent' }} />
      </div>
    );
  }

  const tv = THEME_VARS[theme];

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div
      className="w-full h-screen font-sans flex flex-col overflow-hidden"
      style={{ background: 'var(--pb)', color: 'var(--pt)', '--fs': `${fontSize}pt`, ...tv } as React.CSSProperties}
    >

      {/* ── Header ──────────────────────────────────── */}
      <div className="px-5 pt-5 pb-4 flex-shrink-0" style={{ background: 'var(--ph)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-sm flex-shrink-0"
               style={{ background: 'var(--pc)', border: '2px solid var(--pbo)' }}>
            <span className="text-xl leading-none">🤟</span>
          </div>
          <div className="flex-1">
            <h1 className="text-base font-bold leading-tight tracking-wide" style={{ color: 'var(--pt)' }}>
              ASL Interpreter
            </h1>
            <p className="text-xs" style={{ color: 'var(--pm)' }}>Sign language → captions</p>
          </div>
          <button
            onClick={() => setPrivacyOpen((v) => !v)}
            className="p-1.5 rounded-xl transition-colors"
            style={{ color: 'var(--pm)' }}
            title="Privacy info"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="16" x2="12" y2="12"/>
              <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
          </button>
        </div>

        {/* Privacy panel */}
        {privacyOpen && (
          <div className="mt-3 rounded-xl px-4 py-3 text-xs leading-relaxed space-y-2"
               style={{ background: 'var(--pc)', border: '1px solid var(--pbo)', color: 'var(--pt)' }}>
            <p className="font-bold">What is and isn't sent:</p>
            <div className="space-y-1.5">
              <p>
                <span className="font-semibold" style={{ color: '#22c55e' }}>✓ Sent to local backend only:</span>{' '}
                Hand landmark coordinates — 144 numbers per frame. No images, no video.
              </p>
              <p>
                <span className="font-semibold" style={{ color: '#ef4444' }}>✗ Never sent anywhere:</span>{' '}
                Video frames, audio, your face, screen content, or any personally identifiable information.
              </p>
              <p>
                <span className="font-semibold" style={{ color: '#60a5fa' }}>ℹ Chrome sync storage:</span>{' '}
                Only your settings — theme, font size, enabled state. Only you can access this.
              </p>
              <p>
                <span className="font-semibold" style={{ color: '#60a5fa' }}>ℹ Participant captions:</span>{' '}
                Uses the browser's built-in Web Speech API. All processing happens on your device.
              </p>
              <p className="italic pt-0.5" style={{ color: 'var(--pm)' }}>
                The backend runs at localhost:8000 — on your own machine, not the cloud.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── Controls ────────────────────────────────── */}
      <div className="px-4 pt-4 pb-3 flex-shrink-0 flex flex-col gap-3">

        {/* Toggle */}
        <button
          onClick={toggle}
          className="w-full py-3 rounded-2xl text-sm font-bold tracking-wide transition-all duration-150 active:scale-95 shadow-sm"
          style={enabled
            ? { background: 'var(--pbtn)', color: 'var(--pbtnt)' }
            : { background: 'var(--pc)', color: 'var(--pa)', border: '2px solid var(--pbo)' }
          }
        >
          {enabled ? "■  Disable" : "▶  Enable"}
        </button>

        {/* Status row */}
        <div className="flex items-center gap-2">
          <div className="flex-1 flex items-center justify-between rounded-xl px-3 py-2 shadow-sm"
               style={{ background: 'var(--pc)', border: '1px solid var(--pbo)' }}>
            <span className="text-xs font-semibold" style={{ color: 'var(--pm)' }}>Backend</span>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full"
                    style={{ background: wsConnected ? '#4ade80' : 'var(--pbo)' }} />
              <span className="text-xs font-semibold"
                    style={{ color: wsConnected ? '#4ade80' : 'var(--pm)' }}>
                {wsConnected ? "connected" : "offline"}
              </span>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-between rounded-xl px-3 py-2 shadow-sm"
               style={{ background: 'var(--pc)', border: '1px solid var(--pbo)' }}>
            <span className="text-xs font-semibold" style={{ color: 'var(--pm)' }}>Camera</span>
            <span className="text-xs font-semibold" style={{ color:
              detectionState === "no_hand"   ? '#f59e0b'
              : detectionState === "thinking"  ? 'var(--pa)'
              : detectionState === "predicted" ? '#4ade80'
              : 'var(--pm)'
            }}>
              {detectionState === "idle"      ? "idle"
               : detectionState === "no_hand"  ? "no hand"
               : detectionState === "thinking" ? "reading…"
               : "detected"}
            </span>
          </div>
        </div>

        {/* Theme selector */}
        <div className="rounded-xl overflow-hidden flex shadow-sm"
             style={{ border: '1px solid var(--pbo)' }}>
          {THEME_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => saveTheme(opt.value)}
              title={opt.label}
              className="flex-1 py-2 text-xs font-semibold flex items-center justify-center gap-1 transition-colors"
              style={theme === opt.value
                ? { background: 'var(--pbtn)', color: 'var(--pbtnt)' }
                : { background: 'var(--pc)', color: 'var(--pm)' }
              }
            >
              <span>{opt.icon}</span>
              <span>{opt.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Current sentence card ────────────────────── */}
      <div className="px-4 pb-3 flex-shrink-0">
        <div className="rounded-2xl px-5 py-4 min-h-20 flex items-center justify-center"
             style={{ background: 'var(--ps)', border: '1px solid var(--pbo)' }}>
          <SentenceDisplay
            enabled={enabled}
            state={detectionState}
            sentence={currentSentence}
            prediction={lastPrediction}
            theme={theme}
          />
        </div>
      </div>

      {/* ── Transcript ──────────────────────────────── */}
      <div className="flex-1 flex flex-col px-4 pb-4 min-h-0">
        <div className="flex items-center justify-between mb-2 flex-shrink-0">
          <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--pa)' }}>
            Word History
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={clearAll}
              className="text-xs font-medium transition-colors"
              style={{ color: 'var(--pm)' }}
            >
              Clear
            </button>
            <button
              onClick={toggleDemo}
              className="text-xs font-bold px-2.5 py-1 rounded-lg transition-all"
              style={demoActive
                ? { background: '#f59e0b', color: '#000' }
                : { background: 'var(--pc)', color: 'var(--pa)', border: '1px solid var(--pbo)' }
              }
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
              <p className="text-xs" style={{ color: 'var(--pm)' }}>
                No words yet.<br />Interpreted signs will appear here.
              </p>
            </div>
          ) : (
            transcript.map((entry, i) => (
              <TranscriptRow key={entry.id} entry={entry} index={i} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function TranscriptRow({ entry, index }: { entry: TranscriptEntry; index: number }) {
  const pct = Math.round(entry.confidence * 100);
  const confColor = pct >= 85 ? '#4ade80' : pct >= 65 ? 'var(--pa)' : '#f59e0b';

  return (
    <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 shadow-sm"
         style={{ background: 'var(--pc)', border: '1px solid var(--pbo)' }}>
      <span className="text-xs w-5 text-right flex-shrink-0 font-mono" style={{ color: 'var(--pm)' }}>
        {index + 1}
      </span>
      <span className="flex-1 font-semibold capitalize" style={{ color: 'var(--pt)', fontSize: 'var(--fs)' }}>
        {entry.word}
      </span>
      <span className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--pm)' }}>
        {formatTime(entry.timestamp)}
      </span>
      <span className="text-xs font-bold flex-shrink-0 w-8 text-right" style={{ color: confColor }}>
        {pct}%
      </span>
    </div>
  );
}

function SentenceDisplay({
  enabled, state, sentence, prediction, theme,
}: {
  enabled: boolean;
  state: DetectionState;
  sentence: string;
  prediction: LastPrediction | null;
  theme: Theme;
}) {
  const accentColor = theme === 'high-contrast' ? '#ffff00' : theme === 'dark' ? '#a78bfa' : '#7c3aed';
  const mutedColor  = theme === 'high-contrast' ? '#bbbbbb' : theme === 'dark' ? '#9b8fc0' : '#7c6fa0';
  const mainColor   = theme === 'high-contrast' ? '#ffffff' : theme === 'dark' ? '#e8e8ff' : '#1a1a2e';

  if (!enabled || state === "idle") {
    return (
      <div className="text-center">
        <div className="text-2xl mb-1.5 opacity-30">🤟</div>
        <p className="text-xs leading-relaxed" style={{ color: mutedColor }}>
          Enable and open Google Meet or Zoom<br />to start capturing.
        </p>
      </div>
    );
  }

  if (state === "no_hand") {
    return (
      <div className="text-center">
        <div className="text-2xl mb-1.5">✋</div>
        <p className="text-sm font-semibold" style={{ color: mainColor }}>No hand detected</p>
        <p className="text-xs mt-0.5" style={{ color: mutedColor }}>Show your hand to the camera</p>
      </div>
    );
  }

  if (state === "thinking") {
    return (
      <div className="text-center">
        <div className="flex gap-1.5 justify-center mb-2">
          {[0, 150, 300].map((delay) => (
            <span
              key={delay}
              className="w-2 h-2 rounded-full animate-bounce"
              style={{ background: accentColor, animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
        <p className="text-sm font-semibold" style={{ color: accentColor }}>Thinking…</p>
      </div>
    );
  }

  if (state === "predicted") {
    return (
      <div className="text-center w-full">
        {sentence ? (
          <p className="font-bold leading-snug mb-1 capitalize break-words" style={{ color: mainColor, fontSize: 'var(--fs)' }}>
            {sentence}
          </p>
        ) : (
          prediction && (
            <p className="font-bold tracking-wide mb-1 capitalize" style={{ color: mainColor, fontSize: 'var(--fs)' }}>
              {prediction.word}
            </p>
          )
        )}
        {prediction && (
          <span className="text-xs font-semibold" style={{ color: mutedColor }}>
            {Math.round(prediction.confidence * 100)}% confidence
          </span>
        )}
      </div>
    );
  }

  return null;
}
