import { useEffect, useRef, useState } from "react";
import { getSettings, setSettings } from "../lib/storage";
import { sendToActiveTab } from "../lib/messaging";
import type { Theme } from "../lib/storage";

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
// Two themes only. Light = white background with colored (turquoise) buttons.
// Dark = navy blue background with turquoise accents.

const THEME_VARS: Record<Theme, React.CSSProperties> = {
  light: {
    '--pb':   '#ffffff',          // page background
    '--ph':   '#ffffff',          // header background
    '--pc':   '#ffffff',          // card background (neutral)
    '--ps':   '#f0fbfa',          // pale soft fill
    '--pfill':'rgba(236,250,248,0.94)', // overlay transcript (--peer-bg) tint for boxes
    '--pt':   '#0f2e2b',          // primary text
    '--pm':   '#5e8a86',          // muted text
    '--pa':   '#0d9488',          // accent (turquoise)
    '--pbo':  'rgba(13,148,136,0.24)', // border
    '--pbtn': '#0d9488',          // active button bg
    '--pbtnt':'#ffffff',          // active button text
  } as React.CSSProperties,
  dark: {
    '--pb':   '#0a1a2f',          // navy blue
    '--ph':   '#0c2138',
    '--pc':   '#102a45',
    '--ps':   '#0d2236',
    '--pfill':'#0c2238',          // soft tint for boxes
    '--pt':   '#e6f6f4',
    '--pm':   '#7ba8b3',
    '--pa':   '#2dd4bf',          // turquoise
    '--pbo':  'rgba(45,212,191,0.20)',
    '--pbtn': '#14b8a6',
    '--pbtnt':'#052e29',
  } as React.CSSProperties,
};

// ── Constants ─────────────────────────────────────────────────────────────────

const DEMO_WORDS = [
  "hello", "thank you", "yes", "please", "help",
  "water", "good morning", "more", "stop", "go",
  "sorry", "no", "finished", "name", "nice to meet you",
];

let entryId = 0;

// Older builds may have stored the removed 'high-contrast' theme — fold it to dark.
function coerceTheme(t: unknown): Theme {
  return t === 'light' ? 'light' : 'dark';
}

// Font size is controlled from the overlay; here we only read it. Coerce to a
// valid number so a legacy string/NaN value can't break the --fs CSS variable.
function toFontSize(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(32, Math.max(8, Math.round(n))) : 20;
}

// ── Icons (sun / moon for the theme toggle) ─────────────────────────────────────

function SunIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Popup() {
  const [enabled, setEnabled]               = useState(false);
  const [loading, setLoading]               = useState(true);
  const [transcript, setTranscript]         = useState<TranscriptEntry[]>([]);
  const [demoActive, setDemoActive]         = useState(false);
  const [theme, setThemeState]              = useState<Theme>('dark');
  const [fontSize, setFontSizeState]        = useState<number>(20);
  const [privacyOpen, setPrivacyOpen]       = useState(false);
  const [voiceEnabled, setVoiceEnabled]     = useState(false);
  const [voiceURI, setVoiceURI]             = useState('');
  const [speechRate, setSpeechRate]         = useState(1.0);
  const [speechPitch, setSpeechPitch]       = useState(1.0);
  const [voices, setVoices]                 = useState<SpeechSynthesisVoice[]>([]);
  const [voiceOpen, setVoiceOpen]           = useState(false);

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
    getSettings().then((settings) => {
      setEnabled(settings.enabled);
      setThemeState(coerceTheme(settings.theme));
      setFontSizeState(toFontSize(settings.fontSize));
      setVoiceEnabled(settings.voiceEnabled);
      setVoiceURI(settings.voiceURI);
      setSpeechRate(settings.speechRate);
      setSpeechPitch(settings.speechPitch);
      setLoading(false);
    });

    function loadVoices() {
      const list = speechSynthesis.getVoices();
      if (list.length > 0) setVoices(list);
    }
    loadVoices();
    speechSynthesis.addEventListener('voiceschanged', loadVoices);

    const listener = (changes: Record<string, chrome.storage.StorageChange>) => {
      if (changes.enabled)  setEnabled(changes.enabled.newValue);
      if (changes.theme)    setThemeState(coerceTheme(changes.theme.newValue));
      if (changes.fontSize) setFontSizeState(toFontSize(changes.fontSize.newValue));
      if (changes.lastPrediction) {
        const p = changes.lastPrediction.newValue as LastPrediction | undefined;
        if (p) {
          sentenceRef.current = sentenceRef.current ? sentenceRef.current + ' ' + p.word : p.word;
          setTranscript((prev) => [
            ...prev,
            { id: entryId++, word: p.word, confidence: p.confidence, timestamp: new Date() },
          ]);
        }
      }
    };

    chrome.storage.onChanged.addListener(listener);
    return () => {
      chrome.storage.onChanged.removeListener(listener);
      speechSynthesis.removeEventListener('voiceschanged', loadVoices);
    };
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

  async function saveVoiceEnabled(next: boolean) {
    setVoiceEnabled(next);
    await setSettings({ voiceEnabled: next });
  }

  async function saveVoiceURI(uri: string) {
    setVoiceURI(uri);
    await setSettings({ voiceURI: uri });
    window.postMessage({ type: 'ASL_SET_VOICE_PREFS', voiceURI: uri, rate: speechRate, pitch: speechPitch }, '*');
  }

  async function saveSpeechRate(rate: number) {
    setSpeechRate(rate);
    await setSettings({ speechRate: rate });
    window.postMessage({ type: 'ASL_SET_VOICE_PREFS', voiceURI, rate, pitch: speechPitch }, '*');
  }

  async function saveSpeechPitch(pitch: number) {
    setSpeechPitch(pitch);
    await setSettings({ speechPitch: pitch });
    window.postMessage({ type: 'ASL_SET_VOICE_PREFS', voiceURI, rate: speechRate, pitch }, '*');
  }

  function clearAll() {
    sentenceRef.current = '';
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
        sentenceRef.current = sentenceRef.current ? sentenceRef.current + ' ' + word : word;
        setTranscript((prev) => [
          ...prev,
          { id: entryId++, word, confidence, timestamp: new Date(), demo: true },
        ]);
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
      className="h-screen flex flex-col overflow-hidden"
      style={{ width: 380, background: 'var(--pb)', color: 'var(--pt)', '--fs': `${fontSize}pt`, ...tv } as React.CSSProperties}
    >

      {/* ── Header: extension name + theme circles ───────────── */}
      <div className="px-6 pt-6 pb-4 flex-shrink-0" style={{ background: 'var(--ph)' }}>
        <div className="flex items-center justify-between gap-3">
          <h1 className="font-serif-body leading-tight" style={{ color: 'var(--pa)', fontSize: '28px', fontWeight: 700 }}>
            ASL Interpreter
          </h1>

          {/* Theme toggle — two circles, sun (light) / moon (dark) */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => saveTheme('light')}
              aria-pressed={theme === 'light'}
              aria-label="Light theme"
              title="Light"
              className="w-9 h-9 rounded-full flex items-center justify-center transition-all"
              style={theme === 'light'
                ? { background: 'var(--pbtn)', color: 'var(--pbtnt)' }
                : { background: 'var(--ps)', color: 'var(--pm)', border: '1px solid var(--pbo)' }
              }
            >
              <SunIcon />
            </button>
            <button
              onClick={() => saveTheme('dark')}
              aria-pressed={theme === 'dark'}
              aria-label="Dark theme"
              title="Dark"
              className="w-9 h-9 rounded-full flex items-center justify-center transition-all"
              style={theme === 'dark'
                ? { background: 'var(--pbtn)', color: 'var(--pbtnt)' }
                : { background: 'var(--ps)', color: 'var(--pm)', border: '1px solid var(--pbo)' }
              }
            >
              <MoonIcon />
            </button>
          </div>
        </div>

        <button
          onClick={() => setPrivacyOpen((v) => !v)}
          className="mt-1 text-sm underline-offset-2 hover:underline"
          style={{ color: 'var(--pm)' }}
        >
          Privacy &amp; data policy
        </button>

        {/* Privacy panel */}
        {privacyOpen && (
          <div className="mt-3 rounded-xl px-4 py-3 text-sm leading-relaxed space-y-2"
               style={{ background: 'var(--pc)', border: '1px solid var(--pbo)', color: 'var(--pt)' }}>
            <p className="font-semibold">What is and isn't sent:</p>
            <div className="space-y-1.5">
              <p>
                <span className="font-semibold" style={{ color: '#22c55e' }}>Sent to local backend only:</span>{' '}
                Hand landmark coordinates — 144 numbers per frame. No images, no video.
              </p>
              <p>
                <span className="font-semibold" style={{ color: '#ef4444' }}>Never sent anywhere:</span>{' '}
                Video frames, audio, your face, screen content, or any personally identifiable information.
              </p>
              <p>
                <span className="font-semibold" style={{ color: 'var(--pa)' }}>Chrome sync storage:</span>{' '}
                Only your settings — theme, font size, enabled state. Only you can access this.
              </p>
              <p>
                <span className="font-semibold" style={{ color: 'var(--pa)' }}>Participant captions:</span>{' '}
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
      <div className="px-5 pt-2 pb-3 flex-shrink-0 flex flex-col gap-3">

        {/* Enable / Disable */}
        <button
          onClick={toggle}
          aria-pressed={enabled}
          className="w-full py-3.5 rounded-2xl text-lg font-semibold tracking-wide transition-all duration-150 active:scale-95 shadow-sm"
          style={enabled
            ? { background: 'var(--pbtn)', color: 'var(--pbtnt)' }
            : { background: 'var(--pc)', color: 'var(--pa)', border: '2px solid var(--pbo)' }
          }
        >
          {enabled ? "Disable" : "Enable"}
        </button>

        {/* Voice settings accordion — collapsible even when voice is on */}
        <div className="rounded-xl overflow-hidden shadow-sm" style={{ border: '1px solid var(--pbo)' }}>

          {/* Header row: title (toggles collapse) + on/off switch */}
          <div className="flex items-center px-4 py-3" style={{ background: 'var(--pfill)' }}>
            <button
              onClick={() => setVoiceOpen((v) => !v)}
              aria-expanded={voiceOpen}
              className="flex items-center gap-2 flex-1 min-w-0"
              style={{ color: 'var(--pt)' }}
            >
              <span className="text-base font-semibold tracking-wide">Voice Output</span>
              <span
                className="transition-transform"
                style={{
                  color: 'var(--pm)',
                  fontSize: '12px',
                  transform: voiceOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                }}
              >
                ›
              </span>
            </button>

            {/* On/off switch */}
            <button
              onClick={() => {
                const next = !voiceEnabled;
                saveVoiceEnabled(next);
                if (next) setVoiceOpen(true);
              }}
              aria-pressed={voiceEnabled}
              aria-label="Toggle voice output"
              className="relative w-11 h-6 rounded-full transition-colors flex-shrink-0"
              style={{ background: voiceEnabled ? 'var(--pa)' : 'var(--pbo)' }}
            >
              <span
                className="absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-all"
                style={{ left: voiceEnabled ? '22px' : '2px' }}
              />
            </button>
          </div>

          {voiceOpen && (
            <div className="px-4 pb-4 pt-3 flex flex-col gap-3"
                 style={{ background: 'var(--pfill)', borderTop: '1px solid var(--pbo)' }}>

              {/* Voice picker */}
              <div className="flex flex-col gap-1">
                <label className="text-sm font-semibold" style={{ color: 'var(--pm)' }}>
                  Voice
                </label>
                <select
                  value={voiceURI}
                  onChange={(e) => saveVoiceURI(e.target.value)}
                  className="w-full rounded-lg px-2 py-2 text-sm"
                  style={{
                    background: 'var(--pb)',
                    color: 'var(--pt)',
                    border: '1px solid var(--pbo)',
                    outline: 'none',
                    fontFamily: 'var(--font-serif)',
                  }}
                  aria-label="Select voice"
                >
                  <option value="">Default voice</option>
                  {voices.map((v) => (
                    <option key={v.voiceURI} value={v.voiceURI}>
                      {v.name} ({v.lang})
                    </option>
                  ))}
                </select>
              </div>

              {/* Rate slider */}
              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold" style={{ color: 'var(--pm)' }}>Speed</label>
                  <span className="text-sm" style={{ color: 'var(--pa)', fontVariantNumeric: 'tabular-nums' }}>{speechRate.toFixed(1)}×</span>
                </div>
                <input
                  type="range"
                  min={0.5} max={2.0} step={0.1}
                  value={speechRate}
                  onChange={(e) => saveSpeechRate(parseFloat(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                  style={{
                    accentColor: 'var(--pa)',
                    background: `linear-gradient(to right, var(--pa) ${((speechRate - 0.5) / 1.5) * 100}%, var(--pbo) ${((speechRate - 0.5) / 1.5) * 100}%)`,
                  }}
                  aria-label="Speech rate"
                />
                <div className="flex justify-between text-xs" style={{ color: 'var(--pm)' }}>
                  <span>0.5× slow</span><span>2.0× fast</span>
                </div>
              </div>

              {/* Pitch slider */}
              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold" style={{ color: 'var(--pm)' }}>Pitch</label>
                  <span className="text-sm" style={{ color: 'var(--pa)', fontVariantNumeric: 'tabular-nums' }}>{speechPitch.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={0.5} max={1.5} step={0.1}
                  value={speechPitch}
                  onChange={(e) => saveSpeechPitch(parseFloat(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                  style={{
                    accentColor: 'var(--pa)',
                    background: `linear-gradient(to right, var(--pa) ${((speechPitch - 0.5) / 1.0) * 100}%, var(--pbo) ${((speechPitch - 0.5) / 1.0) * 100}%)`,
                  }}
                  aria-label="Speech pitch"
                />
                <div className="flex justify-between text-xs" style={{ color: 'var(--pm)' }}>
                  <span>0.5 low</span><span>1.5 high</span>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>

      {/* ── Sentence history ────────────────────────────── */}
      <div className="flex-1 flex flex-col px-5 pb-5 min-h-0">
        <div className="flex items-center justify-between mb-2 flex-shrink-0">
          <span className="text-base font-semibold tracking-wide" style={{ color: 'var(--pa)' }}>
            Word History
          </span>
          <div className="flex items-center gap-3">
            <button
              onClick={clearAll}
              className="text-sm transition-colors hover:underline underline-offset-2"
              style={{ color: 'var(--pm)' }}
            >
              Clear
            </button>
            <button
              onClick={toggleDemo}
              className="text-sm font-semibold px-3 py-1 rounded-lg transition-all"
              style={demoActive
                ? { background: '#f59e0b', color: '#1a1a1a' }
                : { background: 'var(--pc)', color: 'var(--pa)', border: '1px solid var(--pbo)' }
              }
            >
              {demoActive ? "Stop Demo" : "Demo"}
            </button>
          </div>
        </div>

        <div
          ref={scrollRef}
          role="log"
          aria-live="polite"
          aria-label="Word history"
          className="flex-1 overflow-y-auto flex flex-col gap-1.5 pr-0.5"
          style={{ scrollBehavior: "smooth" }}
        >
          {transcript.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-6">
              <p className="text-sm" style={{ color: 'var(--pm)' }}>
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
  const confColor = pct >= 85 ? '#22c55e' : pct >= 65 ? 'var(--pa)' : '#f59e0b';

  return (
    <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 shadow-sm"
         style={{ background: 'var(--pfill)', border: '1px solid var(--pbo)' }}>
      <span className="text-sm w-5 text-right flex-shrink-0" style={{ color: 'var(--pm)', fontVariantNumeric: 'tabular-nums' }}>
        {index + 1}
      </span>
      <span className="flex-1 text-base font-semibold capitalize" style={{ color: 'var(--pt)' }}>
        {entry.word}
      </span>
      <span className="text-sm flex-shrink-0" style={{ color: 'var(--pm)', fontVariantNumeric: 'tabular-nums' }}>
        {formatTime(entry.timestamp)}
      </span>
      <span className="text-sm font-semibold flex-shrink-0 w-10 text-right" style={{ color: confColor, fontVariantNumeric: 'tabular-nums' }}>
        {pct}%
      </span>
    </div>
  );
}
