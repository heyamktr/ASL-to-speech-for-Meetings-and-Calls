import { useState, useEffect, useRef, useCallback } from 'react';
import type { Theme } from '../../lib/storage';

// ── Types ────────────────────────────────────────────────────────────────────

interface SentenceBlock {
  id: number;
  gloss: string;        // raw space-joined signs, shown while the sentence is refining
  sentence: string;     // the rewritten natural sentence ('' until resolved)
  refining: boolean;    // true while the /refine call is in flight
  timestamp: Date;
}

interface PeerEntry {
  id: number;
  text: string;
  timestamp: Date;
  isFinal: boolean;
}

export interface OverlayCallbacks {
  addWord: (word: string) => void;
  beginBlock: (gloss: string) => number;
  resolveBlock: (id: number, sentence: string) => void;
  clear: () => void;
}

interface Props {
  onReady: (cbs: OverlayCallbacks) => void;
  onClose: () => void;
  onSpeak: () => void;
}

const MIN_W = 220;
const MIN_H = 160;
const DEFAULT_W = 360;
const DEFAULT_H = 420;
const MIN_FONT = 8;
const MAX_FONT = 32;
const DEFAULT_FONT = 20;
const MIN_PEER_H = 70;       // min height of the participants transcript
const MIN_ASL_H = 90;        // keep at least this much for the ASL area
const DEFAULT_PEER_H = 150;

// Older builds may have stored the removed 'high-contrast' theme — fold it to dark.
function coerceTheme(t: unknown): Theme {
  return t === 'light' ? 'light' : 'dark';
}

// Coerce a stored value (which may be a string or null from older builds) into a
// valid clamped font size — guards against the "NaNpt" corruption.
function toFontSize(v: unknown): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return DEFAULT_FONT;
  return Math.min(MAX_FONT, Math.max(MIN_FONT, Math.round(n)));
}

// Coerce a stored number, falling back when it's a string/NaN/null from an older
// build. Guards the panel geometry the same way toFontSize guards the font.
function toNum(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function fmtTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

let _blockId = 0;
let _peerId = 0;

// ── Component ─────────────────────────────────────────────────────────────────

export function OverlayApp({ onReady, onClose, onSpeak }: Props) {
  const [theme, setTheme] = useState<Theme>('dark');
  const [fontSize, setFontSize] = useState(DEFAULT_FONT);
  const [pos, setPos] = useState({ x: 20, y: 20 });
  const [size, setSize] = useState({ w: DEFAULT_W, h: DEFAULT_H });
  const [peerH, setPeerH] = useState(DEFAULT_PEER_H);
  const [pastBlocks, setPastBlocks] = useState<SentenceBlock[]>([]);
  const [currentWords, setCurrentWords] = useState<string[]>([]);
  const [peerEntries, setPeerEntries] = useState<PeerEntry[]>([]);
  const [speechActive, setSpeechActive] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);

  const currentWordsRef = useRef<string[]>([]);
  // refs for direct DOM manipulation — avoids React re-renders during drag/resize
  const panelRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef({ active: false, ox: 0, oy: 0 });
  const resizeRef = useRef({ active: false, startX: 0, startY: 0, startW: 0, startH: 0 });
  const peerResizeRef = useRef({ active: false, startY: 0, startH: 0 });
  const posRef = useRef(pos);
  const sizeRef = useRef(size);
  const peerHRef = useRef(peerH);
  const aslScrollRef = useRef<HTMLDivElement>(null);
  const peerScrollRef = useRef<HTMLDivElement>(null);

  posRef.current = pos;
  sizeRef.current = size;
  peerHRef.current = peerH;

  // Load settings
  useEffect(() => {
    chrome.storage.sync.get(
      { theme: 'dark', fontSize: DEFAULT_FONT, overlayX: 20, overlayY: 20, overlayW: DEFAULT_W, overlayH: DEFAULT_H, overlayPeerH: DEFAULT_PEER_H },
      (r) => {
        setTheme(coerceTheme(r.theme));

        // Every enable starts a fresh session at the default size — the previous
        // session's A−/A+ adjustments are intentionally not carried over.
        setFontSize(DEFAULT_FONT);
        if (r.fontSize !== DEFAULT_FONT) chrome.storage.sync.set({ fontSize: DEFAULT_FONT });

        // Coerce + clamp geometry so a corrupt/off-screen stored value can never
        // strand the panel where it can't be seen or recovered.
        const w = Math.max(MIN_W, Math.min(window.innerWidth,  toNum(r.overlayW, DEFAULT_W)));
        const h = Math.max(MIN_H, Math.min(window.innerHeight, toNum(r.overlayH, DEFAULT_H)));
        const x = Math.max(0, Math.min(window.innerWidth  - w, toNum(r.overlayX, 20)));
        const y = Math.max(0, Math.min(window.innerHeight - h, toNum(r.overlayY, 20)));
        setSize({ w, h });
        setPos({ x, y });

        // Participants transcript height — clamp so it can't crowd out the ASL area.
        const ph = Math.max(MIN_PEER_H, Math.min(h - MIN_ASL_H, toNum(r.overlayPeerH, DEFAULT_PEER_H)));
        setPeerH(ph);

        // Heal any previously-corrupted (string/NaN/off-screen) stored geometry.
        if (w !== r.overlayW || h !== r.overlayH || x !== r.overlayX || y !== r.overlayY) {
          chrome.storage.sync.set({ overlayW: w, overlayH: h, overlayX: x, overlayY: y });
        }
      }
    );

    const handler = (changes: Record<string, chrome.storage.StorageChange>) => {
      if (changes.theme)    setTheme(coerceTheme(changes.theme.newValue));
      if (changes.fontSize) setFontSize(toFontSize(changes.fontSize.newValue));
    };
    chrome.storage.sync.onChanged.addListener(handler);
    return () => chrome.storage.sync.onChanged.removeListener(handler);
  }, []);

  // Register callbacks
  useEffect(() => {
    onReady({
      addWord: (word) => {
        currentWordsRef.current = [...currentWordsRef.current, word];
        setCurrentWords([...currentWordsRef.current]);
      },
      // Move the live gloss into a finalized block (shown as "refining…") and clear
      // the live line so a new utterance can start. Returns the block id.
      beginBlock: (gloss) => {
        const id = _blockId++;
        setPastBlocks((prev) => [
          ...prev,
          { id, gloss, sentence: '', refining: true, timestamp: new Date() },
        ]);
        currentWordsRef.current = [];
        setCurrentWords([]);
        return id;
      },
      // Swap the block's text for the rewritten sentence and stop its spinner.
      resolveBlock: (id, sentence) => {
        setPastBlocks((prev) =>
          prev.map((b) => (b.id === id ? { ...b, sentence, refining: false } : b)),
        );
      },
      clear: () => {
        currentWordsRef.current = [];
        setCurrentWords([]);
      },
    });
  }, [onReady]);

  // Auto-scroll
  useEffect(() => {
    if (aslScrollRef.current) aslScrollRef.current.scrollTop = aslScrollRef.current.scrollHeight;
  }, [pastBlocks, currentWords]);

  useEffect(() => {
    if (peerScrollRef.current) peerScrollRef.current.scrollTop = peerScrollRef.current.scrollHeight;
  }, [peerEntries]);

  // Web Speech API — captions whatever audio the microphone hears (participants)
  useEffect(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) {
      setSpeechError('Speech recognition not supported in this browser.');
      return;
    }

    const recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    let interimId: number | null = null;
    let stopped = false;   // set on cleanup — don't restart a torn-down recognizer
    let fatal = false;     // mic permission/capture error — stop retrying, surface it
    let restartTimer: ReturnType<typeof setTimeout> | null = null;

    recognition.onstart = () => { setSpeechActive(true); setSpeechError(null); };
    recognition.onend = () => {
      setSpeechActive(false);
      if (stopped || fatal || document.hidden) return;
      // brief backoff so a failing start() can't spin in a tight loop
      restartTimer = setTimeout(() => {
        try { recognition.start(); } catch (_) { /* already started */ }
      }, 400);
    };

    recognition.onresult = (event: any) => {
      const results: SpeechRecognitionResultList = event.results;
      let interimText = '';

      // A single line per utterance: it shows italic while interim, then the SAME
      // line (same id, same position) switches to normal once finalized — no
      // duplicate "hearing" + "confirmed" pair.
      for (let i = event.resultIndex; i < results.length; i++) {
        const result = results[i];
        const text = result[0].transcript.trim();
        if (!text) continue;
        if (result.isFinal) {
          const finalText = text;
          const activeId = interimId;
          setPeerEntries((prev) => {
            if (activeId !== null && prev.some((e) => e.id === activeId)) {
              return prev
                .map((e) => (e.id === activeId ? { ...e, text: finalText, isFinal: true, timestamp: new Date() } : e))
                .slice(-20);
            }
            return [...prev, { id: _peerId++, text: finalText, timestamp: new Date(), isFinal: true }].slice(-20);
          });
          interimId = null;
        } else {
          interimText += text + ' ';
        }
      }

      if (interimText.trim()) {
        const id = interimId ?? (interimId = _peerId++);
        const text = interimText.trim();
        setPeerEntries((prev) => {
          if (prev.some((e) => e.id === id)) {
            return prev.map((e) => (e.id === id ? { ...e, text, timestamp: new Date() } : e));
          }
          return [...prev, { id, text, timestamp: new Date(), isFinal: false }].slice(-20);
        });
      }
    };

    recognition.onerror = (ev: any) => {
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        fatal = true;
        setSpeechError('Microphone blocked — allow mic access for participant captions.');
      } else if (ev.error === 'audio-capture') {
        fatal = true;
        setSpeechError('No microphone available for captions.');
      } else if (ev.error !== 'no-speech' && ev.error !== 'aborted') {
        console.warn('[ASL Overlay] SpeechRecognition error:', ev.error);
      }
    };

    try { recognition.start(); } catch (_) { /* ignore */ }
    return () => {
      stopped = true;
      if (restartTimer) clearTimeout(restartTimer);
      try { recognition.stop(); } catch (_) { /* ignore */ }
    };
  }, []);

  // ── Drag — bypass React state during move to eliminate lag ────────────────

  const onDragPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      active: true,
      ox: e.clientX - posRef.current.x,
      oy: e.clientY - posRef.current.y,
    };
  }, []);

  const onDragPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.active) return;
    const x = Math.max(0, Math.min(window.innerWidth - sizeRef.current.w, e.clientX - dragRef.current.ox));
    const y = Math.max(0, Math.min(window.innerHeight - sizeRef.current.h, e.clientY - dragRef.current.oy));
    // Update DOM directly — no React re-render
    if (panelRef.current) {
      panelRef.current.style.left = `${x}px`;
      panelRef.current.style.top  = `${y}px`;
    }
    posRef.current = { x, y };
  }, []);

  const onDragPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.active) return;
    dragRef.current.active = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
    const { x, y } = posRef.current;
    setPos({ x, y });
    chrome.storage.sync.set({ overlayX: x, overlayY: y });
  }, []);

  // ── Resize — same direct-DOM approach ─────────────────────────────────────

  const onResizePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    resizeRef.current = {
      active: true,
      startX: e.clientX,
      startY: e.clientY,
      startW: sizeRef.current.w,
      startH: sizeRef.current.h,
    };
  }, []);

  const onResizePointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!resizeRef.current.active) return;
    const { startX, startY, startW, startH } = resizeRef.current;
    const w = Math.max(MIN_W, startW + (e.clientX - startX));
    const h = Math.max(MIN_H, startH + (e.clientY - startY));
    if (panelRef.current) {
      panelRef.current.style.width  = `${w}px`;
      panelRef.current.style.height = `${h}px`;
    }
    sizeRef.current = { w, h };
  }, []);

  const onResizePointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!resizeRef.current.active) return;
    resizeRef.current.active = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
    const { w, h } = sizeRef.current;
    setSize({ w, h });
    chrome.storage.sync.set({ overlayW: w, overlayH: h });
  }, []);

  // ── Divider drag — resize the participants transcript vs. the ASL area ─────

  const peerMax = () => Math.max(MIN_PEER_H, sizeRef.current.h - MIN_ASL_H);

  const onDividerPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    peerResizeRef.current = { active: true, startY: e.clientY, startH: peerHRef.current };
  }, []);

  const onDividerPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!peerResizeRef.current.active) return;
    // Dragging up grows the participants area, dragging down shrinks it.
    const delta = peerResizeRef.current.startY - e.clientY;
    const h = Math.max(MIN_PEER_H, Math.min(peerMax(), peerResizeRef.current.startH + delta));
    if (peerScrollRef.current) peerScrollRef.current.style.height = `${h}px`;
    peerHRef.current = h;
  }, []);

  const onDividerPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!peerResizeRef.current.active) return;
    peerResizeRef.current.active = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
    const h = peerHRef.current;
    setPeerH(h);
    chrome.storage.sync.set({ overlayPeerH: h });
  }, []);

  function adjustFontSize(delta: number) {
    setFontSize((prev) => {
      const next = toFontSize(toFontSize(prev) + delta);
      chrome.storage.sync.set({ fontSize: next });
      return next;
    });
  }

  const currentSentence = currentWords.join(' ');
  // Clamp the stored peer height against the live panel height each render.
  const clampedPeerH = Math.max(MIN_PEER_H, Math.min(size.h - MIN_ASL_H, peerH));

  return (
    <div
      ref={panelRef}
      className="panel"
      data-theme={theme}
      style={{ left: pos.x, top: pos.y, width: size.w, height: size.h, fontSize, ['--fs' as any]: `${fontSize}px` }}
    >
      {/* ── Header ──────────────────────────────────── */}
      <div
        className="header"
        onPointerDown={onDragPointerDown}
        onPointerMove={onDragPointerMove}
        onPointerUp={onDragPointerUp}
      >
        <span className="header-title">ASL Captions</span>
        <div className="header-actions" onPointerDown={(e) => e.stopPropagation()}>
          <button className="icon-btn" onClick={() => adjustFontSize(-1)} title="Decrease text size">A−</button>
          <span className="font-size-label">{fontSize}pt</span>
          <button className="icon-btn" onClick={() => adjustFontSize(1)} title="Increase text size">A+</button>
          <button className="icon-btn" onClick={onClose} title="Close overlay">✕</button>
        </div>
      </div>

      {/* ── ASL sentence area ────────────────────────── */}
      <div className="asl-section" ref={aslScrollRef}>
        {pastBlocks.length === 0 && currentWords.length === 0 && (
          <span className="empty-hint">ASL captions will appear here…</span>
        )}

        {pastBlocks.map((block) => (
          <div key={block.id} className="sentence-block">
            <div className="sentence-timestamp">{fmtTime(block.timestamp)}</div>
            <div className="sentence-text">
              {block.refining ? block.gloss : (block.sentence || block.gloss)}
              {block.refining && <span className="refining-tag"> · refining…</span>}
            </div>
          </div>
        ))}

        <div className="current-block">
          <div className="current-text">
            {currentSentence || <span className="empty-hint">Waiting for signs…</span>}
            {currentWords.length > 0 && <span className="cursor" />}
          </div>
          <button
            className="speak-btn"
            onClick={onSpeak}
            disabled={currentWords.length === 0}
            title="Finish this sentence and speak it now"
          >
            Speak
          </button>
        </div>
      </div>

      {/* ── Divider — drag to resize the participants transcript ──────── */}
      <div
        className="section-divider"
        onPointerDown={onDividerPointerDown}
        onPointerMove={onDividerPointerMove}
        onPointerUp={onDividerPointerUp}
        title="Drag to resize participants transcript"
      >
        <span className="divider-grip" />
      </div>

      {/* ── Peer transcript ──────────────────────────── */}
      <div className="peer-section" ref={peerScrollRef} style={{ height: clampedPeerH }}>
        <div className="peer-header">
          <span className="peer-header-label">
            {speechActive && <span className="peer-dot" />}
            Participants
          </span>
          <button
            className="peer-clear-btn"
            onClick={() => setPeerEntries([])}
            disabled={peerEntries.length === 0}
            title="Clear participant transcript"
          >
            Clear
          </button>
        </div>

        {peerEntries.length === 0 ? (
          <div className="peer-empty">{speechError ?? 'Listening for participants…'}</div>
        ) : (
          peerEntries.map((entry) => (
            <div key={entry.id} className={`peer-entry${entry.isFinal ? '' : ' interim'}`}>
              <span className="peer-timestamp">{fmtTime(entry.timestamp)}</span>
              {entry.text}
            </div>
          ))
        )}
      </div>

      {/* ── Resize handle ────────────────────────────── */}
      <div
        className="resize-handle"
        onPointerDown={onResizePointerDown}
        onPointerMove={onResizePointerMove}
        onPointerUp={onResizePointerUp}
      />
    </div>
  );
}
