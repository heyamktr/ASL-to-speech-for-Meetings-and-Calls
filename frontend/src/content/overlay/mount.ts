import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { OverlayApp } from './OverlayApp';
import type { OverlayCallbacks } from './OverlayApp';

// ── Inline CSS for the shadow DOM (no access to external stylesheets) ─────────

const OVERLAY_CSS = `
* { box-sizing: border-box; margin: 0; padding: 0; }
button { cursor: pointer; border: none; background: none; font-family: inherit; }

.panel {
  position: fixed;
  z-index: 2147483647;
  display: flex;
  flex-direction: column;
  border-radius: 14px;
  overflow: clip; /* clips children to border-radius without affecting absolute resize handle */
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  min-width: 220px;
  min-height: 160px;
}

/* ── Themes ──────────────────────────────────────────────────── */

.panel[data-theme="dark"] {
  background: rgba(12, 12, 24, 0.93);
  color: #e8e8ff;
  border: 1px solid rgba(255,255,255,0.09);
  box-shadow: 0 12px 40px rgba(0,0,0,0.75), 0 0 0 0.5px rgba(255,255,255,0.05);
  --accent: #a78bfa;
  --muted: #6b6b8a;
  --header-bg: rgba(18, 14, 36, 0.98);
  --peer-bg: rgba(16, 20, 44, 0.9);
  --btn-bg: rgba(255,255,255,0.07);
  --btn-hover: rgba(255,255,255,0.14);
  --divider: rgba(255,255,255,0.07);
  --block-bg: rgba(255,255,255,0.04);
}

.panel[data-theme="light"] {
  background: rgba(255,255,255,0.95);
  color: #1a1a2e;
  border: 1px solid rgba(0,0,0,0.10);
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
  --accent: #7c3aed;
  --muted: #8b8ba7;
  --header-bg: rgba(248,246,255,0.98);
  --peer-bg: rgba(240,242,255,0.92);
  --btn-bg: rgba(0,0,0,0.04);
  --btn-hover: rgba(0,0,0,0.09);
  --divider: rgba(0,0,0,0.07);
  --block-bg: rgba(0,0,0,0.03);
}

.panel[data-theme="high-contrast"] {
  background: #000;
  color: #fff;
  border: 2px solid #fff;
  box-shadow: 0 0 0 2px #fff;
  --accent: #ffff00;
  --muted: #aaa;
  --header-bg: #111;
  --peer-bg: #0a0a0a;
  --btn-bg: rgba(255,255,255,0.10);
  --btn-hover: rgba(255,255,255,0.22);
  --divider: rgba(255,255,255,0.25);
  --block-bg: rgba(255,255,255,0.06);
}

/* ── Header ──────────────────────────────────────────────────── */

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 11px;
  background: var(--header-bg);
  border-bottom: 1px solid var(--divider);
  cursor: grab;
  flex-shrink: 0;
  user-select: none;
  gap: 8px;
}
.header:active { cursor: grabbing; }

.header-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
}

.header-actions { display: flex; align-items: center; gap: 4px; }

.font-size-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted);
  min-width: 26px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.icon-btn {
  width: 22px;
  height: 22px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--btn-bg);
  color: inherit;
  font-size: 11px;
  transition: background 0.12s;
  padding: 0;
  line-height: 1;
}
.icon-btn:hover { background: var(--btn-hover); }

/* ── ASL caption section ─────────────────────────────────────── */

.asl-section {
  flex: 1;
  overflow-y: auto;
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 60px;
}

.sentence-block {
  animation: fade-up 0.18s ease;
}

.sentence-timestamp {
  font-size: 0.72em;
  color: var(--muted);
  margin-bottom: 1px;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
}

.sentence-text {
  font-size: var(--fs);
  font-weight: 600;
  line-height: 1.45;
  letter-spacing: 0.01em;
  word-spacing: 0.05em;
}

.current-block {
  padding: 7px 9px;
  background: var(--block-bg);
  border-radius: 8px;
  border-left: 3px solid var(--accent);
  flex-shrink: 0;
}

.current-text {
  font-size: var(--fs);
  font-weight: 700;
  line-height: 1.5;
  min-height: 1.5em;
  word-spacing: 0.08em;
  letter-spacing: 0.01em;
}

.cursor {
  display: inline-block;
  width: 2px;
  height: 0.9em;
  background: var(--accent);
  margin-left: 3px;
  vertical-align: middle;
  animation: blink 1s step-end infinite;
}

.empty-hint {
  color: var(--muted);
  font-style: italic;
  font-size: 0.85em;
}

/* ── Section divider ─────────────────────────────────────────── */

.section-divider {
  height: 1px;
  background: var(--divider);
  flex-shrink: 0;
}

/* ── Peer transcript section ─────────────────────────────────── */

.peer-section {
  flex-shrink: 0;
  background: var(--peer-bg);
  overflow-y: auto;
  padding: 7px 12px 9px;
  min-height: 60px;
  max-height: 40%;
}

.peer-header {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 5px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 5px;
}

.peer-header-label {
  display: flex;
  align-items: center;
  gap: 5px;
}

.peer-clear-btn {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--btn-bg);
  padding: 2px 7px;
  border-radius: 5px;
  transition: background 0.12s, color 0.12s;
}
.peer-clear-btn:hover:not(:disabled) { background: var(--btn-hover); color: var(--accent); }
.peer-clear-btn:disabled { opacity: 0.35; cursor: default; }

.peer-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #22c55e;
  animation: pulse 2s infinite;
  flex-shrink: 0;
}

.peer-entry {
  font-size: calc(var(--fs) * 0.875);
  line-height: 1.45;
  margin-bottom: 3px;
  word-spacing: 0.04em;
}

.peer-entry.interim {
  opacity: 0.5;
  font-style: italic;
}

.peer-timestamp {
  font-size: 0.78em;
  color: var(--muted);
  margin-right: 4px;
  font-variant-numeric: tabular-nums;
}

.peer-empty {
  font-size: 0.82em;
  color: var(--muted);
  font-style: italic;
  text-align: center;
  padding: 6px 0;
}

/* ── Resize handle ───────────────────────────────────────────── */

.resize-handle {
  position: absolute;
  bottom: 0;
  right: 0;
  width: 18px;
  height: 18px;
  cursor: se-resize;
  z-index: 10;
  /* subtle grip indicator */
  background: linear-gradient(
    135deg,
    transparent 30%,
    var(--muted) 30%, var(--muted) 40%,
    transparent 40%,
    transparent 55%,
    var(--muted) 55%, var(--muted) 65%,
    transparent 65%,
    transparent 80%,
    var(--muted) 80%, var(--muted) 90%,
    transparent 90%
  );
  opacity: 0.45;
  border-radius: 0 0 14px 0;
}
.resize-handle:hover { opacity: 0.85; }

/* ── Scrollbar ───────────────────────────────────────────────── */

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 2px; }

/* ── Animations ──────────────────────────────────────────────── */

@keyframes fade-up {
  from { opacity: 0; transform: translateY(5px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.35; }
}
`;

// ── Mount / unmount ───────────────────────────────────────────────────────────

let hostEl: HTMLElement | null = null;
let reactRoot: ReturnType<typeof createRoot> | null = null;
let overlayCallbacks: OverlayCallbacks | null = null;

export function mountOverlay(onClose: () => void): void {
  if (hostEl) return;

  hostEl = document.createElement('div');
  hostEl.id = 'asl-overlay-host';
  // The host itself is invisible; positioning happens on .panel inside shadow DOM
  hostEl.style.cssText = 'all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483646;';
  document.body.appendChild(hostEl);

  const shadow = hostEl.attachShadow({ mode: 'open' });

  const styleEl = document.createElement('style');
  styleEl.textContent = OVERLAY_CSS;
  shadow.appendChild(styleEl);

  const mountPoint = document.createElement('div');
  mountPoint.style.cssText = 'pointer-events: auto;';
  shadow.appendChild(mountPoint);

  reactRoot = createRoot(mountPoint);
  reactRoot.render(
    createElement(OverlayApp, {
      onReady: (cbs) => { overlayCallbacks = cbs; },
      onClose,
    })
  );
}

export function unmountOverlay(): void {
  reactRoot?.unmount();
  hostEl?.remove();
  reactRoot = null;
  hostEl = null;
  overlayCallbacks = null;
}

export function addWordToOverlay(word: string): void {
  overlayCallbacks?.addWord(word);
}

export function clearOverlay(): void {
  overlayCallbacks?.clear();
}
