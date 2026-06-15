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
  font-family: 'Lora', Georgia, 'Times New Roman', serif;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  min-width: 220px;
  min-height: 160px;
}

/* ── Themes ──────────────────────────────────────────────────── */

.panel[data-theme="dark"] {
  background: rgba(10, 26, 47, 0.94);
  color: #e6f6f4;
  border: 1px solid rgba(45,212,191,0.18);
  box-shadow: 0 12px 40px rgba(0,0,0,0.7), 0 0 0 0.5px rgba(45,212,191,0.08);
  --accent: #2dd4bf;
  --muted: #7ba8b3;
  --header-bg: rgba(12, 33, 56, 0.98);
  --peer-bg: rgba(9, 28, 48, 0.92);
  --btn-bg: rgba(45,212,191,0.12);
  --btn-hover: rgba(45,212,191,0.24);
  --divider: rgba(45,212,191,0.16);
  --block-bg: rgba(45,212,191,0.08);
}

.panel[data-theme="light"] {
  background: rgba(255,255,255,0.96);
  color: #0f2e2b;
  border: 1px solid rgba(13,148,136,0.18);
  box-shadow: 0 8px 32px rgba(13,148,136,0.16);
  --accent: #0d9488;
  --muted: #5e8a86;
  --header-bg: rgba(240, 251, 250, 0.98);
  --peer-bg: rgba(236, 250, 248, 0.94);
  --btn-bg: rgba(13,148,136,0.08);
  --btn-hover: rgba(13,148,136,0.16);
  --divider: rgba(13,148,136,0.14);
  --block-bg: rgba(13,148,136,0.06);
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
  position: relative;
  height: 9px;
  flex-shrink: 0;
  cursor: ns-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  border-top: 1px solid var(--divider);
  border-bottom: 1px solid var(--divider);
  background: var(--block-bg);
  user-select: none;
  touch-action: none;
}
.section-divider:hover { background: var(--btn-hover); }

.divider-grip {
  width: 34px;
  height: 3px;
  border-radius: 2px;
  background: var(--muted);
  opacity: 0.5;
}
.section-divider:hover .divider-grip { opacity: 0.9; }

/* ── Peer transcript section ─────────────────────────────────── */

.peer-section {
  flex-shrink: 0;
  background: var(--peer-bg);
  overflow-y: auto;
  padding: 7px 12px 9px;
  /* height is set inline and adjusted by dragging the divider */
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
