import { HandTracker } from "./mediapipe";
import { WSClient } from "./websocket/client";
import { onMessage } from "../lib/messaging";
import type { PredictionMessage } from "../lib/types";
import { getSettings, setSettings, setRuntimeState } from "../lib/storage";
import type { DetectionState } from "../lib/storage";
import {
  mountOverlay,
  unmountOverlay,
  addWordToOverlay,
  beginOverlayBlock,
  resolveOverlayBlock,
  clearOverlay,
} from "./overlay/mount";

let tracker: HandTracker | null = null;
let wsClient: WSClient | null = null;
let isRunning = false;
let localDetectionState: DetectionState = 'idle';

// ── Utterance assembly ─────────────────────────────────────────────────────────
// Emitted words accumulate into the current utterance. It is "finalized" (rewritten
// into a natural sentence, then shown + spoken) when the signer pauses for
// `autoSpeakMs`, or immediately when they press the overlay's Speak button.
let pendingWords: string[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
let autoSpeakMs = 2500;

function hasHands(landmarks: number[]): boolean {
  // Indices [0:126] are right + left hand landmarks (21 × 3 each); [126:144] is pose.
  // All-zero hand block means MediaPipe detected no hands. Matches backend _has_hand.
  return landmarks.slice(0, 126).some((v) => v !== 0);
}

function updateDetectionState(next: DetectionState, extra?: { word: string; confidence: number }): void {
  if (localDetectionState === next && !extra) return;
  localDetectionState = next;
  const patch = extra
    ? { detectionState: next, lastPrediction: extra }
    : { detectionState: next };
  setRuntimeState(patch);
}

// Deterministic offline rewrite — used when the LLM overlay is disabled or the
// /refine call fails. Mirrors the backend's local fallback.
function localSentence(words: string[]): string {
  const text = words.map((w) => w.trim()).filter(Boolean).join(' ');
  if (!text) return '';
  const capped = text.charAt(0).toUpperCase() + text.slice(1);
  return /[.!?]$/.test(capped) ? capped : capped + '.';
}

// Turn a completed utterance's gloss words into one natural sentence via the
// backend /refine endpoint. Falls back to a local join on any error so the
// caption + voice pipeline never stalls.
async function refineWords(words: string[]): Promise<string> {
  const { llmEnabled, refineEndpoint } = await getSettings();
  if (llmEnabled && refineEndpoint) {
    try {
      const res = await fetch(refineEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words }),
      });
      if (res.ok) {
        const data = (await res.json()) as { sentence?: string };
        if (data && typeof data.sentence === 'string' && data.sentence.trim()) {
          return data.sentence.trim();
        }
      } else {
        console.warn('[ASL] /refine returned', res.status);
      }
    } catch (err) {
      console.warn('[ASL] /refine failed, using local join:', err);
    }
  }
  return localSentence(words);
}

function scheduleFlush(): void {
  if (flushTimer) clearTimeout(flushTimer);
  flushTimer = setTimeout(flushUtterance, autoSpeakMs);
}

function cancelFlush(): void {
  if (flushTimer) clearTimeout(flushTimer);
  flushTimer = null;
}

// Finalize the current utterance: rewrite it to a sentence, show it, speak it.
// Safe to call with no pending words (no-op) — the overlay Speak button uses it.
function flushUtterance(): void {
  cancelFlush();
  const words = pendingWords;
  pendingWords = [];
  if (words.length === 0) return;

  // Immediately move the live gloss into a finalized block (shown as "refining…"),
  // so a new utterance can start on a fresh line while this one is rewritten.
  const gloss = words.join(' ');
  const blockId = beginOverlayBlock(gloss);

  refineWords(words).then((sentence) => {
    const finalText = sentence || gloss;
    resolveOverlayBlock(blockId, finalText);

    // Speak the ONE finished sentence (not word-by-word) into the mic stream.
    getSettings().then(({ voiceEnabled, ttsEndpoint, speechRate, speechPitch, voiceURI }) => {
      if (!voiceEnabled) return;
      window.postMessage({
        type: 'ASL_SPEAK_SENTENCE',
        text: finalText,
        ttsEndpoint,
        rate: speechRate,
        pitch: speechPitch,
        voiceURI,
      }, '*');
    });
  });
}

function onPrediction(msg: PredictionMessage): void {
  if (!msg.prediction || msg.prediction === "uncertain") return;
  updateDetectionState('predicted', { word: msg.prediction, confidence: msg.confidence ?? 0 });
  pendingWords.push(msg.prediction);
  addWordToOverlay(msg.prediction);
  scheduleFlush();
}

async function start(): Promise<void> {
  if (isRunning) return;
  isRunning = true;
  console.log("[ASL] Starting...");

  const settings = await getSettings();
  autoSpeakMs = settings.autoSpeakMs || 2500;

  // The overlay's Speak button finalizes the current utterance immediately.
  mountOverlay(closeFromOverlay, flushUtterance);

  const wsUrl = settings.backendUrl || "ws://localhost:8000/ws";

  wsClient = new WSClient(
    onPrediction,
    wsUrl,
    () => setRuntimeState({ wsConnected: true }),
    () => setRuntimeState({ wsConnected: false }),
  );
  wsClient.connect();

  tracker = new HandTracker((landmarks) => {
    if (hasHands(landmarks)) {
      updateDetectionState('thinking');
    } else {
      updateDetectionState('no_hand');
    }
    wsClient?.sendLandmarks(landmarks);
  });

  try {
    await tracker.start();
  } catch (err) {
    console.error("[ASL] Failed to start:", err);
    console.warn(
      "[ASL] If using Meet on Google domain, you may need to grant camera permission when prompted",
    );
    isRunning = false;
    updateDetectionState('idle');
    setRuntimeState({ wsConnected: false });
  }
}

function stop(): void {
  if (!isRunning) return;
  isRunning = false;
  console.log("[ASL] Stopping...");
  cancelFlush();
  pendingWords = [];
  tracker?.stop();
  wsClient?.disconnect();
  unmountOverlay();
  tracker = null;
  wsClient = null;
  updateDetectionState('idle');
  setRuntimeState({ wsConnected: false });
}

// The overlay's ✕ closes the whole feature: persist enabled=false (so the popup
// toggle reflects it) and tear everything down.
function closeFromOverlay(): void {
  setSettings({ enabled: false });
  stop();
}

onMessage((message) => {
  if (message.type === "TOGGLE") {
    if (message.enabled) {
      start();
    } else {
      stop();
    }
  } else if (message.type === "CLEAR_SENTENCE") {
    cancelFlush();
    pendingWords = [];
    clearOverlay();
  }
});

getSettings().then(({ enabled }) => {
  if (enabled) start();
});

console.log("[ASL] Content script loaded on", window.location.hostname);
