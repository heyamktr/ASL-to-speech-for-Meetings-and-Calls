import { HandTracker } from "./mediapipe";
import { WSClient } from "./websocket/client";
import { onMessage } from "../lib/messaging";
import type { PredictionMessage } from "../lib/types";
import { getSettings, setSettings, setRuntimeState } from "../lib/storage";
import type { DetectionState } from "../lib/storage";
import { mountOverlay, unmountOverlay, addWordToOverlay, clearOverlay } from "./overlay/mount";

let tracker: HandTracker | null = null;
let wsClient: WSClient | null = null;
let isRunning = false;
let localDetectionState: DetectionState = 'idle';

function hasHands(landmarks: number[]): boolean {
  // Indices 132-257 are left + right hand landmarks; all zeros means no hands detected
  return landmarks.slice(132).some((v) => v !== 0);
}

function updateDetectionState(next: DetectionState, extra?: { word: string; confidence: number }): void {
  if (localDetectionState === next && !extra) return;
  localDetectionState = next;
  const patch = extra
    ? { detectionState: next, lastPrediction: extra }
    : { detectionState: next };
  setRuntimeState(patch);
}

function onPrediction(msg: PredictionMessage): void {
  if (!msg.prediction || msg.prediction === "uncertain") return;
  updateDetectionState('predicted', { word: msg.prediction, confidence: msg.confidence ?? 0 });
  addWordToOverlay(msg.prediction);

  // Speak the word — postMessage reaches inject.ts in the MAIN world
  getSettings().then(({ voiceEnabled, ttsEndpoint, speechRate, speechPitch, voiceURI }) => {
    if (!voiceEnabled) return;
    window.postMessage({
      type: 'ASL_SPEAK_WORD',
      word: msg.prediction,
      ttsEndpoint,
      rate: speechRate,
      pitch: speechPitch,
      voiceURI,
    }, '*');
  });
}

async function start(): Promise<void> {
  if (isRunning) return;
  isRunning = true;
  console.log("[ASL] Starting...");

  mountOverlay(closeFromOverlay);

  const settings = await getSettings();
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
    clearOverlay();
  }
});

getSettings().then(({ enabled }) => {
  if (enabled) start();
});

console.log("[ASL] Content script loaded on", window.location.hostname);
