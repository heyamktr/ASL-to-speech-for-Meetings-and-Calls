import { HandTracker } from "./mediapipe";
import { WSClient } from "./websocket/client";
import { onMessage } from "../lib/messaging";
import type { PredictionMessage } from "../lib/types";
import { getSettings, setRuntimeState } from "../lib/storage";
import type { DetectionState } from "../lib/storage";

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
  updateDetectionState('predicted', { word: msg.prediction, confidence: msg.confidence });
}

async function start(): Promise<void> {
  if (isRunning) return;
  isRunning = true;
  console.log("[ASL] Starting...");

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
  tracker = null;
  wsClient = null;
  updateDetectionState('idle');
  setRuntimeState({ wsConnected: false });
}

onMessage((message) => {
  if (message.type === "TOGGLE") {
    if (message.enabled) {
      start();
    } else {
      stop();
    }
  }
});

getSettings().then(({ enabled }) => {
  if (enabled) start();
});

console.log("[ASL] Content script loaded on", window.location.hostname);
