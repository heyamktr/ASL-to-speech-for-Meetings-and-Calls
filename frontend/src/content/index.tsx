import { HandTracker } from "./mediapipe";
import { WSClient } from "./websocket/client";
import { onMessage } from "../lib/messaging";
import type { PredictionMessage } from "../lib/types";
import { getSettings } from "../lib/storage";

let tracker: HandTracker | null = null;
let wsClient: WSClient | null = null;
let isRunning = false;

function onPrediction(msg: PredictionMessage): void {
  // Week 1: just log. Week 2: render in overlay.
  console.log("[ASL] Got prediction:", msg);
}

async function start(): Promise<void> {
  if (isRunning) return;
  isRunning = true;
  console.log("[ASL] Starting...");

  wsClient = new WSClient(onPrediction);
  wsClient.connect();

  tracker = new HandTracker((landmarks) => {
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
}

// Listen for toggle messages from the popup
onMessage((message) => {
  if (message.type === "TOGGLE") {
    if (message.enabled) {
      start();
    } else {
      stop();
    }
  }
});

// Restore state from last session on page load
getSettings().then(({ enabled }) => {
  if (enabled) start();
});

console.log("[ASL] Content script loaded on", window.location.hostname);
