 # ASL-to-Speech — Codebase Tour

This is a Chrome extension that lets a deaf or hard-of-hearing person sign into their camera during a video call, and have the signs automatically recognized and spoken aloud to other participants. The repo is a **monorepo** with three independent pieces.

---

## Big Picture: How Data Flows

```
Camera → MediaPipe (browser) → 144 numbers/frame → WebSocket
  → FastAPI Server → ONNX model → predicted word → WebSocket back
  → Overlay UI shows word + TTS speaks it into the mic stream
```

---

## 1. `frontend/` — Chrome Extension (TypeScript + React)

This is a **Manifest V3** Chrome extension. MV3 has strict rules about what scripts can run where, which shapes the whole architecture.

### `public/manifest.json`
The extension's identity card. Declares:
- It only activates on `https://meet.google.com/*`
- Needs `storage` and `activeTab` permissions, plus optional `camera`
- Has three entry points: a content script, a background service worker, and a popup

### `src/popup/Popup.tsx`
The small UI that appears when you click the extension icon in Chrome's toolbar. It's a minimal React component with a single **Enable/Disable** toggle button. When toggled, it saves state to `chrome.storage.sync` and sends a `TOGGLE` message to the active tab's content script.

### `src/content/index.tsx` — The Content Script Entry Point
This is injected into every Google Meet page. It:
1. Listens for `TOGGLE` messages from the popup
2. On enable: creates a `HandTracker` and a `WSClient`, wires them together
3. On disable: tears both down cleanly
4. Also restores the "was it enabled before?" state on page load

### `src/content/mediapipe/index.ts` — `HandTracker` class
This is the trickiest part of the extension. MediaPipe can't run in the extension's sandboxed content script context, so it uses a **two-world bridge pattern**:
`
1. Injects `injected/mediapipe-bridge.js` as a `<script>` tag into the **main page context** (where MediaPipe can run)
2. Posts a `window.postMessage` to that bridge saying "init MediaPipe"
3. Waits for an `ASL_MEDIAPIPE_READY` message back
4. Acquires the camera, creates a hidden `<video>` element, and runs a `requestAnimationFrame` loop
5. Each frame: draws the video onto a canvas, creates an `ImageBitmap`, and posts it to the bridge via `window.postMessage`
6. The bridge runs MediaPipe and posts back `ASL_LANDMARKS` with 144 numbers
7. `HandTracker` forwards those 144 numbers to the `WSClient`

### `src/content/websocket/client.ts` — `WSClient` class
The WebSocket client that talks to the backend. Key behaviors:
- Generates a unique `session_id` per page load (used server-side for Redis isolation)
- Sends each frame as `{ landmarks: number[144], session_id, timestamp }`
- Buffers up to 100 messages if disconnected, then flushes them on reconnect
- Exponential-ish reconnect backoff, up to 5 attempts

### `src/injected/inject.ts`
A stub for **Phase 2 audio injection**. It overrides `navigator.mediaDevices.getUserMedia` so it can intercept the mic stream and mix in TTS audio. Right now it just calls through to the original — the mixing logic is a TODO.

### `src/lib/types.ts`
The shared TypeScript types that keep both sides of the WebSocket protocol honest:
- `LandmarkMessage` — what the browser sends (144 landmarks + session_id + timestamp)
- `PredictionMessage` — what the server replies (prediction word + confidence + echoed timestamp)
- `ExtensionMessage` — popup ↔ content script message types

---

## 2. `backend/` — FastAPI Inference Server (Python)

A Python 3.11 FastAPI server that receives landmark frames over WebSocket, accumulates them into a sliding window per session, and runs the ONNX model when the window is full.

### `app/config.py`
A frozen dataclass that reads all configuration from environment variables, with sane defaults. Things like `REDIS_URL`, `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, etc. This makes the server easy to deploy with Docker env vars.

### `app/main.py`
FastAPI app entry point. Uses the **lifespan** pattern (a newer FastAPI feature) to:
- Load the ONNX model **once** at startup and store it as `app.state.classifier`
- Close the Redis connection cleanly at shutdown
- Register the WebSocket router and a `/health` endpoint

### `app/websocket.py`
The WebSocket handler. For every connected client it runs a loop:
1. Receive a JSON message
2. Validate it — exactly 144 numbers, a non-empty `session_id`, a numeric `timestamp`
3. Pass landmarks to `add_frame()` (Redis buffer)
4. If the buffer has enough frames, call `classifier.predict()`
5. If the smoothing gate passes, include `prediction` + `confidence` in the response
6. Always echo back the `timestamp` (lets the browser measure round-trip latency)
7. Log per-message latency broken down into receive/infer/send phases

`LANDMARK_COUNT = 144` = 21 right-hand landmarks × 3 (x,y,z) + 21 left-hand × 3 + 6 pose joints × 3.

### `app/session/buffer.py`
Redis-backed session state. Two responsibilities:

**Sliding window** (`add_frame`): Every incoming frame is appended to a Redis list keyed by `session:{id}:frames`. The list is trimmed to the last 100 frames with `LTRIM`. Once there are 100 frames, the full window is returned for inference. Sessions expire after 5 minutes of inactivity.

**Smoothing gate** (`should_emit`): To avoid flooding the frontend with every prediction, the same word must be predicted `SMOOTHING_K = 3` consecutive times before it's sent. This prevents flickering from noisy single-frame predictions.

### `app/inference.py` — `SignClassifier` class
The inference engine. On construction:
1. Loads the `.onnx` file via ONNX Runtime
2. Loads a `label_map.json` that maps class indices to English words
3. Runs a warm-up inference so the first real request isn't slow

`normalize_window()` applies the same preprocessing used during training:
- Centers each hand on its wrist point (wrist-relative coords)
- Normalizes scale by the distance from wrist to middle-finger knuckle (landmark 9)
- Centers pose on the shoulder midpoint and normalizes by shoulder width

`build_model_features()` then:
1. Pads/trims the window to exactly `seq_len` frames
2. Adds binary hand-presence flags (0 or 1) alongside coords
3. Computes **velocity** = frame-to-frame differences and appends them
4. Result: 144 raw coords + 2 presence bits + 146 velocity dims = **292 features/frame**

`predict()` runs the ONNX session, applies softmax to get probabilities, and returns `"uncertain"` if the top confidence is below the threshold.

---

## 3. `model-training/` — PyTorch Training Pipeline

Offline ML code that produces the `.onnx` model file deployed to the backend.

### `src/models/lstm.py` — `SignGRU` (the actual model)
Despite the file being named `lstm.py`, the model is a **bidirectional GRU** (the LSTM was replaced; `SignLSTM` remains as an alias for backward compatibility). Architecture:

1. **Projection layer**: Linear → LayerNorm → ReLU — projects the raw 292-dim input into `hidden_size` dims. LayerNorm stabilizes training because coordinates and presence bits live on very different scales.

2. **Bidirectional GRU** with 2 layers — processes the time sequence in both directions so each frame has context from past and future frames.

3. **Temporal attention**: A single linear layer produces a score per frame; softmax turns those into weights; the weighted sum across time gives a single fixed-size representation. Crucially, padding frames are masked to `-inf` before softmax so they can't contaminate the pooled result.

4. **Classification head**: Dropout → Linear → logits over N classes.

**Why GRU over LSTM**: GRU has ~25% fewer parameters. With only ~10 training samples per class, overfitting is the #1 risk, so fewer parameters help.

### `src/train.py`
The training loop. Heavy data augmentation is applied to fight overfitting on a small dataset:
- **Random temporal crop**: a different 100-frame window is cut each epoch from sequences longer than 100 frames
- **Time warp**: stretch or compress the temporal axis by ±20% and resample back — simulates signers who sign faster or slower
- **Gaussian jitter** on coordinates (noise std 0.07)
- **Scale jitter**: random ±20% scaling of all coordinates
- **Horizontal flip**: mirrors the x-axis for a random half of each batch (simulates left-handed signers)
- **Frame dropout**: randomly zeros 25% of frames — forces the model to classify from partial observations
- **Mixup**: blends two training samples and their labels (Beta(0.4, 0.4) mixing weight)

Training uses:
- `CrossEntropyLoss` with class weights (inverse frequency) and label smoothing 0.1
- Adam optimizer with `weight_decay=1e-3` and `eps=1e-3`
- `ReduceLROnPlateau` scheduler (halves LR after 15 epochs without improvement)
- **Stochastic Weight Averaging (SWA)**: from epoch 120 onward, model weights are averaged across epochs — often gives a small accuracy boost for free
- **4-crop Test-Time Augmentation** at validation: averages logits over 4 evenly-spaced temporal crops
- Early stopping after 30 epochs of no improvement in validation loss

### `src/export_onnx.py`
Converts a trained PyTorch `.pt` checkpoint to an `.onnx` file ready for the backend. Steps:
1. Loads the checkpoint and reconstructs `SignGRU` with the saved config
2. Creates dummy inputs and calls `torch.onnx.export`
3. Uses `dynamic_axes` so the batch dimension is dynamic (can serve any batch size)
4. Copies `label_map.json` and writes `export_meta.json` alongside the model

The exported file goes to `model-training/exports/` — you then copy it to `backend/models/` for the server to pick up.

---

## Key Cross-Cutting Numbers

| Number | What it means |
|--------|--------------|
| **144** | Raw landmark dimensions per frame (21 RH × 3 + 21 LH × 3 + 6 pose × 3) |
| **146** | After adding 2 hand-presence bits (one per hand) |
| **292** | Model input after appending velocity (146 position + 146 velocity) |
| **100** | Sliding window size (frames buffered before each inference) |
| **3** | Smoothing gate — same word must appear 3 times consecutively to emit |
| **0.4** | Confidence threshold below which the server returns `"uncertain"` |

---

## What's Still In Progress

- `inject.ts` — the `getUserMedia` override is stubbed; TTS audio mixing into the mic stream hasn't been implemented yet
- `content/index.tsx:11-13` — prediction callback just `console.log`s; the overlay UI (`overlay/App.tsx`) isn't wired up yet
- The `manifest.json` only targets Google Meet; Zoom/Teams/Messenger support is planned but not wired
