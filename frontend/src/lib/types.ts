// Shared TypeScript types used across content, popup, and background.

export interface LandmarkFrame {
  // 21 hand landmarks × (x, y, z) = 63 numbers
  values: number[];
  timestamp: number;
}

export interface PredictionResult {
  word: string;
  confidence: number;
  timestamp: number;
}

// Outgoing: content script → backend
export interface LandmarkMessage {
  landmarks: number[];   // exactly 144 values: [0:63] right hand (x,y,z) + [63:126] left hand (x,y,z) + [126:144] 6 pose joints (x,y,z)
  session_id: string;
  timestamp: number;     // Date.now() — echoed back for RTT measurement
}

// Incoming: backend → content script.
// prediction/confidence are present only on the frame where a sign is emitted;
// every other frame carries just the echoed timestamp.
export interface PredictionMessage {
  prediction?: string;
  confidence?: number;   // 0.0–1.0
  timestamp: number;     // echoed from the request
}

// Popup ↔ content script messaging
export type ExtensionMessage =
  | { type: 'TOGGLE'; enabled: boolean }
  | { type: 'STATUS_REQUEST' }
  | { type: 'STATUS_RESPONSE'; enabled: boolean; wsConnected: boolean }
  | { type: 'CLEAR_SENTENCE' };
