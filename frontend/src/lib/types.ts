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
