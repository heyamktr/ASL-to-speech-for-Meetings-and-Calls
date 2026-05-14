import type { PredictionMessage } from '../../lib/types';

export class WSClient {
  private sessionId: string;
  private onPrediction: (msg: PredictionMessage) => void;

  constructor(onPrediction: (msg: PredictionMessage) => void) {
    this.sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    this.onPrediction = onPrediction;
  }

  connect(): void {
    console.log('[ASL WS] Stub mode — Dev B server not available yet');
  }

  disconnect(): void {}

  sendLandmarks(landmarks: number[]): void {
    console.log('[ASL WS] Would send:', {
      landmarks: landmarks.slice(0, 9),
      total: landmarks.length,
      session_id: this.sessionId,
      timestamp: Date.now(),
    });
  }

  get isConnected(): boolean {
    return false;
  }
}
