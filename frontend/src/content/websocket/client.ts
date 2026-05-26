import type { LandmarkMessage, PredictionMessage } from '../../lib/types';

const RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_ATTEMPTS = 5;

export class WSClient {
  private sessionId: string;
  private onPrediction: (msg: PredictionMessage) => void;
  private onConnect?: () => void;
  private onDisconnect?: () => void;
  private wsUrl: string;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private isConnecting = false;
  private pendingMessages: LandmarkMessage[] = [];

  constructor(
    onPrediction: (msg: PredictionMessage) => void,
    wsUrl: string = 'ws://localhost:8000/ws',
    onConnect?: () => void,
    onDisconnect?: () => void,
  ) {
    this.sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    this.onPrediction = onPrediction;
    this.onConnect = onConnect;
    this.onDisconnect = onDisconnect;
    this.wsUrl = wsUrl;
  }

  connect(): void {
    if (this.ws || this.isConnecting) return;
    this.isConnecting = true;

    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('[ASL WS] Connected to backend at', this.wsUrl);
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        this.onConnect?.();
        this.flushPendingMessages();
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const message = JSON.parse(event.data) as
            | PredictionMessage
            | { error: string };

          if ('error' in message) {
            console.warn('[ASL WS] Server error:', message.error);
          } else {
            this.onPrediction(message);
          }
        } catch (err) {
          console.error('[ASL WS] Failed to parse message:', err, event.data);
        }
      };

      this.ws.onerror = () => {
        console.error('[ASL WS] WebSocket error');
      };

      this.ws.onclose = () => {
        console.log('[ASL WS] Disconnected from backend');
        this.ws = null;
        this.isConnecting = false;
        this.onDisconnect?.();
        this.attemptReconnect();
      };
    } catch (err) {
      console.error('[ASL WS] Failed to create WebSocket:', err);
      this.isConnecting = false;
      this.attemptReconnect();
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error(
        '[ASL WS] Max reconnection attempts reached, giving up'
      );
      return;
    }

    this.reconnectAttempts++;
    const delay = RECONNECT_DELAY_MS * this.reconnectAttempts;
    console.log(`[ASL WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnecting = false;
  }

  sendLandmarks(landmarks: number[]): void {
    if (landmarks.length !== 258) {
      console.warn(
        `[ASL WS] Invalid landmark count: ${landmarks.length}, expected 258`
      );
      return;
    }

    const message: LandmarkMessage = {
      landmarks,
      session_id: this.sessionId,
      timestamp: Date.now(),
    };

    if (!this.isConnected) {
      this.pendingMessages.push(message);
      if (this.pendingMessages.length > 100) {
        this.pendingMessages.shift();
      }
      return;
    }

    try {
      this.ws!.send(JSON.stringify(message));
    } catch (err) {
      console.error('[ASL WS] Failed to send landmarks:', err);
    }
  }

  private flushPendingMessages(): void {
    while (this.pendingMessages.length > 0 && this.isConnected) {
      const message = this.pendingMessages.shift();
      if (message) {
        try {
          this.ws!.send(JSON.stringify(message));
        } catch (err) {
          console.error('[ASL WS] Failed to flush pending message:', err);
          break;
        }
      }
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
