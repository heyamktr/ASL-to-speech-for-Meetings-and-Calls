// WebSocket client to the FastAPI inference backend.

export class InferenceSocket {
  private ws: WebSocket | null = null;

  connect(url: string): void {
    this.ws = new WebSocket(url);
    // TODO: wire up open/message/close/error handlers
  }

  sendLandmarks(_landmarks: number[]): void {
    // TODO: send a single frame's 63 numbers
  }

  onWord(_callback: (word: string) => void): void {
    // TODO: subscribe to predicted-word messages from the server
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}
