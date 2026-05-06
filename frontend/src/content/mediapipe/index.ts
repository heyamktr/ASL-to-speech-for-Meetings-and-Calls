export type LandmarkCallback = (landmarks: number[]) => void;

export class HandTracker {
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private animFrameId: number | null = null;
  private onLandmarks: LandmarkCallback;
  private running = false;
  private ready = false;

  constructor(onLandmarks: LandmarkCallback) {
    this.onLandmarks = onLandmarks;
  }

  async start(): Promise<void> {
    console.log('[ASL MediaPipe] Starting...');

    // Listen for landmarks coming back from main world
    window.addEventListener('message', this.handleMessage.bind(this));

    // Inject the bridge script into main world
    await this.injectBridge();

    // Tell bridge to init MediaPipe, passing the extension base URL
    window.postMessage({
      type: 'ASL_INIT_MEDIAPIPE',
      extensionUrl: chrome.runtime.getURL(''),
    }, '*');

    // Wait for bridge to confirm MediaPipe is ready
    await this.waitForReady();

    // Start camera
    const stream = await this.acquireCamera();
    this.video = this.createHiddenVideo(stream);
    this.canvas = document.createElement('canvas');
    this.canvas.width = 640;
    this.canvas.height = 480;
    await this.videoReady();

    this.running = true;
    this.loop();
    console.log('[ASL MediaPipe] Hand tracking started');
  }

  stop(): void {
    this.running = false;
    if (this.animFrameId != null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.video) {
      (this.video.srcObject as MediaStream)?.getTracks().forEach((t) => t.stop());
      this.video.remove();
      this.video = null;
    }
    console.log('[ASL MediaPipe] Hand tracking stopped');
  }

  // ── private ──────────────────────────────────────────────────────────────

  private handleMessage(event: MessageEvent): void {
    if (event.source !== window) return;
    if (event.data?.type === 'ASL_LANDMARKS') {
      this.onLandmarks(event.data.landmarks);
      console.log(`[ASL MediaPipe] Hand detected — ${event.data.landmarks.length} values`);
    }
  }

  private injectBridge(): Promise<void> {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL('injected/mediapipe-bridge.js');
      script.onload = () => {
        console.log('[ASL MediaPipe] Bridge injected');
        resolve();
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  private waitForReady(): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('MediaPipe bridge timed out'));
      }, 15000);

      const handler = (event: MessageEvent) => {
        if (event.data?.type === 'ASL_MEDIAPIPE_READY') {
          clearTimeout(timeout);
          window.removeEventListener('message', handler);
          this.ready = true;
          console.log('[ASL MediaPipe] Bridge ready');
          resolve();
        }
      };

      window.addEventListener('message', handler);
    });
  }

  private async loop(): Promise<void> {
    if (!this.running || !this.video || !this.canvas || !this.ready) return;

    if (this.video.readyState >= 2) {
      try {
        const ctx = this.canvas.getContext('2d')!;
        ctx.drawImage(this.video, 0, 0, 640, 480);
        const bitmap = await createImageBitmap(this.canvas);
        // Don't transfer — send as a copy so bridge can read it
        window.postMessage({ type: 'ASL_SEND_FRAME', bitmap }, '*');
      } catch (err) {
        console.error('[ASL MediaPipe] Frame error:', err);
      }
    }

    this.animFrameId = requestAnimationFrame(() => this.loop());
  }

  private async acquireCamera(): Promise<MediaStream> {
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
    } catch (err: any) {
      console.error('[ASL MediaPipe] Camera access denied:', err);
      throw err;
    }
  }

  private createHiddenVideo(stream: MediaStream): HTMLVideoElement {
    const video = document.createElement('video');
    video.style.cssText = 'display:none;position:absolute;';
    video.srcObject = stream;
    video.autoplay = true;
    video.playsInline = true;
    document.body.appendChild(video);
    return video;
  }

  private videoReady(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.video) return resolve();
      if (this.video.readyState >= 2) return resolve();
      this.video.onloadeddata = () => resolve();
    });
  }
}
