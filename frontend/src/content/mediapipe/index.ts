// MediaPipe Hands wrapper.
// Initializes the model, runs per-frame inference on the webcam stream,
// and emits 21 landmarks × (x, y, z) = 63 numbers per frame.

export type LandmarkCallback = (landmarks: number[]) => void;

export async function startMediaPipe(_video: HTMLVideoElement, _onFrame: LandmarkCallback): Promise<void> {
  // TODO: load MediaPipe Hands, run detection loop, invoke onFrame(landmarks)
  throw new Error("not implemented");
}
