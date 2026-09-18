export type Theme = 'light' | 'dark';
export type FontSize = number;

export interface StoredSettings {
  enabled: boolean;
  backendUrl?: string;
  theme: Theme;
  fontSize: FontSize;
  voiceEnabled: boolean;
  voiceURI: string;
  speechRate: number;
  speechPitch: number;
  ttsEndpoint: string;
  // ── LLM sentence overlay ──────────────────────────────────────────────────
  // When on, a completed utterance (the accumulated gloss words) is sent to the
  // backend /refine endpoint and rewritten into one natural English sentence
  // before it is shown and spoken. Off = the raw space-joined words are used.
  llmEnabled: boolean;
  refineEndpoint: string;
  // Milliseconds of no new sign before an utterance is auto-finalized (spoken).
  autoSpeakMs: number;
}

export type DetectionState = 'idle' | 'no_hand' | 'thinking' | 'predicted';

export interface RuntimeState {
  wsConnected: boolean;
  detectionState: DetectionState;
  lastPrediction?: { word: string; confidence: number };
}

const SETTINGS_DEFAULTS: StoredSettings = {
  enabled: false,
  backendUrl: 'wss://asl-to-speech-for-meetings-and-calls-sunlit-wind-7569.fly.dev/ws',
  theme: 'dark',
  fontSize: 20,
  voiceEnabled: false,
  voiceURI: '',
  speechRate: 1.0,
  speechPitch: 1.0,
  ttsEndpoint: 'https://asl-to-speech-for-meetings-and-calls-sunlit-wind-7569.fly.dev/tts',
  llmEnabled: true,
  refineEndpoint: 'https://asl-to-speech-for-meetings-and-calls-sunlit-wind-7569.fly.dev/refine',
  autoSpeakMs: 2500,
};

const STATE_DEFAULTS: RuntimeState = {
  wsConnected: false,
  detectionState: 'idle',
};

export async function getSettings(): Promise<StoredSettings> {
  return new Promise((resolve) => {
    chrome.storage.sync.get(SETTINGS_DEFAULTS, (result) => {
      resolve(result as StoredSettings);
    });
  });
}

export async function setSettings(patch: Partial<StoredSettings>): Promise<void> {
  return new Promise((resolve) => {
    chrome.storage.sync.set(patch, resolve);
  });
}

export async function getRuntimeState(): Promise<RuntimeState> {
  return new Promise((resolve) => {
    chrome.storage.local.get(STATE_DEFAULTS, (result) => {
      resolve(result as RuntimeState);
    });
  });
}

export async function setRuntimeState(patch: Partial<RuntimeState>): Promise<void> {
  return new Promise((resolve) => {
    chrome.storage.local.set(patch, resolve);
  });
}
