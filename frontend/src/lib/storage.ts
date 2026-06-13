export type Theme = 'light' | 'dark' | 'high-contrast';
export type FontSize = number;

export interface StoredSettings {
  enabled: boolean;
  backendUrl?: string;
  theme: Theme;
  fontSize: FontSize;
}

export type DetectionState = 'idle' | 'no_hand' | 'thinking' | 'predicted';

export interface RuntimeState {
  wsConnected: boolean;
  detectionState: DetectionState;
  lastPrediction?: { word: string; confidence: number };
}

const SETTINGS_DEFAULTS: StoredSettings = {
  enabled: false,
  backendUrl: 'ws://localhost:8000/ws',
  theme: 'dark',
  fontSize: 20,
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
