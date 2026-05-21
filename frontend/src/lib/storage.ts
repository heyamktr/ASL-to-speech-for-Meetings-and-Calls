export interface StoredSettings {
  enabled: boolean;
  backendUrl?: string;
}

const DEFAULTS: StoredSettings = {
  enabled: false,
  backendUrl: 'ws://localhost:8000/ws',
};

export async function getSettings(): Promise<StoredSettings> {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULTS, (result) => {
      resolve(result as StoredSettings);
    });
  });
}

export async function setSettings(patch: Partial<StoredSettings>): Promise<void> {
  return new Promise((resolve) => {
    chrome.storage.sync.set(patch, resolve);
  });
}