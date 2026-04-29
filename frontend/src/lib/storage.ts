// chrome.storage.sync helpers — persists user settings across sessions.

export interface UserSettings {
  enabled: boolean;
  fontSize: "small" | "medium" | "large";
  theme: "light" | "dark" | "high-contrast";
  voice: string | null;
}

const defaults: UserSettings = {
  enabled: true,
  fontSize: "medium",
  theme: "light",
  voice: null,
};

export async function getSettings(): Promise<UserSettings> {
  const stored = await chrome.storage.sync.get(defaults);
  return stored as UserSettings;
}

export async function setSettings(patch: Partial<UserSettings>): Promise<void> {
  await chrome.storage.sync.set(patch);
}
