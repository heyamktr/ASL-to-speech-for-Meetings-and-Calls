const syncStore: Record<string, unknown> = {};
const localStore: Record<string, unknown> = {};

(global as unknown as { chrome: unknown }).chrome = {
  storage: {
    sync: {
      get: jest.fn((defaults: Record<string, unknown>, cb: (r: Record<string, unknown>) => void) => {
        cb({ ...defaults, ...syncStore });
      }),
      set: jest.fn((patch: Record<string, unknown>, cb: () => void) => {
        Object.assign(syncStore, patch);
        cb();
      }),
    },
    local: {
      get: jest.fn((defaults: Record<string, unknown>, cb: (r: Record<string, unknown>) => void) => {
        cb({ ...defaults, ...localStore });
      }),
      set: jest.fn((patch: Record<string, unknown>, cb: () => void) => {
        Object.assign(localStore, patch);
        cb();
      }),
    },
  },
};

import { getSettings, setSettings, getRuntimeState, setRuntimeState } from '../lib/storage';

const PROD_WS = 'wss://asl-to-speech-for-meetings-and-calls-sunlit-wind-7569.fly.dev/ws';
const PROD_TTS = 'https://asl-to-speech-for-meetings-and-calls-sunlit-wind-7569.fly.dev/tts';

describe('storage defaults', () => {
  test('backendUrl defaults to production WebSocket URL', async () => {
    const settings = await getSettings();
    expect(settings.backendUrl).toBe(PROD_WS);
  });

  test('ttsEndpoint defaults to production TTS URL', async () => {
    const settings = await getSettings();
    expect(settings.ttsEndpoint).toBe(PROD_TTS);
  });

  test('extension is disabled by default', async () => {
    const settings = await getSettings();
    expect(settings.enabled).toBe(false);
  });

  test('voice is disabled by default', async () => {
    const settings = await getSettings();
    expect(settings.voiceEnabled).toBe(false);
  });
});

describe('setSettings', () => {
  test('persists a patch to chrome.storage.sync', async () => {
    await setSettings({ enabled: true });
    const settings = await getSettings();
    expect(settings.enabled).toBe(true);
  });
});

describe('getRuntimeState', () => {
  test('wsConnected defaults to false', async () => {
    const state = await getRuntimeState();
    expect(state.wsConnected).toBe(false);
  });

  test('detectionState defaults to idle', async () => {
    const state = await getRuntimeState();
    expect(state.detectionState).toBe('idle');
  });
});

describe('setRuntimeState', () => {
  test('persists wsConnected to chrome.storage.local', async () => {
    await setRuntimeState({ wsConnected: true });
    const state = await getRuntimeState();
    expect(state.wsConnected).toBe(true);
  });
});
