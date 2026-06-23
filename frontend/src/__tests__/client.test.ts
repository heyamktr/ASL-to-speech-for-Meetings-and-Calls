import { WSClient } from '../content/websocket/client';

class MockWebSocket {
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  sent: string[] = [];

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

let mockWs: MockWebSocket;

beforeEach(() => {
  mockWs = new MockWebSocket();
  (global as unknown as { WebSocket: unknown }).WebSocket = jest.fn(() => mockWs);
  (global as unknown as { WebSocket: { OPEN: number } }).WebSocket.OPEN = MockWebSocket.OPEN;
});

afterEach(() => {
  jest.clearAllMocks();
});

describe('WSClient.sendLandmarks', () => {
  test('rejects arrays shorter than 144 floats', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws');
    client.connect();
    mockWs.simulateOpen();

    client.sendLandmarks(new Array(100).fill(0));

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('100'));
    expect(mockWs.sent).toHaveLength(0);
    warnSpy.mockRestore();
  });

  test('rejects arrays longer than 144 floats', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws');
    client.connect();
    mockWs.simulateOpen();

    client.sendLandmarks(new Array(200).fill(0));

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('200'));
    expect(mockWs.sent).toHaveLength(0);
    warnSpy.mockRestore();
  });

  test('sends valid 144-float frames when connected', () => {
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws');
    client.connect();
    mockWs.simulateOpen();

    client.sendLandmarks(new Array(144).fill(0.5));

    expect(mockWs.sent).toHaveLength(1);
    const parsed = JSON.parse(mockWs.sent[0]);
    expect(parsed.landmarks).toHaveLength(144);
    expect(parsed.session_id).toBeDefined();
    expect(parsed.timestamp).toBeDefined();
  });
});

describe('WSClient pending message buffer', () => {
  test('buffers messages when not connected', () => {
    mockWs.readyState = MockWebSocket.CLOSED;
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws');

    client.sendLandmarks(new Array(144).fill(0));

    expect(mockWs.sent).toHaveLength(0);
  });

  test('caps pending buffer at 100 messages', () => {
    mockWs.readyState = MockWebSocket.CLOSED;
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws');

    for (let i = 0; i < 120; i++) {
      client.sendLandmarks(new Array(144).fill(i));
    }

    client.connect();
    mockWs.simulateOpen();

    expect(mockWs.sent.length).toBeLessThanOrEqual(100);
  });
});

describe('WSClient reconnection', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('attempts reconnect after disconnect', () => {
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws');
    client.connect();
    mockWs.simulateOpen();
    mockWs.close();

    jest.advanceTimersByTime(1500);

    expect(global.WebSocket).toHaveBeenCalledTimes(2);
  });

  test('fires onDisconnect callback when connection drops', () => {
    const onDisconnect = jest.fn();
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws', undefined, onDisconnect);
    client.connect();
    mockWs.simulateOpen();
    mockWs.close();

    expect(onDisconnect).toHaveBeenCalledTimes(1);
  });

  test('fires onConnect callback on successful connection', () => {
    const onConnect = jest.fn();
    const client = new WSClient(jest.fn(), 'ws://localhost:8000/ws', onConnect);
    client.connect();
    mockWs.simulateOpen();

    expect(onConnect).toHaveBeenCalledTimes(1);
  });
});

describe('WSClient.onmessage', () => {
  test('calls onPrediction for valid prediction messages', () => {
    const onPrediction = jest.fn();
    const client = new WSClient(onPrediction, 'ws://localhost:8000/ws');
    client.connect();
    mockWs.simulateOpen();

    mockWs.simulateMessage({ word: 'hello', confidence: 0.95 });

    expect(onPrediction).toHaveBeenCalledWith({ word: 'hello', confidence: 0.95 });
  });

  test('does not call onPrediction for error messages', () => {
    const onPrediction = jest.fn();
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const client = new WSClient(onPrediction, 'ws://localhost:8000/ws');
    client.connect();
    mockWs.simulateOpen();

    mockWs.simulateMessage({ error: 'invalid frame' });

    expect(onPrediction).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});
