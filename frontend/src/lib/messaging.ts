// Message types and helpers for content ↔ background communication.

export type Message =
  | { type: "TOGGLE_RECOGNITION"; enabled: boolean }
  | { type: "PREDICTED_WORD"; word: string };

export function sendMessage(msg: Message): Promise<unknown> {
  return chrome.runtime.sendMessage(msg);
}
