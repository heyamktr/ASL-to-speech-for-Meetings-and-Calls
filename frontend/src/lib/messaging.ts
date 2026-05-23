import type { ExtensionMessage } from './types';

export function sendToActiveTab(message: ExtensionMessage): void {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tabId = tabs[0]?.id;
    if (tabId == null) return;
    chrome.tabs.sendMessage(tabId, message);
  });
}

export function onMessage(
  handler: (message: ExtensionMessage) => void
): void {
  chrome.runtime.onMessage.addListener((message: ExtensionMessage) => {
    handler(message);
  });
}