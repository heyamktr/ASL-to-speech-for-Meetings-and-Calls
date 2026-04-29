// Background service worker — runs in the extension context.
// Handles cross-tab state, message routing, and lifecycle events.

chrome.runtime.onInstalled.addListener(() => {
  console.log("ASL-to-Speech extension installed.");
});
