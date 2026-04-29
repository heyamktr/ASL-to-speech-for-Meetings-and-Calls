# Frontend — Chrome Extension (MV3)

The browser side of the project. Runs MediaPipe Hands on the webcam feed, streams 63 landmark numbers per frame to the backend over WebSocket, displays predicted words in an overlay panel, and (Phase 2) injects synthesized voice into the meeting's outgoing mic stream.

## Stack

- Chrome Extension Manifest V3
- React 18 + TypeScript
- esbuild (multi-entry bundler)
- Tailwind CSS (via PostCSS)

## Install

```sh
npm install
```

## Develop

```sh
npm run dev
```

Watches `src/` and rebuilds `dist/` on every save.

Then in Chrome:

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** and select the `frontend/dist/` directory
4. Open Google Meet (or another supported platform) and the overlay should appear

## Production build

```sh
npm run build
```

## Lint / format

```sh
npm run lint
npm run format
```

## Entry points

esbuild bundles four separate entry points into `dist/`:

| Entry | Purpose |
| --- | --- |
| `background/service-worker.ts` | MV3 background service worker |
| `content/index.tsx` | Content script — injects the overlay into Meet/Zoom/Teams |
| `popup/index.tsx` | Extension popup (toolbar icon) |
| `injected/inject.ts` | MAIN-world script — overrides `getUserMedia` for Phase 2 voice output |
