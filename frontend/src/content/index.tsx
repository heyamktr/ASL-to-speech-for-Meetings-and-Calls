// Content script entry — injected into Meet/Zoom/Teams pages.
// Mounts the React overlay and (eventually) starts the MediaPipe → WebSocket pipeline.

import { createRoot } from "react-dom/client";
import App from "./overlay/App";

const container = document.createElement("div");
container.id = "asl-to-speech-overlay-root";
document.body.appendChild(container);

createRoot(container).render(<App />);
