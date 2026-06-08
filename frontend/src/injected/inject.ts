// MAIN-world script — runs in the page's JS context (NOT the extension context).
// Overrides navigator.mediaDevices.getUserMedia so we can mix synthesized TTS
// audio into the outgoing mic stream for voice output.

const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

// Persistent AudioContext and destination shared across all calls
let audioCtx: AudioContext | null = null;
let mixDest: MediaStreamAudioDestinationNode | null = null;

// Current voice preferences — updated live from popup via postMessage
let voicePrefs = {
  rate: 1.0,
  pitch: 1.0,
  voiceURI: '',
  ttsEndpoint: 'http://localhost:8001/tts',
};

navigator.mediaDevices.getUserMedia = async function (constraints) {
  const stream = await original(constraints);

  // Only intercept when audio is requested (video-only calls: pass through)
  const wantsAudio = constraints && (
    constraints.audio === true ||
    (typeof constraints.audio === 'object' && constraints.audio !== null)
  );
  if (!wantsAudio || mixDest) return stream;

  // Build the AudioContext mixer
  audioCtx = new AudioContext();
  mixDest = audioCtx.createMediaStreamDestination();

  const micSource = audioCtx.createMediaStreamSource(stream);
  const gainNode = audioCtx.createGain();
  gainNode.gain.value = 1.0;

  micSource.connect(gainNode);
  gainNode.connect(mixDest);

  // Return a new stream: original video tracks + our mixed audio track
  // Meet will use this stream — all audio (mic + TTS) flows through mixDest
  return new MediaStream([
    ...stream.getVideoTracks(),
    ...mixDest.stream.getAudioTracks(),
  ]);
};

// Listen for words to speak — sent by the content script (index.tsx)
window.addEventListener('message', async (event) => {
  if (event.source !== window) return;
  if (event.data?.type !== 'ASL_SPEAK_WORD') return;
  if (!audioCtx || !mixDest) return;

  const { word, ttsEndpoint, rate, pitch } = event.data as {
    word: string;
    ttsEndpoint: string;
    rate: number;
    pitch: number;
  };

  // AudioContext may be suspended until a user gesture — resume it
  if (audioCtx.state === 'suspended') await audioCtx.resume();

  try {
    // Call Dev B's TTS endpoint — expects audio bytes back (WAV/MP3/OGG)
    const url = `${ttsEndpoint}?text=${encodeURIComponent(word)}&rate=${rate}&pitch=${pitch}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`TTS endpoint returned ${response.status}`);

    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(mixDest);  // inject directly into the mic stream Mix
    source.start();
  } catch (err) {
    // Fallback: speak to the system speaker (participants won't hear this,
    // but at least the ASL user hears their own word confirmed)
    console.warn('[ASL Inject] TTS endpoint failed, falling back to speechSynthesis:', err);
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.rate = rate;
    utterance.pitch = pitch;
    if (voicePrefs.voiceURI) {
      const match = speechSynthesis.getVoices().find(v => v.voiceURI === voicePrefs.voiceURI);
      if (match) utterance.voice = match;
    }
    speechSynthesis.speak(utterance);
  }
});

// Live voice prefs update — so popup changes take effect without reloading
window.addEventListener('message', (event) => {
  if (event.source !== window) return;
  if (event.data?.type !== 'ASL_SET_VOICE_PREFS') return;
  voicePrefs = { ...voicePrefs, ...event.data };
});