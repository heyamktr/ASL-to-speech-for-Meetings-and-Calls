// MAIN-world script — runs in the page's JS context (NOT the extension context).
// Overrides navigator.mediaDevices.getUserMedia so we can mix synthesized TTS
// audio into the outgoing mic stream for voice output (Phase 2).

const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

navigator.mediaDevices.getUserMedia = async function (constraints) {
  const stream = await original(constraints);
  // TODO: when audio is requested, mix in a MediaStreamAudioDestinationNode
  //       driven by speechSynthesis output and return the combined stream.
  return stream;
};
