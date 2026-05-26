let holistic: any = null;
let processing = false; // prevent overlapping calls

// Set up Trusted Types policy early to handle all dynamic script loading
const tt = (window as any).trustedTypes;
let policy: any = null;
if (tt?.createPolicy) {
  policy = tt.createPolicy('asl-mediapipe', {
    createScriptURL: (url: string) => url,
  });

  // Monkey-patch setAttribute to handle script src assignments through the policy
  const originalSetAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function(name: string, value: string) {
    if ((this as any).tagName === 'SCRIPT' && name === 'src' && policy) {
      return originalSetAttribute.call(this, name, policy.createScriptURL(value));
    }
    return originalSetAttribute.call(this, name, value);
  };
}

window.addEventListener('message', async (event) => {
  if (event.source !== window) return;
  if (event.data?.type !== 'ASL_INIT_MEDIAPIPE') return;

  const extensionUrl = event.data.extensionUrl as string;

  // Set up Module.locateFile globally for all loaders (including packed assets)
  (window as any).Module = {
    locateFile: (file: string) => {
      const url = `${extensionUrl}mediapipe/${file}`;
      console.log('[ASL Bridge] Module.locateFile:', file, '->', url);
      return url;
    },
  };

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    if (policy) {
      script.src = policy.createScriptURL(`${extensionUrl}mediapipe/holistic.js`);
    } else {
      script.src = `${extensionUrl}mediapipe/holistic.js`;
    }
    script.onload = () => resolve();
    script.onerror = reject;
    document.head.appendChild(script);
  });

  const Holistic = (window as any).Holistic;
  if (!Holistic) {
    console.error('[ASL Bridge] Holistic not on window after script load');
    return;
  }

  holistic = new Holistic({
    locateFile: (file: string) => {
      const url = `${extensionUrl}mediapipe/${file}`;
      console.log('[ASL Bridge] Holistic.locateFile:', file, '->', url);
      return url;
    },
  });

  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // Upper-body pose indices that match the model training layout
  // 11=left_shoulder 12=right_shoulder 13=left_elbow 14=right_elbow 15=left_wrist 16=right_wrist
  const POSE_JOINT_INDICES = [11, 12, 13, 14, 15, 16];

  holistic.onResults((results: any) => {
    processing = false; // ready for next frame
    const landmarks: number[] = [];

    // [0:63] Right hand — 21 landmarks × (x, y, z)
    if (results.rightHandLandmarks?.length) {
      results.rightHandLandmarks.forEach((lm: any) => landmarks.push(lm.x, lm.y, lm.z));
    } else {
      landmarks.push(...Array(63).fill(0));
    }

    // [63:126] Left hand — 21 landmarks × (x, y, z)
    if (results.leftHandLandmarks?.length) {
      results.leftHandLandmarks.forEach((lm: any) => landmarks.push(lm.x, lm.y, lm.z));
    } else {
      landmarks.push(...Array(63).fill(0));
    }

    // [126:144] Upper-body pose joints — 6 joints × (x, y, z) = 18
    if (results.poseLandmarks?.length) {
      POSE_JOINT_INDICES.forEach((i) => {
        const lm = results.poseLandmarks[i];
        landmarks.push(lm.x, lm.y, lm.z);
      });
    } else {
      landmarks.push(...Array(18).fill(0));
    }

    // Total: 63 + 63 + 18 = 144
    console.log('[ASL Bridge] Holistic detected, landmarks:', landmarks.length);
    window.postMessage({ type: 'ASL_LANDMARKS', landmarks }, '*');
  });

  console.log('[ASL Bridge] MediaPipe Holistic initialized');
  window.postMessage({ type: 'ASL_MEDIAPIPE_READY' }, '*');
});

window.addEventListener('message', async (event) => {
  if (event.source !== window) return;
  if (event.data?.type !== 'ASL_SEND_FRAME') return;
  if (!holistic || processing) return; // drop frame if still processing

  processing = true;
  try {
    await holistic.send({ image: event.data.bitmap });
  } catch (err) {
    processing = false;
    console.error('[ASL Bridge] holistic.send error:', err);
  }
});
