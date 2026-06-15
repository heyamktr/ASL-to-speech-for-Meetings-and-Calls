"""TTS endpoint tests.

The TTS engine (espeak/SAPI) may not be installed in CI, so these tests cover the
contract: validation, the disabled path, and both the available (audio/wav) and
unavailable (503) engine outcomes via monkeypatching.
"""

from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.tts import TTSRequest, TTSUnavailable

client = TestClient(app)


def test_request_validation_rejects_empty_text() -> None:
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        TTSRequest(text="")


def test_tts_returns_wav_when_engine_available(monkeypatch) -> None:
    monkeypatch.setattr(main, "synthesize", lambda text, rate, voice: b"RIFFfake-wav")
    resp = client.post("/tts", json={"text": "hello world"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFFfake-wav"


def test_tts_returns_503_when_engine_unavailable(monkeypatch) -> None:
    def boom(text, rate, voice):
        raise TTSUnavailable("no engine")

    monkeypatch.setattr(main, "synthesize", boom)
    resp = client.post("/tts", json={"text": "hello"})
    assert resp.status_code == 503


def test_tts_rejects_overlong_text() -> None:
    long_text = "a" * (main.settings.tts_max_chars + 1)
    resp = client.post("/tts", json={"text": long_text})
    assert resp.status_code == 413
