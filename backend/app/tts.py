"""Server-side text-to-speech (Week 5).

Exposes ``synthesize()`` which turns a predicted sentence into a WAV byte string
the frontend can play / mix into the outgoing mic stream (voice injection).

Engine: ``pyttsx3`` — a fully **offline** TTS wrapper. On Linux/Docker it drives
``espeak-ng`` (installed in the image); on Windows it uses SAPI5; on macOS NSSS.
Because the engine is optional and platform-dependent, every failure path raises
``TTSUnavailable`` so the API layer can return a clean 503 instead of a 500 — the
endpoint stays "live" and testable even where no speech engine is installed.

The browser's Web Speech API (``speechSynthesis``) remains the primary voice path
in the extension; this endpoint is the server-side fallback / alternative.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field


class TTSUnavailable(RuntimeError):
    """Raised when no usable speech-synthesis engine is available."""


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Sentence to synthesize")
    rate: int | None = Field(default=None, ge=50, le=400, description="Words per minute")
    voice: str | None = Field(default=None, description="Engine voice id (optional)")


def available() -> bool:
    """Best-effort check that the TTS engine can be initialized."""
    try:
        import pyttsx3  # noqa: F401

        engine = pyttsx3.init()
        engine.stop()
        return True
    except Exception:
        return False


def list_voices() -> list[dict]:
    """Return available engine voices (id, name, languages)."""
    try:
        import pyttsx3
    except Exception as exc:  # pragma: no cover - import guard
        raise TTSUnavailable(f"pyttsx3 not installed: {exc}") from exc

    try:
        engine = pyttsx3.init()
        voices = [
            {
                "id": v.id,
                "name": getattr(v, "name", ""),
                "languages": [
                    lang.decode() if isinstance(lang, bytes) else lang
                    for lang in getattr(v, "languages", []) or []
                ],
            }
            for v in engine.getProperty("voices")
        ]
        engine.stop()
        return voices
    except Exception as exc:
        raise TTSUnavailable(f"could not enumerate voices: {exc}") from exc


def synthesize(text: str, rate: int | None = None, voice: str | None = None) -> bytes:
    """Synthesize ``text`` to WAV bytes. Raises ``TTSUnavailable`` on any failure.

    NOTE: pyttsx3's ``runAndWait`` is blocking; call this from a threadpool
    (e.g. ``starlette.concurrency.run_in_threadpool``) inside async handlers.
    """
    try:
        import pyttsx3
    except Exception as exc:  # pragma: no cover - import guard
        raise TTSUnavailable(f"pyttsx3 not installed: {exc}") from exc

    out_path: Path | None = None
    try:
        engine = pyttsx3.init()
        if rate is not None:
            engine.setProperty("rate", rate)
        if voice is not None:
            engine.setProperty("voice", voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)

        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        engine.stop()

        data = out_path.read_bytes()
        if not data:
            raise TTSUnavailable("engine produced empty audio")
        return data
    except TTSUnavailable:
        raise
    except Exception as exc:
        raise TTSUnavailable(f"synthesis failed: {exc}") from exc
    finally:
        if out_path is not None:
            out_path.unlink(missing_ok=True)
