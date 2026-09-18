"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    model_path: str = os.getenv("MODEL_PATH", "./models/asl_model.onnx")
    label_map_path: str = os.getenv("LABEL_MAP_PATH", "./models/label_map.json")
    model_seq_len: int = int(os.getenv("MODEL_SEQ_LEN", "100"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_connections: int = int(os.getenv("MAX_CONNECTIONS", "100"))

    # ── Adaptive sliding-window stride (Week 5 latency lever #1) ──────────────
    # Run inference every N frames while a hand is visible (active), but back off
    # to a larger stride when no hand is detected so idle sessions don't burn CPU.
    inference_stride: int = int(os.getenv("INFERENCE_STRIDE", "5"))
    idle_inference_stride: int = int(os.getenv("IDLE_INFERENCE_STRIDE", "15"))

    # ── Prediction cache (Week 5 latency lever #2) ───────────────────────────
    # Memoize recent inference results keyed by a quantized fingerprint of the
    # window so repeated/near-identical windows bypass the full ONNX run.
    prediction_cache_enabled: bool = os.getenv(
        "PREDICTION_CACHE_ENABLED", "1"
    ) not in ("0", "false", "False")
    prediction_cache_size: int = int(os.getenv("PREDICTION_CACHE_SIZE", "20"))
    prediction_cache_decimals: int = int(os.getenv("PREDICTION_CACHE_DECIMALS", "2"))

    # ── Text-to-speech endpoint (Week 5 latency lever #3 / voice output) ──────
    tts_enabled: bool = os.getenv("TTS_ENABLED", "1") not in ("0", "false", "False")
    tts_rate: int = int(os.getenv("TTS_RATE", "170"))         # words per minute
    tts_max_chars: int = int(os.getenv("TTS_MAX_CHARS", "300"))

    # ── LLM sentence refinement (ASL gloss → natural English) ─────────────────
    # A recognized utterance is a list of isolated WLASL words in signed order
    # ("gloss"). This layer rewrites that into one grammatical, punctuated English
    # sentence for the caption + voice output. Falls back to a deterministic local
    # join when disabled or when the API is unavailable (see app/refine.py).
    llm_enabled: bool = os.getenv("LLM_ENABLED", "1") not in ("0", "false", "False")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    llm_max_words: int = int(os.getenv("LLM_MAX_WORDS", "60"))


settings = Settings()
