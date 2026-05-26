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


settings = Settings()
