"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    model_path: str = os.getenv("MODEL_PATH", "./models/asl_model.onnx")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
