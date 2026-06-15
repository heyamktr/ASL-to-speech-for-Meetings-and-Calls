FROM python:3.11-slim

WORKDIR /app

# curl: container healthcheck. espeak-ng + libespeak1: offline TTS engine for /tts.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl espeak-ng libespeak1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
