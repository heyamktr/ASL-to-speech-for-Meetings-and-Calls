.PHONY: help backend-dev backend-docker frontend-install frontend-dev frontend-build train evaluate export-onnx

help:
	@echo "ASL-to-Speech — top-level shortcut commands"
	@echo ""
	@echo "  backend-dev        Run FastAPI dev server (requires local venv)"
	@echo "  backend-docker     Run backend stack (FastAPI + Redis) in Docker"
	@echo "  frontend-install   Install frontend npm dependencies"
	@echo "  frontend-dev       Build extension in watch mode"
	@echo "  frontend-build     Production build of the extension"
	@echo "  train              Run the training pipeline"
	@echo "  evaluate           Evaluate the latest checkpoint on the test split"
	@echo "  export-onnx        Export a trained checkpoint to ONNX"

backend-dev:
	cd backend && uvicorn app.main:app --reload

backend-docker:
	cd backend && docker-compose up --build

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

train:
	cd model-training && python -m src.train

evaluate:
	cd model-training && python -m src.evaluate

export-onnx:
	cd model-training && python -m src.export_onnx
