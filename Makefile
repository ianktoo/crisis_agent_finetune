# Makefile for crisis-agent fine-tuning pipeline
# Usage: make <target>

.PHONY: help install train evaluate evaluate-ai evaluate-ai-openai evaluate-ai-gemini merge infer clean setup

# Default target
help:
	@echo "Crisis-Agent Fine-Tuning Pipeline"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make setup            - Install dependencies"
	@echo "  make train            - Run training"
	@echo "  make evaluate         - Evaluate model (standard)"
	@echo "  make evaluate-ai      - Evaluate with AI (Claude)"
	@echo "  make evaluate-ai-openai  - Evaluate with AI (OpenAI)"
	@echo "  make evaluate-ai-gemini  - Evaluate with AI (Gemini)"
	@echo "  make merge            - Merge LoRA weights"
	@echo "  make infer            - Run inference (interactive)"
	@echo "  make clean            - Clean output directories"
	@echo ""

# Install dependencies
setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed!"

# Run training
train:
	@echo "Starting training..."
	python scripts/train.py

# Evaluate model (standard)
evaluate:
	@echo "Evaluating model..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final

# Evaluate with AI (Claude)
evaluate-ai:
	@echo "Evaluating with AI (Claude)..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai

# Evaluate with AI (OpenAI)
evaluate-ai-openai:
	@echo "Evaluating with AI (OpenAI)..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider openai

# Evaluate with AI (Gemini)
evaluate-ai-gemini:
	@echo "Evaluating with AI (Gemini)..."
	python scripts/evaluate.py --checkpoint outputs/checkpoints/final --ai --ai-provider gemini

# Merge LoRA weights
merge:
	@echo "Merging LoRA weights..."
	python scripts/merge_lora.py --checkpoint outputs/checkpoints/final --output outputs/final_model

# Run inference (interactive)
infer:
	@echo "Starting inference (interactive mode)..."
	python scripts/infer.py --checkpoint outputs/checkpoints/final

# Clean output directories
clean:
	@echo "Cleaning output directories..."
	rm -rf outputs/checkpoints/*
	rm -rf outputs/logs/*
	rm -rf outputs/final_model/*
	@echo "Cleaned!"

# Full pipeline: train -> evaluate -> merge
pipeline: train evaluate merge
	@echo "Full pipeline completed!"

# Verify setup before training
verify:
	@echo "Verifying setup..."
	python scripts/verify_setup.py

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run only unit tests
test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

# Run only integration tests
test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

# Upload model to Hugging Face
upload-hf:
	@echo "Uploading model to Hugging Face..."
	@echo "Usage: make upload-hf CHECKPOINT=path REPO=username/repo-name"
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(REPO)" ]; then \
		echo "Error: CHECKPOINT and REPO required"; \
		echo "Example: make upload-hf CHECKPOINT=outputs/final_model REPO=username/crisis-agent-v1"; \
		exit 1; \
	fi
	python scripts/upload_to_hf.py --checkpoint $(CHECKPOINT) --repo-name $(REPO)
