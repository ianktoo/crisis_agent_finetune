# Makefile for crisis-agent fine-tuning pipeline
# Usage: make <target>

.PHONY: help install train evaluate evaluate-ai evaluate-ai-openai evaluate-ai-gemini merge infer clean setup export-gguf export-ollama export-lmstudio optimize optimize-small optimize-balanced optimize-quality

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
	@echo "  make export-gguf      - Export to GGUF format (for LM Studio)"
	@echo "  make export-ollama    - Export to GGUF and setup Ollama"
	@echo "  make export-lmstudio  - Export to GGUF for LM Studio (q8_0)"
	@echo "  make optimize         - Optimize model for Ollama/LM Studio (recommended)"
	@echo "  make optimize-small   - Smallest model (~3GB, q3_k_m)"
	@echo "  make optimize-balanced - Balanced model (~4GB, q4_k_m, recommended)"
	@echo "  make optimize-quality  - Higher quality model (~5GB, q5_k_m)"
	@echo "  make list-exports     - List all GGUF exports with sizes"
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

# Export to GGUF format (default: q4_k_m quantization)
export-gguf:
	@echo "Exporting model to GGUF format..."
	CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf

# Export to GGUF and setup for Ollama
export-ollama:
	@echo "Exporting model to GGUF for Ollama..."
	CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf --ollama

# Export to GGUF for LM Studio (higher quality q8_0)
export-lmstudio:
	@echo "Exporting model to GGUF for LM Studio (q8_0)..."
	CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf -q q8_0

# Export to GGUF with custom settings
# Usage: make export-gguf-custom CHECKPOINT=path QUANT=q4_k_m OUTPUT=outputs/gguf
export-gguf-custom:
	@echo "Exporting model to GGUF with custom settings..."
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT required"; \
		echo "Example: make export-gguf-custom CHECKPOINT=outputs/checkpoints/final QUANT=q8_0"; \
		exit 1; \
	fi
	CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py \
		--checkpoint $(CHECKPOINT) \
		--output $(or $(OUTPUT),outputs/gguf) \
		-q $(or $(QUANT),q4_k_m) \
		$(if $(OLLAMA),--ollama)

# Push GGUF to Hugging Face Hub
upload-gguf:
	@echo "Pushing GGUF to Hugging Face Hub..."
	@echo "Usage: make upload-gguf CHECKPOINT=path REPO=username/repo-name-gguf"
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(REPO)" ]; then \
		echo "Error: CHECKPOINT and REPO required"; \
		echo "Example: make upload-gguf CHECKPOINT=outputs/checkpoints/final REPO=username/crisis-agent-gguf"; \
		exit 1; \
	fi
	CRISIS_GGUF_EXEC_DIR=/home/jovyan python scripts/export_gguf.py --checkpoint $(CHECKPOINT) --push-to-hub $(REPO) -q $(or $(QUANT),q4_k_m)

# Optimize model for Ollama/LM Studio (recommended: q4_k_m)
optimize:
	@echo "Optimizing model for Ollama/LM Studio (balanced: q4_k_m)..."
	@bash scripts/optimize_model.sh --balanced --ollama

# Optimize model - smallest size (~3GB)
optimize-small:
	@echo "Optimizing model - smallest size (q3_k_m)..."
	@bash scripts/optimize_model.sh --small --ollama

# Optimize model - balanced (recommended, ~4GB)
optimize-balanced:
	@echo "Optimizing model - balanced (q4_k_m, recommended)..."
	@bash scripts/optimize_model.sh --balanced --ollama

# Optimize model - higher quality (~5GB)
optimize-quality:
	@echo "Optimizing model - higher quality (q5_k_m)..."
	@bash scripts/optimize_model.sh --quality --ollama

# List all GGUF exports
list-exports:
	@echo "Listing all GGUF exports..."
	@python scripts/export_gguf.py --list-exports --output outputs/gguf
