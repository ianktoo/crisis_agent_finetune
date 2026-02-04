#!/bin/bash
# Model optimization script for Ollama/LM Studio
# Exports model with optimal quantization settings

set -e

# Default values
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/final}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gguf}"
OLLAMA="${OLLAMA:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint|-c)
      CHECKPOINT="$2"
      shift 2
      ;;
    --quantization|-q)
      QUANTIZATION="$2"
      shift 2
      ;;
    --output|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --ollama)
      OLLAMA="true"
      shift
      ;;
    --small)
      QUANTIZATION="q3_k_m"
      shift
      ;;
    --balanced)
      QUANTIZATION="q4_k_m"
      shift
      ;;
    --quality)
      QUANTIZATION="q5_k_m"
      shift
      ;;
    --help|-h)
      echo "Model Optimization Script"
      echo ""
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -c, --checkpoint PATH    Checkpoint path (default: outputs/checkpoints/final)"
      echo "  -q, --quantization TYPE  Quantization: q3_k_m, q4_k_m, q5_k_m, q8_0 (default: q4_k_m)"
      echo "  -o, --output PATH        Output directory (default: outputs/gguf)"
      echo "  --ollama                 Create Ollama Modelfile"
      echo "  --small                  Use q3_k_m (smallest, ~3GB)"
      echo "  --balanced               Use q4_k_m (recommended, ~4GB)"
      echo "  --quality                Use q5_k_m (higher quality, ~5GB)"
      echo "  -h, --help               Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --small --ollama              # Smallest model for Ollama"
      echo "  $0 --balanced                   # Recommended balanced model"
      echo "  $0 -q q5_k_m --output gguf-5bit # Custom quantization"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
  echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
  echo "Available checkpoints:"
  ls -d outputs/checkpoints/* 2>/dev/null || echo "  (none found)"
  exit 1
fi

# Set executable directory
export CRISIS_GGUF_EXEC_DIR="${CRISIS_GGUF_EXEC_DIR:-/home/jovyan}"

# Show configuration
echo -e "${GREEN}Model Optimization${NC}"
echo "==================="
echo "Checkpoint:    $CHECKPOINT"
echo "Quantization:  $QUANTIZATION"
echo "Output:        $OUTPUT_DIR"
echo "Ollama:        $OLLAMA"
echo ""

# Extract model name from checkpoint path for informative filename
MODEL_NAME=$(basename "$CHECKPOINT")
if [ "$MODEL_NAME" = "final" ] || [ "$MODEL_NAME" = "checkpoint" ]; then
  # Try parent directory
  MODEL_NAME=$(basename "$(dirname "$CHECKPOINT")")
  if [ "$MODEL_NAME" = "checkpoints" ]; then
    MODEL_NAME="crisis-agent"
  fi
fi

# Build command
CMD="python scripts/export_gguf.py"
CMD="$CMD --checkpoint \"$CHECKPOINT\""
CMD="$CMD --output \"$OUTPUT_DIR\""
CMD="$CMD -q $QUANTIZATION"
CMD="$CMD --model-name \"$MODEL_NAME\""

if [ "$OLLAMA" = "true" ]; then
  CMD="$CMD --ollama"
fi

# Show expected size
case $QUANTIZATION in
  q3_k_m)
    EXPECTED_SIZE="~3GB"
    ;;
  q4_k_m)
    EXPECTED_SIZE="~4GB"
    ;;
  q5_k_m)
    EXPECTED_SIZE="~5GB"
    ;;
  q8_0)
    EXPECTED_SIZE="~8GB"
    ;;
  *)
    EXPECTED_SIZE="unknown"
    ;;
esac

echo -e "${YELLOW}Expected model size: $EXPECTED_SIZE${NC}"
echo ""
echo "Running export..."
echo ""

# Run export
eval $CMD

# Check result
if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}✓ Export completed successfully!${NC}"
  echo ""
  
  # Show file sizes and summary
  if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Exported files:"
    ls -lh "$OUTPUT_DIR"/*.gguf 2>/dev/null | awk '{printf "  %-60s %8s\n", $9, $5}'
    echo ""
    
    # Show summary using the export script
    echo "All exports in this directory:"
    python scripts/export_gguf.py --list-exports --output "$OUTPUT_DIR" 2>/dev/null || echo "  (run with --list-exports for details)"
  fi
  
  echo ""
  echo "Next steps:"
  if [ "$OLLAMA" = "true" ]; then
    echo "  Run: ollama run crisis-agent"
  else
    echo "  For Ollama: Run again with --ollama flag"
    echo "  For LM Studio: Import the GGUF file into LM Studio"
  fi
else
  echo ""
  echo -e "${RED}✗ Export failed${NC}"
  exit 1
fi
