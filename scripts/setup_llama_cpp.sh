#!/usr/bin/env bash
# Clone and build llama.cpp for GGUF export (convert_hf_to_gguf.py + llama-quantize).
# Usage:
#   ./scripts/setup_llama_cpp.sh
#   CRISIS_GGUF_EXEC_DIR=$HOME ./scripts/setup_llama_cpp.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_REPO="${LLAMA_CPP_REPO:-https://github.com/ggerganov/llama.cpp.git}"

# Use CRISIS_GGUF_EXEC_DIR if set (for noexec mounts), else project root
BASE_DIR="${CRISIS_GGUF_EXEC_DIR:-$PROJECT_ROOT}"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
LLAMA_DIR="$BASE_DIR/llama.cpp"

echo "=============================================="
echo "llama.cpp setup for GGUF export"
echo "=============================================="
echo "Target directory: $LLAMA_DIR"
echo ""

if [[ -d "$LLAMA_DIR/.git" ]]; then
  echo "llama.cpp already cloned at $LLAMA_DIR"
  cd "$LLAMA_DIR"
  git fetch --quiet 2>/dev/null || true
  git pull --quiet 2>/dev/null || true
else
  echo "Cloning llama.cpp..."
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
  cd "$LLAMA_DIR"
fi

if [[ ! -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]]; then
  echo "Warning: convert_hf_to_gguf.py not found in $LLAMA_DIR (repo layout may have changed)."
fi

echo "Building (Release, static)..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"

if [[ -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
  echo ""
  echo "=============================================="
  echo "Done. llama-quantize: $LLAMA_DIR/build/bin/llama-quantize"
  echo "convert_hf_to_gguf.py: $LLAMA_DIR/convert_hf_to_gguf.py"
  if [[ -n "$CRISIS_GGUF_EXEC_DIR" ]]; then
    echo "Using CRISIS_GGUF_EXEC_DIR=$CRISIS_GGUF_EXEC_DIR for export."
  else
    echo "Export will use project llama.cpp (e.g. make export-gguf)."
  fi
  echo "=============================================="
else
  echo "Build finished but llama-quantize not found at $LLAMA_DIR/build/bin/llama-quantize" >&2
  exit 1
fi
