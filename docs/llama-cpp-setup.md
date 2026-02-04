# llama.cpp Setup for GGUF Export

GGUF export (for LM Studio, Ollama) uses **llama.cpp** tools: `convert_hf_to_gguf.py` and `llama-quantize`. This repo does not ship llama.cpp; you clone and build it yourself.

## Option A: Clone inside the project (default)

If your project directory is on an **executable** filesystem (binaries can run):

```bash
cd /path/to/crisis_agent_finetune
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j$(nproc)
```

Expected layout:

- `llama.cpp/convert_hf_to_gguf.py` – used by Unsloth/export
- `llama.cpp/build/bin/llama-quantize` – used for quantization

Then run export as usual:

```bash
python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf
# or: make export-gguf
```

**Note:** The top-level `llama.cpp/` directory is in `.gitignore`, so it will not be committed.

## Option B: Clone in a separate executable directory (noexec mounts)

If the project is on a **noexec** mount (e.g. some shared drives), clone and build llama.cpp somewhere where execution is allowed (e.g. home):

```bash
# Example: use home directory
export CRISIS_GGUF_EXEC_DIR=$HOME
mkdir -p "$CRISIS_GGUF_EXEC_DIR"
cd "$CRISIS_GGUF_EXEC_DIR"
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j$(nproc)
```

Then set the env var when running export:

```bash
CRISIS_GGUF_EXEC_DIR=$HOME python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf
# or: CRISIS_GGUF_EXEC_DIR=$HOME make export-gguf
```

The script will use `$CRISIS_GGUF_EXEC_DIR/llama.cpp/` (e.g. `~/llama.cpp/`) for `convert_hf_to_gguf.py` and `llama-quantize`.

## What the export script expects

| Location | Source |
|----------|--------|
| `convert_hf_to_gguf.py` | Copied from `<llama.cpp>/convert_hf_to_gguf.py` into the exec dir if missing |
| `llama-quantize` | `<llama.cpp>/build/bin/llama-quantize` or `<llama.cpp>/llama-quantize`; copied into exec dir if needed |

- **With llama.cpp in project:** use `project_root/llama.cpp` (Option A).
- **With CRISIS_GGUF_EXEC_DIR:** use `$CRISIS_GGUF_EXEC_DIR/llama.cpp` (Option B).

## Quick setup script (optional)

From the project root:

```bash
./scripts/setup_llama_cpp.sh
```

This clones llama.cpp into the project `llama.cpp/` and builds it. If you prefer the exec-dir layout, set `CRISIS_GGUF_EXEC_DIR` before running:

```bash
export CRISIS_GGUF_EXEC_DIR=$HOME
./scripts/setup_llama_cpp.sh
```

## Troubleshooting

- **"No working quantizer found in llama.cpp"** – Build llama.cpp and ensure `llama-quantize` exists and is executable; if the project disk is noexec, use Option B and `CRISIS_GGUF_EXEC_DIR`.
- **"Permission denied"** – Run from a directory that allows executing binaries, or use `CRISIS_GGUF_EXEC_DIR` on an executable path.
- **convert_hf_to_gguf.py** – The export script can copy it from your llama.cpp clone into the exec dir; ensure the clone contains `convert_hf_to_gguf.py` (it’s in the root of the llama.cpp repo).

See also: [Deployment](../wiki/Deployment.md), [Troubleshooting](../wiki/Troubleshooting.md).
