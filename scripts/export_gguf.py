"""
Export script for GGUF format (LM Studio and Ollama).
Exports fine-tuned model to GGUF format for local deployment.

Usage:
    # Export to GGUF for LM Studio
    python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf

    # Export with specific quantization
    python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf --quantization q4_k_m

    # Export and create Ollama Modelfile
    python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --output outputs/gguf --ollama

    # Push to Hugging Face Hub
    python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --push-to-hub username/model-name
"""

import sys
import os
import argparse
import subprocess
import contextlib
from pathlib import Path
from typing import Optional, List, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import check_cuda_available


# Available quantization methods
QUANTIZATION_METHODS = {
    "q4_k_m": "4-bit quantization (recommended for most use cases)",
    "q8_0": "8-bit quantization (higher quality, larger size)",
    "q5_k_m": "5-bit quantization (balance between quality and size)",
    "q6_k": "6-bit quantization",
    "q3_k_m": "3-bit quantization (smallest, lower quality)",
    "q2_k": "2-bit quantization (experimental, lowest quality)",
    "f16": "Float16 (full precision, largest size)",
    "f32": "Float32 (original precision, very large)",
}

def _resolve_path(p: Path) -> Path:
    return p.expanduser().resolve()


def _format_file_size(size_bytes: int) -> dict:
    """
    Format file size in multiple units.
    
    Returns:
        dict with 'bytes', 'mb', 'gb', and 'formatted' keys
    """
    size_gb = size_bytes / (1024 ** 3)
    size_mb = size_bytes / (1024 ** 2)
    
    if size_gb >= 1:
        formatted = f"{size_gb:.2f} GB ({size_mb:.0f} MB)"
    elif size_mb >= 1:
        formatted = f"{size_mb:.2f} MB"
    else:
        formatted = f"{size_bytes:,} bytes"
    
    return {
        'bytes': size_bytes,
        'mb': size_mb,
        'gb': size_gb,
        'formatted': formatted
    }


def _generate_informative_filename(
    checkpoint_path: Path,
    quantization: str,
    model_name: Optional[str] = None,
    include_date: bool = True
) -> str:
    """
    Generate an informative filename for GGUF export.
    
    Format: {model_name}-{quantization}-{date}.gguf
    Example: crisis-agent-v1-q4_k_m-20260204.gguf
    
    Args:
        checkpoint_path: Path to checkpoint (used to extract model name)
        quantization: Quantization method
        model_name: Optional explicit model name (overrides checkpoint name)
        include_date: Whether to include date in filename
        
    Returns:
        Filename string (without path)
    """
    # Extract model name from checkpoint path if not provided
    if model_name is None:
        checkpoint_name = checkpoint_path.name
        # Clean up common checkpoint names
        if checkpoint_name in ['final', 'checkpoint']:
            # Try parent directory
            parent = checkpoint_path.parent.name
            if parent and parent != 'checkpoints':
                model_name = parent
            else:
                model_name = 'crisis-agent'
        else:
            model_name = checkpoint_name
    else:
        model_name = model_name
    
    # Sanitize model name (remove special chars, spaces)
    model_name = model_name.replace(' ', '-').replace('_', '-')
    # Remove any remaining special characters except hyphens
    model_name = ''.join(c if c.isalnum() or c == '-' else '' for c in model_name)
    
    # Build filename parts
    parts = [model_name, quantization]
    
    if include_date:
        date_str = datetime.now().strftime("%Y%m%d")
        parts.append(date_str)
    
    filename = '-'.join(parts) + '.gguf'
    return filename


def _rename_gguf_file(
    original_path: Path,
    new_filename: str,
    logger=None
) -> Path:
    """
    Rename GGUF file to informative name.
    
    Args:
        original_path: Path to original GGUF file
        new_filename: New filename (without path)
        
    Returns:
        Path to renamed file
    """
    new_path = original_path.parent / new_filename
    
    # If target exists, add a counter
    counter = 1
    base_new_path = new_path
    while new_path.exists():
        stem = base_new_path.stem
        suffix = base_new_path.suffix
        new_path = base_new_path.parent / f"{stem}-v{counter}{suffix}"
        counter += 1
    
    try:
        original_path.rename(new_path)
        if logger:
            logger.info(f"Renamed: {original_path.name} -> {new_path.name}")
        return new_path
    except Exception as e:
        if logger:
            logger.warning(f"Could not rename file: {e}")
        return original_path


@contextlib.contextmanager
def _chdir(path: Path):
    old_cwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old_cwd))


def _is_executable_file(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(str(path), os.X_OK)


def _try_run_help(binary: Path) -> bool:
    try:
        result = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 or "usage" in (result.stdout + result.stderr).lower()
    except Exception:
        return False


def _ensure_exec_llama_cpp(logger=None) -> Path:
    """
    Ensure Unsloth can execute llama.cpp tools (esp. llama-quantize).

    Some environments mount project/work volumes with `noexec`, which causes:
      - "Permission denied" when running llama.cpp/llama-quantize
      - "No working quantizer found in llama.cpp"

    Unsloth expects the quantizer at:
      ./llama.cpp/llama-quantize   (relative to current working directory)

    We solve this by:
      - Using CRISIS_GGUF_EXEC_DIR (or $HOME) as an executable working directory
      - Ensuring <exec_dir>/llama.cpp/llama-quantize exists and is runnable
      - Copying from the repo build output if present:
          <repo>/llama.cpp/build/bin/llama-quantize
    """
    exec_dir = os.environ.get("CRISIS_GGUF_EXEC_DIR") or os.path.expanduser("~")
    exec_dir = _resolve_path(Path(exec_dir))
    llama_exec_dir = exec_dir / "llama.cpp"
    llama_exec_dir.mkdir(parents=True, exist_ok=True)

    # Ensure converter script exists (Unsloth also checks for this file)
    converter_src = project_root / "llama.cpp" / "convert_hf_to_gguf.py"
    converter_dst = llama_exec_dir / "convert_hf_to_gguf.py"
    if converter_src.exists() and not converter_dst.exists():
        try:
            converter_dst.write_bytes(converter_src.read_bytes())
        except Exception as e:
            if logger:
                logger.warning(f"Could not copy convert_hf_to_gguf.py into {llama_exec_dir}: {e}")

    quant_dst = llama_exec_dir / "llama-quantize"
    if _is_executable_file(quant_dst) and _try_run_help(quant_dst):
        return exec_dir

    quant_src_candidates = [
        project_root / "llama.cpp" / "build" / "bin" / "llama-quantize",
        project_root / "llama.cpp" / "llama-quantize",
    ]
    quant_src = next((p for p in quant_src_candidates if p.exists()), None)
    if quant_src is None:
        raise RuntimeError(
            "Unsloth GGUF export requires llama.cpp's `llama-quantize` binary, but it was not found.\n"
            "Fix:\n"
            "  1) Build llama.cpp once via CMake:\n"
            "     cd llama.cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && cmake --build build -j\"$(nproc)\"\n"
            "  2) Re-run this export.\n"
            "Tip: You can also set CRISIS_GGUF_EXEC_DIR to an executable path (e.g. /home/jovyan)."
        )

    try:
        quant_dst.write_bytes(Path(quant_src).read_bytes())
        os.chmod(str(quant_dst), 0o755)
    except Exception as e:
        raise RuntimeError(f"Failed to place llama-quantize into {quant_dst}: {e}") from e

    if not _try_run_help(quant_dst):
        raise RuntimeError(
            "Found/created llama-quantize but cannot execute it (likely due to a `noexec` mount).\n"
            f"Tried: {quant_dst}\n"
            "Fix:\n"
            "  - Set CRISIS_GGUF_EXEC_DIR to a path on an executable filesystem (e.g. /home/jovyan)\n"
            "  - Re-run export_gguf.py"
        )

    if logger:
        logger.info(f"Using executable llama.cpp at: {llama_exec_dir}")
    return exec_dir


def export_to_gguf_in_memory(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    quantization: str = "q4_k_m",
    model_name: Optional[str] = None,
    logger=None
) -> Path:
    """
    Export an in-memory model to GGUF format (no disk load).
    Use after training to avoid reloading the model; saves memory and time.

    Args:
        model: Unsloth FastLanguageModel instance (in memory)
        tokenizer: Tokenizer instance
        output_dir: Directory to save GGUF files
        quantization: Quantization method (q4_k_m, q8_0, f16, etc.)
        logger: Logger instance

    Returns:
        Path to the exported GGUF file
    """
    output_dir = _resolve_path(Path(output_dir))
    if logger:
        logger.info(f"Exporting in-memory model to GGUF (quantization: {quantization})")

    output_dir.mkdir(parents=True, exist_ok=True)
    exec_dir = _ensure_exec_llama_cpp(logger=logger)
    with _chdir(exec_dir):
        model.save_pretrained_gguf(
            str(output_dir),
            tokenizer,
            quantization_method=quantization,
        )

    gguf_files = list(output_dir.glob("*.gguf"))
    if gguf_files:
        gguf_path = gguf_files[0]
        
        # Generate informative filename
        informative_name = _generate_informative_filename(
            checkpoint_path=Path("."),  # In-memory export, use generic name
            quantization=quantization,
            model_name=model_name or "crisis-agent"
        )
        
        # Rename to informative name
        gguf_path = _rename_gguf_file(gguf_path, informative_name, logger)
        
        # Display file size prominently
        if logger:
            size_info = _format_file_size(gguf_path.stat().st_size)
            logger.info("")
            logger.info("=" * 80)
            logger.info("GGUF Export Complete")
            logger.info("=" * 80)
            logger.info(f"File:     {gguf_path.name}")
            logger.info(f"Path:     {gguf_path}")
            logger.info(f"Size:     {size_info['formatted']}")
            logger.info(f"          ({size_info['gb']:.3f} GB, {size_info['mb']:.0f} MB, {size_info['bytes']:,} bytes)")
            logger.info("=" * 80)
            logger.info("")
        
        return gguf_path
    raise FileNotFoundError("GGUF file was not created")


def push_to_hub_gguf_in_memory(
    model: Any,
    tokenizer: Any,
    repo_name: str,
    quantization: str = "q4_k_m",
    private: bool = False,
    logger=None
) -> str:
    """
    Push an in-memory model to Hugging Face Hub as GGUF (no disk load).

    Args:
        model: Unsloth FastLanguageModel instance
        tokenizer: Tokenizer instance
        repo_name: Hugging Face repo name (username/repo-name)
        quantization: Quantization method
        private: Whether to make the repo private
        logger: Logger instance

    Returns:
        URL of the uploaded model
    """
    if logger:
        logger.info(f"Pushing GGUF to Hugging Face Hub: {repo_name} (quantization: {quantization})")

    exec_dir = _ensure_exec_llama_cpp(logger=logger)
    with _chdir(exec_dir):
        model.push_to_hub_gguf(
            repo_name,
            tokenizer,
            quantization_method=quantization,
            private=private,
        )

    hub_url = f"https://huggingface.co/{repo_name}"
    if logger:
        logger.info(f"Model uploaded: {hub_url}")
    return hub_url


def export_to_gguf(
    checkpoint_path: Path,
    output_dir: Path,
    quantization: str = "q4_k_m",
    max_seq_length: int = 2048,
    model_name: Optional[str] = None,
    logger = None
) -> Path:
    """
    Export model to GGUF format.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save GGUF files
        quantization: Quantization method (q4_k_m, q8_0, f16, etc.)
        max_seq_length: Maximum sequence length
        logger: Logger instance
        
    Returns:
        Path to the exported GGUF file
    """
    from unsloth import FastLanguageModel

    checkpoint_path = _resolve_path(checkpoint_path)
    output_dir = _resolve_path(output_dir)
    
    if logger:
        logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    if logger:
        logger.info(f"Model loaded successfully")
        logger.info(f"Exporting to GGUF with quantization: {quantization}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to GGUF (requires executable llama.cpp tools)
    exec_dir = _ensure_exec_llama_cpp(logger=logger)
    with _chdir(exec_dir):
        model.save_pretrained_gguf(
            str(output_dir),
            tokenizer,
            quantization_method=quantization
        )
    
    # Find the generated GGUF file
    gguf_files = list(output_dir.glob("*.gguf"))
    if gguf_files:
        gguf_path = gguf_files[0]
        
        # Generate informative filename from checkpoint path
        informative_name = _generate_informative_filename(
            checkpoint_path=checkpoint_path,
            quantization=quantization,
            model_name=model_name  # Use provided name or extract from checkpoint path
        )
        
        # Rename to informative name
        gguf_path = _rename_gguf_file(gguf_path, informative_name, logger)
        
        # Display file size prominently
        if logger:
            size_info = _format_file_size(gguf_path.stat().st_size)
            logger.info("")
            logger.info("=" * 80)
            logger.info("GGUF Export Complete")
            logger.info("=" * 80)
            logger.info(f"File:     {gguf_path.name}")
            logger.info(f"Path:     {gguf_path}")
            logger.info(f"Size:     {size_info['formatted']}")
            logger.info(f"          ({size_info['gb']:.3f} GB, {size_info['mb']:.0f} MB, {size_info['bytes']:,} bytes)")
            logger.info("=" * 80)
            logger.info("")
        
        return gguf_path
    else:
        raise FileNotFoundError("GGUF file was not created")


def push_to_hub_gguf(
    checkpoint_path: Path,
    repo_name: str,
    quantization: str = "q4_k_m",
    max_seq_length: int = 2048,
    private: bool = False,
    logger = None
) -> str:
    """
    Export model to GGUF and push to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        repo_name: Hugging Face repo name (username/repo-name)
        quantization: Quantization method
        max_seq_length: Maximum sequence length
        private: Whether to make the repo private
        logger: Logger instance
        
    Returns:
        URL of the uploaded model
    """
    from unsloth import FastLanguageModel

    checkpoint_path = _resolve_path(checkpoint_path)
    
    if logger:
        logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    if logger:
        logger.info(f"Pushing to Hugging Face Hub: {repo_name}")
        logger.info(f"Quantization: {quantization}")
    
    # Push to Hub (requires executable llama.cpp tools)
    exec_dir = _ensure_exec_llama_cpp(logger=logger)
    with _chdir(exec_dir):
        model.push_to_hub_gguf(
            repo_name,
            tokenizer,
            quantization_method=quantization,
            private=private
        )
    
    hub_url = f"https://huggingface.co/{repo_name}"
    if logger:
        logger.info(f"Model uploaded successfully: {hub_url}")
    
    return hub_url


def create_ollama_modelfile(
    gguf_path: Path,
    output_dir: Path,
    model_name: str = "crisis-agent",
    system_prompt: Optional[str] = None,
    logger = None
) -> Path:
    """
    Create an Ollama Modelfile for the exported GGUF.
    
    Args:
        gguf_path: Path to the GGUF file
        output_dir: Directory to save the Modelfile
        model_name: Name for the Ollama model
        system_prompt: Optional system prompt
        logger: Logger instance
        
    Returns:
        Path to the created Modelfile
    """
    if system_prompt is None:
        system_prompt = """You are AI Emergency Kit, a specialized AI assistant for crisis response and emergency management. 
You provide structured JSON responses with:
- situation_assessment: Analysis of the emergency situation
- immediate_actions: List of immediate steps to take
- resources_needed: Required resources and personnel
- safety_considerations: Important safety warnings
- communication_plan: How to coordinate and communicate

Always prioritize human safety and provide actionable, clear guidance."""

    modelfile_content = f'''# Ollama Modelfile for AI Emergency Kit (Crisis Agent)
# Generated by crisis-agent fine-tuning pipeline

FROM {gguf_path.name}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"

# System prompt
SYSTEM """{system_prompt}"""

# Template for chat format (Mistral Instruct format)
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""
'''

    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    
    if logger:
        logger.info(f"Modelfile created: {modelfile_path}")
    
    return modelfile_path


def list_existing_exports(
    output_dir: Path,
    logger = None
) -> List[dict]:
    """
    List all existing GGUF exports with their sizes.
    
    Args:
        output_dir: Directory to search for GGUF files
        logger: Logger instance
        
    Returns:
        List of dicts with 'path', 'name', 'size_info', 'modified' keys
    """
    output_dir = _resolve_path(output_dir)
    
    if not output_dir.exists():
        return []
    
    exports = []
    for gguf_file in sorted(output_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True):
        size_info = _format_file_size(gguf_file.stat().st_size)
        modified = datetime.fromtimestamp(gguf_file.stat().st_mtime)
        
        exports.append({
            'path': gguf_file,
            'name': gguf_file.name,
            'size_info': size_info,
            'modified': modified
        })
    
    return exports


def display_exports_summary(
    output_dir: Path,
    logger = None
):
    """
    Display a formatted summary of all GGUF exports.
    
    Args:
        output_dir: Directory containing GGUF files
        logger: Logger instance (optional, uses print if None)
    """
    exports = list_existing_exports(output_dir, logger)
    
    if not exports:
        msg = "No GGUF exports found in this directory."
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return
    
    # Use logger if available, otherwise print
    log_func = logger.info if logger else print
    
    log_func("")
    log_func("=" * 100)
    log_func("Existing GGUF Exports")
    log_func("=" * 100)
    log_func(f"{'Filename':<60} {'Size':<20} {'Modified':<20}")
    log_func("-" * 100)
    
    total_size = 0
    for exp in exports:
        log_func(f"{exp['name']:<60} {exp['size_info']['formatted']:<20} {exp['modified'].strftime('%Y-%m-%d %H:%M'):<20}")
        total_size += exp['size_info']['bytes']
    
    log_func("-" * 100)
    total_info = _format_file_size(total_size)
    log_func(f"{'Total':<60} {total_info['formatted']:<20} {len(exports)} files")
    log_func("=" * 100)
    log_func("")


def register_with_ollama(
    gguf_path: Path,
    modelfile_path: Path,
    model_name: str = "crisis-agent",
    logger = None
) -> bool:
    """
    Register the model with Ollama.
    
    Args:
        gguf_path: Path to the GGUF file
        modelfile_path: Path to the Modelfile
        model_name: Name to register the model as
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if Ollama is installed
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            if logger:
                logger.warning("Ollama is not installed or not in PATH")
            return False
        
        if logger:
            logger.info(f"Creating Ollama model: {model_name}")
        
        # Create the model
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            cwd=str(gguf_path.parent)
        )
        
        if result.returncode == 0:
            if logger:
                logger.info(f"Model registered with Ollama as: {model_name}")
                logger.info(f"Run with: ollama run {model_name}")
            return True
        else:
            if logger:
                logger.error(f"Failed to register with Ollama: {result.stderr}")
            return False
            
    except FileNotFoundError:
        if logger:
            logger.warning("Ollama is not installed. Install from: https://ollama.ai")
        return False


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export model to GGUF format for LM Studio and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default settings (q4_k_m quantization)
  python scripts/export_gguf.py --checkpoint outputs/checkpoints/final

  # Export with 8-bit quantization (higher quality)
  python scripts/export_gguf.py --checkpoint outputs/checkpoints/final -q q8_0

  # Export and register with Ollama
  python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --ollama

  # Push directly to Hugging Face Hub
  python scripts/export_gguf.py --checkpoint outputs/checkpoints/final --push-to-hub username/model-gguf

Quantization Methods:
  q4_k_m  - 4-bit (recommended, good balance)
  q8_0    - 8-bit (higher quality, larger)
  q5_k_m  - 5-bit (quality/size balance)
  q3_k_m  - 3-bit (smaller, lower quality)
  f16     - Float16 (full precision)
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/final",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/gguf",
        help="Output directory for GGUF files"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="q4_k_m",
        choices=list(QUANTIZATION_METHODS.keys()),
        help="Quantization method (default: q4_k_m)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Create Ollama Modelfile and register model"
    )
    parser.add_argument(
        "--ollama-name",
        type=str,
        default="crisis-agent",
        help="Name for Ollama model (default: crisis-agent)"
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO",
        help="Push GGUF to Hugging Face Hub (format: username/repo-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Hugging Face repo private (when using --push-to-hub)"
    )
    parser.add_argument(
        "--list-quantizations",
        action="store_true",
        help="List available quantization methods and exit"
    )
    parser.add_argument(
        "--list-exports",
        action="store_true",
        help="List existing GGUF exports and exit"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name for filename (default: extracted from checkpoint path)"
    )
    
    args = parser.parse_args()
    
    # List quantizations and exit
    if args.list_quantizations:
        print("\nAvailable Quantization Methods:")
        print("-" * 60)
        for method, desc in QUANTIZATION_METHODS.items():
            print(f"  {method:10s} - {desc}")
        print()
        return
    
    # Setup logging for list-exports
    if args.list_exports:
        logger = setup_logging()
        output_dir = Path(args.output)
        display_exports_summary(output_dir, logger)
        return
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("GGUF Export Pipeline")
    logger.info("=" * 80)
    
    try:
        # Check CUDA
        check_cuda_available()
        
        checkpoint_path = Path(args.checkpoint)
        output_dir = Path(args.output)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Quantization: {args.quantization} ({QUANTIZATION_METHODS[args.quantization]})")
        
        # Push to Hub if requested
        if args.push_to_hub:
            logger.info("\n" + "-" * 40)
            logger.info("Pushing to Hugging Face Hub")
            logger.info("-" * 40)
            
            hub_url = push_to_hub_gguf(
                checkpoint_path=checkpoint_path,
                repo_name=args.push_to_hub,
                quantization=args.quantization,
                max_seq_length=args.max_seq_length,
                private=args.private,
                logger=logger
            )
            
            logger.info(f"\nModel available at: {hub_url}")
        
        else:
            # Export locally
            logger.info("\n" + "-" * 40)
            logger.info("Exporting to GGUF")
            logger.info("-" * 40)
            
            gguf_path = export_to_gguf(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                quantization=args.quantization,
                max_seq_length=args.max_seq_length,
                model_name=args.model_name,
                logger=logger
            )
            
            # Create Ollama files if requested
            if args.ollama:
                logger.info("\n" + "-" * 40)
                logger.info("Setting up Ollama")
                logger.info("-" * 40)
                
                modelfile_path = create_ollama_modelfile(
                    gguf_path=gguf_path,
                    output_dir=output_dir,
                    model_name=args.ollama_name,
                    logger=logger
                )
                
                # Try to register with Ollama
                register_with_ollama(
                    gguf_path=gguf_path,
                    modelfile_path=modelfile_path,
                    model_name=args.ollama_name,
                    logger=logger
                )
            
            # Print usage instructions
            logger.info("\n" + "=" * 80)
            logger.info("Export Complete!")
            logger.info("=" * 80)
            
            logger.info(f"\nGGUF file: {gguf_path}")
            
            logger.info("\n--- LM Studio ---")
            logger.info(f"1. Import the GGUF file into LM Studio:")
            logger.info(f"   lms import {gguf_path}")
            logger.info(f"2. Or manually copy to: ~/.lmstudio/models/crisis-agent/")
            logger.info(f"3. Load in LM Studio and start chatting!")
            
            if args.ollama:
                logger.info("\n--- Ollama ---")
                logger.info(f"Run: ollama run {args.ollama_name}")
            else:
                logger.info("\n--- Ollama ---")
                logger.info(f"To use with Ollama, run again with --ollama flag:")
                logger.info(f"  python scripts/export_gguf.py --checkpoint {checkpoint_path} --ollama")
        
    except KeyboardInterrupt:
        logger.warning("\nExport interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nExport failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
