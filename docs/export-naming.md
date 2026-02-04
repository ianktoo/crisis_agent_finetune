# GGUF Export Naming and File Size Display

## Overview

The export script now automatically creates versionable, informative filenames and displays file sizes prominently.

## File Naming Format

Exports are automatically renamed with a structured format:

```
{model-name}-{quantization}-{date}.gguf
```

### Examples

- `crisis-agent-v1-q4_k_m-20260204.gguf`
- `final-q8_0-20260204.gguf`
- `crisis-agent-q3_k_m-20260204.gguf`

### Components

1. **Model Name**: Extracted from checkpoint path (or provided via `--model-name`)
   - Default: `crisis-agent`
   - Custom: Use `--model-name` flag

2. **Quantization**: The quantization method used
   - Examples: `q4_k_m`, `q8_0`, `q3_k_m`

3. **Date**: Export date in YYYYMMDD format
   - Example: `20260204` for February 4, 2026

### Version Handling

If a file with the same name already exists, a version number is appended:
- `crisis-agent-v1-q4_k_m-20260204-v1.gguf`
- `crisis-agent-v1-q4_k_m-20260204-v2.gguf`

## File Size Display

File sizes are displayed in multiple formats for clarity:

```
================================================================================
GGUF Export Complete
================================================================================
File:     crisis-agent-v1-q4_k_m-20260204.gguf
Path:     /path/to/outputs/gguf/crisis-agent-v1-q4_k_m-20260204.gguf
Size:     4.23 GB (4,230 MB, 4,540,000,000 bytes)
          (4.230 GB, 4230 MB, 4540000000 bytes)
================================================================================
```

### Size Formatting

The `_format_file_size()` function provides:
- **GB**: Gigabytes (3 decimal places)
- **MB**: Megabytes (rounded)
- **Bytes**: Exact byte count
- **Formatted**: Human-readable string (e.g., "4.23 GB (4,230 MB)")

## Listing All Exports

View all existing exports with their sizes:

```bash
# Using Makefile
make list-exports

# Using script directly
python scripts/export_gguf.py --list-exports --output outputs/gguf
```

### Output Format

```
====================================================================================================
Existing GGUF Exports
====================================================================================================
Filename                                                          Size                 Modified
----------------------------------------------------------------------------------------------------
crisis-agent-v1-q4_k_m-20260204.gguf                             4.23 GB (4230 MB)   2026-02-04 14:30
crisis-agent-v1-q8_0-20260203.gguf                               8.15 GB (8150 MB)   2026-02-03 10:15
crisis-agent-v1-q3_k_m-20260202.gguf                             3.12 GB (3120 MB)   2026-02-02 16:45
----------------------------------------------------------------------------------------------------
Total                                                             15.50 GB (15500 MB)  3 files
====================================================================================================
```

## Usage Examples

### Basic Export (Auto-naming)

```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m
```

Creates: `final-q4_k_m-20260204.gguf`

### Custom Model Name

```bash
python scripts/export_gguf.py \
  --checkpoint outputs/checkpoints/final \
  --output outputs/gguf \
  -q q4_k_m \
  --model-name crisis-agent-v1
```

Creates: `crisis-agent-v1-q4_k_m-20260204.gguf`

### Using Makefile

```bash
# Uses informative naming automatically
make optimize-balanced

# List all exports
make list-exports
```

## Benefits

1. **Version Tracking**: Date in filename makes it easy to track when exports were created
2. **Quick Identification**: Quantization level visible in filename
3. **No Conflicts**: Automatic versioning prevents overwriting
4. **Size Awareness**: Clear size display helps with storage planning
5. **Easy Management**: List command shows all exports at a glance

## Technical Details

### Filename Generation

The `_generate_informative_filename()` function:
- Sanitizes model names (removes special characters)
- Handles default names (`final`, `checkpoint`)
- Optionally includes date
- Creates clean, filesystem-safe names

### File Renaming

The `_rename_gguf_file()` function:
- Renames after export (Unsloth generates generic names)
- Handles conflicts with version numbers
- Preserves original if rename fails

### Size Formatting

The `_format_file_size()` function:
- Converts bytes to GB/MB
- Provides formatted string
- Returns dict with all formats for flexibility

## Integration

These features are automatically used by:
- `scripts/export_gguf.py` - Main export script
- `scripts/optimize_model.sh` - Optimization helper
- `scripts/train.py` - In-memory export after training
- `Makefile` targets - All export commands

No changes needed to existing workflows - the improvements are automatic!
