# CUDA Version Information

## System CUDA Information

### NVIDIA Driver
- **Driver Version**: 565.57.01
- **CUDA Version (Driver)**: 12.7
- **GPU**: inference-ai GPU
- **GPU Memory**: 16,216 MiB (16GB)

### CUDA Toolkit
- **nvcc Version**: 12.4.131
- **Build Date**: Thu Mar 28 02:18:24 PDT 2024

## PyTorch CUDA Compatibility

### Current Installation (in conda environment)
- **PyTorch Version**: 2.10.0+cu128
- **CUDA Available**: ✅ Yes
- **PyTorch CUDA Version**: 12.8
- **cuDNN Version**: 91002 (9.1.0.2)

## Compatibility Notes

✅ **Compatible**: PyTorch 2.10.0 with CUDA 12.8 is compatible with:
- NVIDIA Driver 565.57.01 (supports CUDA 12.7)
- CUDA Toolkit 12.4
- GPU: inference-ai GPU (16GB VRAM)

### Version Compatibility Matrix

| Component | Version | Status |
|-----------|---------|--------|
| NVIDIA Driver | 565.57.01 | ✅ Compatible |
| CUDA Driver Support | 12.7 | ✅ Compatible |
| CUDA Toolkit (nvcc) | 12.4 | ✅ Compatible |
| PyTorch CUDA | 12.8 | ✅ Compatible |
| cuDNN | 9.1.0.2 | ✅ Compatible |

## Conda Environment

- **Environment Name**: `crisis_agent`
- **Python Version**: 3.10.19
- **Location**: `/home/jovyan/miniconda3/envs/crisis_agent`

## Activation

To activate the conda environment:

```bash
conda activate crisis_agent
```

Or if conda is not initialized:

```bash
eval "$(conda shell.bash hook)"
conda activate crisis_agent
```

## Notes

- PyTorch 2.10.0 with CUDA 12.8 is backward compatible with CUDA 12.4 toolkit
- The GPU has 16GB VRAM, which is sufficient for the training configuration
- All CUDA components are properly configured and compatible
