# Hardware Setup Guide

## Server Specifications

This pipeline is configured for the following hardware:

- **GPU**: 1x NVIDIA GPU with 16GB VRAM
- **CPU**: 4x CPU cores
- **RAM**: 32GB
- **Storage**: 100GB (Ephemeral)
- **Machine Image**: `harbor.service-inference.ai/library/pytorch:cu124_conda_test`

## Memory Optimization

### Model Memory Usage (16GB VRAM)

With the current configuration:

- **Base Model (4-bit quantized)**: ~4-5GB
- **LoRA Adapters**: ~200-300MB
- **Training Overhead** (batch_size=2, seq_len=2048): ~8-10GB
- **Total Usage**: ~12-15GB (safe margin for 16GB VRAM)

### If You Encounter OOM Errors

1. **Reduce batch size**:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 8  # Increase to maintain effective batch size
   ```

2. **Reduce sequence length**:
   ```yaml
   max_seq_length: 1024  # In model_config.yaml
   ```

3. **Reduce LoRA rank**:
   ```yaml
   lora:
     r: 8  # Instead of 16
     lora_alpha: 16
   ```

4. **Enable gradient checkpointing** (already enabled by default)

## Storage Management

⚠️ **Important**: Storage is ephemeral (100GB). Monitor disk usage:

```bash
# Check disk usage
df -h

# Clean old checkpoints if needed
rm -rf outputs/checkpoints/checkpoint-*
```

### Recommended Storage Allocation

- **Model checkpoints**: ~2-5GB per checkpoint (keep last 3)
- **Dataset cache**: ~5-20GB (depends on dataset size)
- **Logs**: ~100-500MB
- **Final merged model**: ~14GB (if merging LoRA)

**Total estimated**: ~30-50GB for a typical training run

## Performance Tuning

### CPU Utilization

- `dataloader_num_workers: 4` matches your 4 CPU cores
- This ensures efficient data loading without over-subscription

### GPU Utilization

- Monitor GPU usage: `nvidia-smi` or `watch -n 1 nvidia-smi`
- Target: 85-95% GPU utilization during training
- If lower, consider increasing batch size or sequence length

## Recommended Settings for 16GB VRAM

### Conservative (Safe)
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_seq_length: 1024
lora_r: 8
```

### Balanced (Current Default)
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_seq_length: 2048
lora_r: 16
```

### Aggressive (If you have headroom)
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
max_seq_length: 2048
lora_r: 32
```

## Monitoring Commands

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Disk usage
df -h
du -sh outputs/*

# Memory usage
free -h

# Process monitoring
htop
```

## Troubleshooting

### CUDA Out of Memory

1. Check current GPU usage: `nvidia-smi`
2. Reduce batch size in `training_config.yaml`
3. Reduce `max_seq_length` in `model_config.yaml`
4. Clear GPU cache: Restart Python process

### Disk Space Issues

1. Remove old checkpoints: `rm -rf outputs/checkpoints/checkpoint-*`
2. Clear dataset cache if needed: `rm -rf data/local_cache/*`
3. Compress logs: `gzip outputs/logs/*.log`

### Slow Training

1. Ensure `use_flash_attention_2: true` in model config
2. Check `dataloader_num_workers: 4` matches CPU count
3. Verify CUDA is being used: `python -c "import torch; print(torch.cuda.is_available())"`

## Expected Training Times

For a typical dataset (~1000-5000 samples):

- **Per epoch**: 30-120 minutes (depends on dataset size)
- **Full training (3 epochs)**: 1.5-6 hours
- **Evaluation**: 5-15 minutes
- **LoRA merge**: 10-30 minutes

## Cost Estimation

At $2/hour:
- **Full pipeline**: ~$3-12 per training run
- **Including evaluation**: ~$3.50-13
- **With LoRA merge**: ~$3.75-14
