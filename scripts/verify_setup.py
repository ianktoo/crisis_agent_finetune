"""
Quick verification script to check if everything is ready for training.
Run this before starting training to catch any issues early.

Usage: python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check if all required packages are installed."""
    print("Checking imports...")
    try:
        import torch
        import datasets
        import transformers
        import yaml
        import unsloth
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("  Run: pip install -r requirements.txt")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available - training will be very slow on CPU")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_configs():
    """Check if configuration files exist and are valid."""
    print("\nChecking configuration files...")
    try:
        import yaml
        configs = [
            "configs/dataset_config.yaml",
            "configs/model_config.yaml",
            "configs/training_config.yaml"
        ]
        all_valid = True
        for config_path in configs:
            path = Path(config_path)
            if not path.exists():
                print(f"✗ Missing: {config_path}")
                all_valid = False
            else:
                try:
                    with open(path) as f:
                        yaml.safe_load(f)
                    print(f"✓ {config_path} is valid")
                except Exception as e:
                    print(f"✗ Invalid YAML in {config_path}: {e}")
                    all_valid = False
        return all_valid
    except Exception as e:
        print(f"✗ Error checking configs: {e}")
        return False

def check_dataset_config():
    """Check dataset configuration."""
    print("\nChecking dataset configuration...")
    try:
        import yaml
        with open("configs/dataset_config.yaml") as f:
            config = yaml.safe_load(f)
        
        dataset_name = config["dataset"]["hf_dataset_name"]
        print(f"✓ Dataset configured: {dataset_name}")
        
        if dataset_name == "your_username/crisis_dataset":
            print("  ⚠ WARNING: Dataset name not updated! Update configs/dataset_config.yaml")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking dataset config: {e}")
        return False

def check_env_file():
    """Check if .env file exists (optional)."""
    print("\nChecking environment setup...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_example.exists():
        print("✓ .env.example exists")
    
    if env_file.exists():
        print("✓ .env file exists")
        # Check if HF_TOKEN is set (basic check)
        with open(env_file) as f:
            content = f.read()
            if "HF_TOKEN" in content and "your_token" not in content.lower():
                print("  ✓ HF_TOKEN appears to be set")
            else:
                print("  ⚠ HF_TOKEN may not be set (required for private datasets)")
    else:
        print("  ⚠ .env file not found (optional if dataset is public)")
        print("  Create from template: cp .env.example .env")
    
    return True

def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    dirs = [
        "outputs/checkpoints",
        "outputs/logs",
        "data/local_cache"
    ]
    all_exist = True
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"  Creating: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✓ {dir_path} exists")
    return True

def main():
    """Run all checks."""
    print("=" * 80)
    print("Crisis-Agent Fine-Tuning Pipeline - Setup Verification")
    print("=" * 80)
    
    checks = [
        ("Imports", check_imports),
        ("CUDA", check_cuda),
        ("Configs", check_configs),
        ("Dataset Config", check_dataset_config),
        ("Environment", check_env_file),
        ("Directories", check_directories),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error in {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. Verify dataset column names match your dataset structure")
        print("  2. Run: python scripts/train.py")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above before training.")
        print("\nSee DEPLOYMENT.md for detailed setup instructions.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
