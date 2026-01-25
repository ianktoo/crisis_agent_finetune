"""
Script to inspect the dataset structure and samples.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.data.load_dataset import load_dataset_from_config
from src.data.format_records import format_dataset
import json

def main():
    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)
    
    # Load raw dataset
    print("\n[1] Loading raw dataset from Hugging Face...")
    try:
        raw_dataset = load_dataset_from_config()
        print(f"✓ Loaded dataset successfully")
        print(f"  Splits: {list(raw_dataset.keys())}")
        for split_name, split_data in raw_dataset.items():
            print(f"  {split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                print(f"    Columns: {list(split_data[0].keys())}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Show raw samples
    print("\n[2] Raw dataset samples:")
    print("-" * 80)
    for split_name in ["train", "validation", "test"]:
        if split_name in raw_dataset and len(raw_dataset[split_name]) > 0:
            print(f"\n{split_name.upper()} split - Sample 0:")
            sample = raw_dataset[split_name][0]
            print(json.dumps(sample, indent=2, ensure_ascii=False))
            
            if len(raw_dataset[split_name]) > 1:
                print(f"\n{split_name.upper()} split - Sample 1:")
                sample = raw_dataset[split_name][1]
                print(json.dumps(sample, indent=2, ensure_ascii=False))
            break
    
    # Show formatted samples
    print("\n[3] Formatted dataset samples:")
    print("-" * 80)
    try:
        formatted_dataset = format_dataset(raw_dataset)
        for split_name in ["train", "validation", "test"]:
            if split_name in formatted_dataset and len(formatted_dataset[split_name]) > 0:
                print(f"\n{split_name.upper()} split - Sample 0:")
                sample = formatted_dataset[split_name][0]
                print(f"Keys: {list(sample.keys())}")
                print(f"\nInstruction:")
                print(sample.get("instruction", "N/A")[:500])
                print(f"\nResponse:")
                print(sample.get("response", "N/A")[:500])
                print(f"\nFull text (first 800 chars):")
                print(sample.get("text", "N/A")[:800])
                break
    except Exception as e:
        print(f"✗ Error formatting dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Check for JSON in responses
    print("\n[4] Checking response formats:")
    print("-" * 80)
    json_count = 0
    text_count = 0
    for split_name, split_data in raw_dataset.items():
        for idx in range(min(10, len(split_data))):
            sample = split_data[idx]
            response = sample.get("Output") or sample.get("response") or sample.get("Response", "")
            if isinstance(response, str):
                try:
                    json.loads(response)
                    json_count += 1
                    if json_count == 1:
                        print(f"✓ Found JSON response in {split_name}[{idx}]")
                        print(f"  Preview: {response[:200]}...")
                except:
                    text_count += 1
                    if text_count == 1:
                        print(f"✗ Found text response in {split_name}[{idx}]")
                        print(f"  Preview: {response[:200]}...")
    
    print(f"\nSummary: {json_count} JSON responses, {text_count} text responses (in first 10 samples)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
