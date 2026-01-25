"""
Convert the existing dataset from structured text to JSON format.
This creates a new dataset with JSON responses that can be used for training.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset, Dataset, DatasetDict
from src.data.load_dataset import load_dataset_from_config


def parse_structured_text_to_json(text: str) -> Dict[str, Any]:
    """
    Parse structured text response into JSON format.
    
    Example input:
    FACTS:
      • Fact 1
      • Fact 2
    UNCERTAINTIES:
      • Uncertainty 1
    ANALYSIS:
      • Analysis point
    GUIDANCE:
      • Guidance item
    
    Returns:
        Dictionary with facts, uncertainties, analysis, guidance, and confidence
    """
    result = {
        "facts": [],
        "uncertainties": [],
        "analysis": [],
        "guidance": [],
        "confidence": None
    }
    
    # Extract confidence if present
    confidence_match = re.search(r'Confidence:\s*([\d.]+)', text, re.IGNORECASE)
    if confidence_match:
        try:
            result["confidence"] = float(confidence_match.group(1))
        except:
            pass
    
    # Split by sections
    sections = {
        "facts": re.compile(r'FACTS?:?\s*\n', re.IGNORECASE),
        "uncertainties": re.compile(r'UNCERTAINTIES?:?\s*\n', re.IGNORECASE),
        "analysis": re.compile(r'ANALYSIS?:?\s*\n', re.IGNORECASE),
        "guidance": re.compile(r'GUIDANCE?:?\s*\n', re.IGNORECASE),
    }
    
    current_section = None
    current_items = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line starts a new section
        section_found = False
        for section_name, pattern in sections.items():
            if pattern.match(line) or line.upper().startswith(section_name.upper() + ':'):
                # Save previous section
                if current_section:
                    result[current_section] = current_items
                # Start new section
                current_section = section_name
                current_items = []
                section_found = True
                break
        
        if not section_found and current_section:
            # This is a bullet point or item
            # Skip confidence lines
            if re.match(r'Confidence:\s*[\d.]+', line, re.IGNORECASE):
                continue
            # Remove bullet markers (•, -, *, etc.)
            cleaned = re.sub(r'^[•\-\*]\s*', '', line)
            if cleaned and not re.match(r'Confidence:\s*[\d.]+', cleaned, re.IGNORECASE):
                current_items.append(cleaned)
    
    # Save last section
    if current_section:
        result[current_section] = current_items
    
    return result


def convert_dataset_to_json(
    input_dataset: DatasetDict,
    output_path: Path = Path("data/crisis_response_json.json")
) -> DatasetDict:
    """
    Convert dataset from structured text to JSON format.
    
    Args:
        input_dataset: Original dataset with text responses
        output_path: Path to save converted dataset
        
    Returns:
        New DatasetDict with JSON responses
    """
    converted_splits = {}
    
    for split_name, split_data in input_dataset.items():
        print(f"\nConverting {split_name} split ({len(split_data)} samples)...")
        
        converted_records = []
        success_count = 0
        error_count = 0
        
        for idx, record in enumerate(split_data):
            try:
                # Get the original response
                original_response = record.get("Output") or record.get("response", "")
                
                # Parse to JSON
                json_response = parse_structured_text_to_json(original_response)
                
                # Create new record with JSON response
                new_record = {
                    "instruction": record.get("Input") or record.get("instruction", ""),
                    "response": json.dumps(json_response, indent=2, ensure_ascii=False),
                    "original_response": original_response,  # Keep original for reference
                    "category": record.get("category", ""),
                    "role": record.get("role", ""),
                }
                
                converted_records.append(new_record)
                success_count += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(split_data)} samples...")
                    
            except Exception as e:
                error_count += 1
                print(f"  Error converting sample {idx}: {e}")
                if error_count > 10:
                    print(f"  Too many errors, stopping conversion of {split_name}")
                    break
        
        converted_splits[split_name] = Dataset.from_list(converted_records)
        print(f"  ✓ Converted {success_count} samples, {error_count} errors")
    
    # Save to file
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving converted dataset to {output_path}...")
        # Save as JSON lines
        with open(output_path, 'w', encoding='utf-8') as f:
            for split_name, split_data in converted_splits.items():
                for record in split_data:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
        print(f"✓ Saved {len(converted_splits)} splits to {output_path}")
    
    return DatasetDict(converted_splits)


def main():
    print("=" * 80)
    print("DATASET CONVERSION: Structured Text → JSON")
    print("=" * 80)
    
    # Load original dataset
    print("\n[1] Loading original dataset...")
    try:
        original_dataset = load_dataset_from_config()
        print(f"✓ Loaded dataset with splits: {list(original_dataset.keys())}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Show sample conversion
    print("\n[2] Testing conversion on sample...")
    sample = original_dataset["train"][0]
    original_response = sample.get("Output", "")
    print("Original response (first 300 chars):")
    print(original_response[:300])
    
    json_response = parse_structured_text_to_json(original_response)
    print("\nConverted JSON:")
    print(json.dumps(json_response, indent=2, ensure_ascii=False))
    
    # Convert entire dataset
    print("\n[3] Converting entire dataset...")
    converted_dataset = convert_dataset_to_json(
        original_dataset,
        output_path=Path("data/crisis_response_json.jsonl")
    )
    
    # Show statistics
    print("\n[4] Conversion Statistics:")
    print("-" * 80)
    for split_name, split_data in converted_dataset.items():
        print(f"{split_name}: {len(split_data)} samples")
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Response type: {type(sample.get('response', ''))}")
            try:
                parsed = json.loads(sample.get('response', '{}'))
                print(f"  ✓ Response is valid JSON with keys: {list(parsed.keys())}")
            except:
                print(f"  ✗ Response is not valid JSON")
    
    print("\n" + "=" * 80)
    print("Conversion complete!")
    print(f"Converted dataset saved to: data/crisis_response_json.jsonl")
    print("\nTo use this dataset, update configs/dataset_config.yaml:")
    print("  - Set hf_dataset_name to load from local file")
    print("  - Or upload to Hugging Face and use the new dataset name")


if __name__ == "__main__":
    main()
