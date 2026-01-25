"""
Check if dataset has JSON responses, and convert if needed.
"""

import sys
import json
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset, Dataset, DatasetDict
from src.data.load_dataset import load_dataset_from_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def check_json_in_responses(dataset: DatasetDict, sample_size: int = 20) -> dict:
    """
    Check if responses in dataset are JSON format.
    
    Returns:
        Dictionary with statistics about JSON vs text responses
    """
    stats = {
        "total_checked": 0,
        "json_count": 0,
        "text_count": 0,
        "json_samples": [],
        "text_samples": [],
    }
    
    for split_name, split_data in dataset.items():
        check_count = min(sample_size, len(split_data))
        logger.info(f"Checking {check_count} samples from {split_name} split...")
        
        for idx in range(check_count):
            sample = split_data[idx]
            response = sample.get("Output") or sample.get("response") or sample.get("Response", "")
            
            if not response:
                continue
            
            stats["total_checked"] += 1
            
            # Try to parse as JSON
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    # Check if it's a dict/object (not just a string that happens to be valid JSON)
                    if isinstance(parsed, (dict, list)):
                        stats["json_count"] += 1
                        if len(stats["json_samples"]) < 3:
                            stats["json_samples"].append({
                                "split": split_name,
                                "idx": idx,
                                "preview": response[:200]
                            })
                        continue
                except:
                    pass
            
            # Not JSON - it's text
            stats["text_count"] += 1
            if len(stats["text_samples"]) < 3:
                stats["text_samples"].append({
                    "split": split_name,
                    "idx": idx,
                    "preview": response[:200]
                })
    
    return stats


def parse_structured_text_to_json(text: str) -> dict:
    """
    Parse structured text response into JSON format.
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
        
        # Skip confidence lines
        if re.match(r'Confidence:\s*[\d.]+', line, re.IGNORECASE):
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
    output_path: Path = None
) -> DatasetDict:
    """
    Convert dataset from structured text to JSON format.
    """
    converted_splits = {}
    
    for split_name, split_data in input_dataset.items():
        logger.info(f"\nConverting {split_name} split ({len(split_data)} samples)...")
        
        converted_records = []
        success_count = 0
        error_count = 0
        
        for idx, record in enumerate(split_data):
            try:
                # Get the original response
                original_response = record.get("Output") or record.get("response", "")
                
                # Skip if already JSON
                if isinstance(original_response, str):
                    try:
                        parsed = json.loads(original_response)
                        if isinstance(parsed, (dict, list)):
                            # Already JSON, keep as is
                            json_response = original_response
                        else:
                            # Parse text to JSON
                            json_obj = parse_structured_text_to_json(original_response)
                            json_response = json.dumps(json_obj, indent=2, ensure_ascii=False)
                    except:
                        # Not JSON, parse text to JSON
                        json_obj = parse_structured_text_to_json(original_response)
                        json_response = json.dumps(json_obj, indent=2, ensure_ascii=False)
                else:
                    # Already a dict/list, convert to JSON string
                    json_response = json.dumps(original_response, indent=2, ensure_ascii=False)
                
                # Create new record with JSON response
                new_record = {
                    "instruction": record.get("Input") or record.get("instruction", ""),
                    "response": json_response,
                    "original_response": original_response if original_response != json_response else None,
                    "category": record.get("category", ""),
                    "role": record.get("role", ""),
                }
                
                converted_records.append(new_record)
                success_count += 1
                
                if (idx + 1) % 500 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(split_data)} samples...")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"  Error converting sample {idx}: {e}")
                if error_count > 10:
                    logger.warning(f"  Too many errors, stopping conversion of {split_name}")
                    break
        
        converted_splits[split_name] = Dataset.from_list(converted_records)
        logger.info(f"  ✓ Converted {success_count} samples, {error_count} errors")
    
    # Save to JSONL file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nSaving converted dataset to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for split_name, split_data in converted_splits.items():
                for record in split_data:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
        logger.info(f"✓ Saved {len(converted_splits)} splits to {output_path}")
    
    return DatasetDict(converted_splits)


def main():
    print("=" * 80)
    print("DATASET CHECK AND CONVERSION")
    print("=" * 80)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    try:
        dataset = load_dataset_from_config()
        print(f"✓ Loaded dataset with splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check for JSON responses
    print("\n[2] Checking if responses are JSON format...")
    stats = check_json_in_responses(dataset, sample_size=50)
    
    print(f"\nResults:")
    print(f"  Total checked: {stats['total_checked']}")
    print(f"  JSON responses: {stats['json_count']} ({stats['json_count']/stats['total_checked']*100:.1f}%)")
    print(f"  Text responses: {stats['text_count']} ({stats['text_count']/stats['total_checked']*100:.1f}%)")
    
    if stats['json_count'] > 0:
        print(f"\n✓ Found JSON responses! Sample:")
        if stats['json_samples']:
            sample = stats['json_samples'][0]
            print(f"  From {sample['split']}[{sample['idx']}]:")
            print(f"  {sample['preview']}...")
    
    if stats['text_count'] > 0:
        print(f"\n✗ Found text responses. Sample:")
        if stats['text_samples']:
            sample = stats['text_samples'][0]
            print(f"  From {sample['split']}[{sample['idx']}]:")
            print(f"  {sample['preview']}...")
    
    # Convert if needed
    if stats['text_count'] > stats['json_count']:
        print(f"\n[3] Converting text responses to JSON format...")
        print(f"    (Text responses: {stats['text_count']}, JSON responses: {stats['json_count']})")
        
        output_path = Path("data/crisis_response_json.jsonl")
        converted_dataset = convert_dataset_to_json(dataset, output_path)
        
        # Verify conversion
        print(f"\n[4] Verifying converted dataset...")
        converted_stats = check_json_in_responses(converted_dataset, sample_size=20)
        print(f"  JSON responses: {converted_stats['json_count']}/{converted_stats['total_checked']}")
        print(f"  Text responses: {converted_stats['text_count']}/{converted_stats['total_checked']}")
        
        if converted_stats['json_count'] > converted_stats['text_count']:
            print(f"\n✓ Conversion successful!")
            print(f"\nTo use the converted dataset, update configs/dataset_config.yaml:")
            print(f"  hf_dataset_name: \"data/crisis_response_json.jsonl\"")
            print(f"  instruction_column: \"instruction\"")
            print(f"  response_column: \"response\"")
            print(f"  validate_json: true")
        else:
            print(f"\n⚠ Warning: Conversion may not have worked correctly")
    else:
        print(f"\n✓ Dataset already has JSON responses! No conversion needed.")
        print(f"  Make sure your config uses:")
        print(f"    instruction_column: \"Input\" (or check actual column name)")
        print(f"    response_column: \"Output\" (or check actual column name)")
        print(f"    validate_json: true")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
