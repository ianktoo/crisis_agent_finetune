"""
JSON validation utilities for crisis-agent responses.
Validates JSON structure and content quality.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_json_structure(
    text: str,
    required_keys: Optional[List[str]] = None,
    strict: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate JSON structure in text.
    
    Args:
        text: Text that should contain JSON
        required_keys: List of required keys in JSON (if any)
        strict: If True, text must be valid JSON; if False, tries to extract JSON
        
    Returns:
        Tuple of (is_valid, parsed_json, error_message)
    """
    try:
        # Try direct parsing
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return False, None, "JSON is not a dictionary/object"
        
        # Check required keys
        if required_keys:
            missing_keys = [key for key in required_keys if key not in parsed]
            if missing_keys:
                return False, parsed, f"Missing required keys: {missing_keys}"
        
        return True, parsed, None
        
    except json.JSONDecodeError as e:
        if strict:
            return False, None, f"Invalid JSON: {str(e)}"
        
        # Try to extract JSON from text
        try:
            # Look for JSON-like structures
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                
                if not isinstance(parsed, dict):
                    return False, None, "Extracted JSON is not a dictionary"
                
                if required_keys:
                    missing_keys = [key for key in required_keys if key not in parsed]
                    if missing_keys:
                        return False, parsed, f"Missing required keys: {missing_keys}"
                
                return True, parsed, None
        except Exception:
            pass
        
        return False, None, f"Could not parse JSON: {str(e)}"


def validate_crisis_response(
    response: Dict[str, Any],
    expected_structure: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a crisis-agent response for structure and quality.
    
    Args:
        response: Parsed JSON response
        expected_structure: Expected structure (for validation)
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    if not isinstance(response, dict):
        return False, ["Response is not a dictionary"]
    
    # Check for common crisis-response fields
    common_fields = ["action", "priority", "reasoning", "resources"]
    
    for field in common_fields:
        if field not in response:
            warnings.append(f"Missing common field: {field}")
    
    # Validate expected structure if provided
    if expected_structure:
        for key, expected_type in expected_structure.items():
            if key not in response:
                warnings.append(f"Missing expected field: {key}")
            elif not isinstance(response[key], expected_type):
                warnings.append(
                    f"Field '{key}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(response[key]).__name__}"
                )
    
    is_valid = len(warnings) == 0 or all("Missing common field" in w for w in warnings)
    
    return is_valid, warnings


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain other content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dictionary, or None if not found
    """
    is_valid, parsed, _ = validate_json_structure(text, strict=False)
    
    if is_valid and parsed:
        return parsed
    
    return None
