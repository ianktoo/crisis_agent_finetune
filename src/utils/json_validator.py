"""
JSON validation utilities for crisis-agent responses.
Validates JSON structure and content quality.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


def clean_json_text(text: str) -> str:
    """
    Clean and fix common JSON formatting errors in model-generated text.
    
    Common errors fixed:
    - Spaces before closing quotes in keys: "facts "] -> "facts"
    - Wrong bracket placement: "facts "]: [ -> "facts": [
    - Extra spaces in keys
    - Missing colons after keys
    - Malformed array items with extra quotes
    
    Args:
        text: Text containing potentially malformed JSON
        
    Returns:
        Cleaned text with common JSON errors fixed
    """
    if not text or not text.strip():
        return text
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove markdown code blocks if present
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    
    # Fix: "key "]: [ -> "key": [ (most common error pattern)
    text = re.sub(r'"(\w+)\s+"\]:\s*\[', r'"\1": [', text)
    
    # Fix: "key "]: " -> "key": "
    text = re.sub(r'"(\w+)\s+"\]:\s*"', r'"\1": "', text)
    
    # Fix: "key "]: number -> "key": number
    text = re.sub(r'"(\w+)\s+"\]:\s*(\d+\.?\d*)', r'"\1": \2', text)
    
    # Fix: "key "]: word -> "key": "word"
    text = re.sub(r'"(\w+)\s+"\]:\s*(\w+)', r'"\1": "\2"', text)
    
    # Fix: "key "]: -> "key":
    text = re.sub(r'"(\w+)\s+"\]:', r'"\1":', text)
    
    # Fix spaces before closing quotes in keys: "facts " -> "facts"
    text = re.sub(r'"(\w+)\s+"\s*:', r'"\1":', text)
    
    # Fix missing colons: "key" [ -> "key": [
    text = re.sub(r'"(\w+)"\s+\[', r'"\1": [', text)
    text = re.sub(r'"(\w+)"\s+"', r'"\1": "', text)
    
    # Fix malformed array items that have "]: [ after them
    # Pattern: "text " "]: [ -> "text",
    # This handles cases like: "Gas leak occurred... " " "]: [
    text = re.sub(r'"([^"]+?)\s+"\s+"\s*"\]:\s*\[', r'"\1",', text)
    text = re.sub(r'"([^"]+?)\s+"\s*"\]:\s*\[', r'"\1",', text)
    
    # Fix: "text "]: [ -> "text", (when it's clearly an array item, not a key)
    # This happens when model generates malformed nested structures
    # Only apply if it's inside an array context (after a comma or opening bracket)
    text = re.sub(r',\s*"([^"]+?)\s+"\]:\s*\[', r', "\1",', text)
    text = re.sub(r'\[\s*"([^"]+?)\s+"\]:\s*\[', r'["\1",', text)
    
    # Fix: "text " " -> "text" (merge adjacent quoted strings in arrays)
    # Handle patterns like: "text " "more text"
    text = re.sub(r'"([^"]+?)\s+"\s+"([^"]+)"', r'"\1 \2"', text)
    
    # Fix trailing spaces in string values (multiple spaces before closing quote)
    text = re.sub(r'":\s*"([^"]*?)\s{2,}"', r'": "\1"', text)
    
    # Fix single trailing space in string values (common error)
    # But only if it's followed by a comma, bracket, or end of line
    text = re.sub(r'":\s*"([^"]+?)\s+"([,\]\}])', r'": "\1"\2', text)
    
    # Remove comments (JSON doesn't support comments)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Ensure JSON starts with { if it doesn't already
    # But only if it looks like it should be an object (starts with a key)
    if not text.strip().startswith('{') and text.strip().startswith('"'):
        # Check if it looks like JSON object content
        if '":' in text or '"]:' in text:
            text = '{' + text
    
    # Ensure JSON ends with } if it doesn't already and looks incomplete
    if not text.strip().endswith('}') and text.strip().endswith(']'):
        # Count braces to see if we need closing brace
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text = text + '}'
    
    return text.strip()


def extract_and_clean_json(text: str) -> Optional[str]:
    """
    Extract JSON from text and clean common formatting errors.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Cleaned JSON string, or None if no JSON found
    """
    if not text:
        return None
    
    # First, try to find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    
    # Extract JSON portion
    json_str = text[start_idx:end_idx + 1]
    
    # Clean the JSON
    json_str = clean_json_text(json_str)
    
    return json_str


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
        
        # Try to clean and extract JSON from text
        try:
            # Clean the text first
            cleaned_text = clean_json_text(text)
            
            # Try parsing cleaned text
            try:
                parsed = json.loads(cleaned_text)
                if not isinstance(parsed, dict):
                    return False, None, "JSON is not a dictionary/object"
                
                if required_keys:
                    missing_keys = [key for key in required_keys if key not in parsed]
                    if missing_keys:
                        return False, parsed, f"Missing required keys: {missing_keys}"
                
                return True, parsed, None
            except json.JSONDecodeError:
                pass
            
            # If direct parsing failed, try to extract JSON object
            json_str = extract_and_clean_json(text)
            if json_str:
                parsed = json.loads(json_str)
                
                if not isinstance(parsed, dict):
                    return False, None, "Extracted JSON is not a dictionary"
                
                if required_keys:
                    missing_keys = [key for key in required_keys if key not in parsed]
                    if missing_keys:
                        return False, parsed, f"Missing required keys: {missing_keys}"
                
                return True, parsed, None
        except Exception as extract_error:
            logger.debug(f"JSON extraction/cleaning failed: {str(extract_error)}")
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
    Uses cleaning utilities to fix common JSON errors.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dictionary, or None if not found
    """
    if not text:
        return None
    
    # Try to extract and clean JSON
    json_str = extract_and_clean_json(text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Fallback to validation method (which also does cleaning)
    is_valid, parsed, _ = validate_json_structure(text, strict=False)
    
    if is_valid and parsed:
        return parsed
    
    return None


def validate_structured_text_response(
    text: str,
    expected_sections: Optional[List[str]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate structured text response (e.g., FACTS, UNCERTAINTIES format).
    
    Args:
        text: Text response to validate
        expected_sections: List of expected section headers (e.g., ["FACTS", "UNCERTAINTIES"])
        
    Returns:
        Tuple of (is_valid, warnings, parsed_structure)
    """
    warnings = []
    parsed_structure = {}
    
    if not text or not text.strip():
        return False, ["Empty response"], {}
    
    # Common section headers in crisis responses
    if expected_sections is None:
        expected_sections = ["FACTS", "UNCERTAINTIES", "ANALYSIS", "ACTIONS", "RESOURCES"]
    
    text_upper = text.upper()
    found_sections = []
    
    # Check for section headers
    for section in expected_sections:
        if section in text_upper or f"{section}:" in text_upper:
            found_sections.append(section)
            # Try to extract section content
            section_patterns = [
                f"{section}:",
                f"{section}\n",
                f"{section} -",
            ]
            for pattern in section_patterns:
                if pattern in text:
                    idx = text.find(pattern)
                    if idx != -1:
                        # Extract content after section header
                        content_start = idx + len(pattern)
                        # Find next section or end
                        next_section_idx = len(text)
                        for other_section in expected_sections:
                            if other_section != section:
                                next_idx = text.find(f"{other_section}:", content_start)
                                if next_idx != -1 and next_idx < next_section_idx:
                                    next_section_idx = next_idx
                        parsed_structure[section] = text[content_start:next_section_idx].strip()
                        break
    
    # Validation checks
    if len(found_sections) == 0:
        warnings.append("No structured sections found (FACTS, UNCERTAINTIES, etc.)")
    
    # Check for minimum required sections
    if "FACTS" not in found_sections:
        warnings.append("Missing FACTS section")
    
    # Check if response has meaningful content
    if len(text.strip()) < 50:
        warnings.append("Response too short (less than 50 characters)")
    
    # Check for bullet points or structured content
    has_bullets = "•" in text or "-" in text or "*" in text
    if not has_bullets and len(found_sections) == 0:
        warnings.append("No structured formatting detected")
    
    is_valid = len(warnings) == 0 or (len(found_sections) > 0 and "FACTS" in found_sections)
    
    return is_valid, warnings, {"sections_found": found_sections, "content": parsed_structure}
