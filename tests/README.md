# Test Suite

This directory contains the test suite for the crisis-agent fine-tuning pipeline.

## Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_json_validator.py
│   ├── test_error_handling.py
│   ├── test_logging.py
│   ├── test_data_loading.py
│   └── test_data_formatting.py
├── integration/             # Integration tests
│   └── test_pipeline.py
└── fixtures/                # Test fixtures and data
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/unit/test_json_validator.py
```

### Run with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run specific test class
```bash
pytest tests/unit/test_json_validator.py::TestValidateJsonStructure
```

### Run specific test function
```bash
pytest tests/unit/test_json_validator.py::TestValidateJsonStructure::test_valid_json
```

### Run by marker
```bash
pytest -m unit              # Run only unit tests
pytest -m integration       # Run only integration tests
pytest -m "not slow"        # Skip slow tests
```

## Test Categories

### Unit Tests
- **test_json_validator.py**: JSON validation utilities
- **test_error_handling.py**: Error handling and exceptions
- **test_logging.py**: Logging configuration
- **test_data_loading.py**: Dataset loading functions
- **test_data_formatting.py**: Data formatting functions

### Integration Tests
- **test_pipeline.py**: End-to-end pipeline tests with mocks

## Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_config_dir`: Sample configuration files
- `mock_dataset`: Mock Hugging Face dataset
- `mock_model`: Mock model object
- `mock_tokenizer`: Mock tokenizer object
- `sample_json_response`: Sample JSON response
- `sample_text_with_json`: Sample text containing JSON

## Writing New Tests

1. **Create test file**: Add to appropriate directory (`unit/` or `integration/`)
2. **Import fixtures**: Use fixtures from `conftest.py`
3. **Use mocks**: Mock external dependencies (models, datasets, APIs)
4. **Follow naming**: Use `test_` prefix for functions, `Test` for classes
5. **Add markers**: Use `@pytest.mark.unit` or `@pytest.mark.integration`

### Example Test

```python
import pytest
from src.utils.json_validator import validate_json_structure

def test_valid_json():
    """Test validation of valid JSON."""
    text = '{"action": "test"}'
    is_valid, parsed, error = validate_json_structure(text)
    
    assert is_valid is True
    assert parsed == {"action": "test"}
    assert error is None
```

## Continuous Integration

Tests are designed to run in CI environments:

- No GPU required (all model tests use mocks)
- No external API calls (all network calls are mocked)
- Fast execution (unit tests run in seconds)
- Deterministic (no random data generation)

## Coverage Goals

- **Unit tests**: >80% coverage for utilities
- **Integration tests**: Cover main pipeline flows
- **Error handling**: Test all error paths
- **Edge cases**: Test boundary conditions

## Notes

- Model loading tests use mocks to avoid downloading large models
- Dataset tests use small mock datasets
- All file I/O uses temporary directories
- Tests are isolated and can run in parallel
