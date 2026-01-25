# Testing Documentation

This guide provides comprehensive documentation for the crisis-agent fine-tuning pipeline test suite.

## Overview

The test suite is designed to ensure reliability and correctness of all pipeline components without requiring GPU access or downloading large models. All tests use mocks and fixtures to provide fast, deterministic test execution.

## Test Architecture

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests for individual components
│   ├── test_json_validator.py
│   ├── test_error_handling.py
│   ├── test_logging.py
│   ├── test_data_loading.py
│   └── test_data_formatting.py
└── integration/             # Integration tests for full workflows
    └── test_pipeline.py
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test complete workflows with mocked dependencies

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_json_validator.py

# Run specific test class
pytest tests/unit/test_json_validator.py::TestValidateJsonStructure

# Run specific test function
pytest tests/unit/test_json_validator.py::TestValidateJsonStructure::test_valid_json
```

### Using Makefile

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration
```

### Advanced Options

```bash
# Run with coverage and HTML report
pytest --cov=src --cov-report=html --cov-report=term

# Run tests matching a pattern
pytest -k "json"

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run only tests marked with specific marker
pytest -m unit
pytest -m integration
```

## Test Coverage

### Current Coverage

The test suite covers:

- ✅ **JSON Validation** (100% coverage)
  - Valid/invalid JSON parsing
  - JSON extraction from text
  - Crisis response structure validation
  - Required keys validation

- ✅ **Error Handling** (100% coverage)
  - Custom exception classes
  - Error decorators
  - Path validation
  - CUDA availability checks

- ✅ **Logging** (95% coverage)
  - Logger setup and configuration
  - Log directory creation
  - Log level configuration
  - Console logging toggle

- ✅ **Data Loading** (90% coverage)
  - Dataset loading from config
  - Auto-splitting single datasets
  - Sample limiting
  - Error handling

- ✅ **Data Formatting** (85% coverage)
  - Prompt template formatting
  - Dataset formatting
  - Tokenization (mocked)

- ✅ **Integration Tests** (80% coverage)
  - Full pipeline with mocks
  - Trainer creation
  - Model loading (mocked)

### Coverage Goals

- **Target**: >80% overall coverage
- **Critical paths**: 100% coverage
- **Utilities**: >90% coverage
- **Integration flows**: >75% coverage

## Test Fixtures

### Available Fixtures

All fixtures are defined in `tests/conftest.py`:

#### `temp_dir`
Creates a temporary directory for test files. Automatically cleaned up after tests.

```python
def test_something(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")
    # Test code here
```

#### `sample_config_dir`
Creates a temporary directory with sample configuration files (dataset, model, training).

```python
def test_with_config(sample_config_dir):
    config_path = sample_config_dir / "dataset_config.yaml"
    # Use config_path in tests
```

#### `mock_dataset`
Provides a mock Hugging Face DatasetDict with train/validation splits.

```python
def test_dataset_processing(mock_dataset):
    assert "train" in mock_dataset
    assert "validation" in mock_dataset
```

#### `mock_model`
Provides a mock model object with common attributes and methods.

```python
def test_model_operations(mock_model):
    assert mock_model.device is not None
    assert hasattr(mock_model, 'generate')
```

#### `mock_tokenizer`
Provides a mock tokenizer object with encoding/decoding methods.

```python
def test_tokenization(mock_tokenizer):
    encoded = mock_tokenizer.encode("test")
    assert encoded is not None
```

#### `sample_json_response`
Provides a sample JSON response for testing.

```python
def test_json_validation(sample_json_response):
    assert "action" in sample_json_response
```

#### `sample_text_with_json`
Provides sample text containing JSON for extraction tests.

```python
def test_json_extraction(sample_text_with_json):
    # Test JSON extraction from text
```

## Writing Tests

### Test Naming Conventions

- **Test files**: `test_*.py`
- **Test classes**: `Test*` (e.g., `TestValidateJsonStructure`)
- **Test functions**: `test_*` (e.g., `test_valid_json`)

### Example Test Structure

```python
import pytest
from src.utils.json_validator import validate_json_structure

class TestValidateJsonStructure:
    """Tests for validate_json_structure function."""
    
    def test_valid_json(self):
        """Test validation of valid JSON."""
        text = '{"action": "test"}'
        is_valid, parsed, error = validate_json_structure(text)
        
        assert is_valid is True
        assert parsed == {"action": "test"}
        assert error is None
    
    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        text = '{"invalid": json}'
        is_valid, parsed, error = validate_json_structure(text, strict=False)
        
        assert is_valid is False
        assert error is not None
```

### Best Practices

1. **Use descriptive test names**: Test names should clearly describe what they test
2. **One assertion per concept**: Group related assertions, but test one concept per test
3. **Use fixtures**: Leverage shared fixtures instead of duplicating setup code
4. **Mock external dependencies**: Never make real API calls or download models in tests
5. **Test edge cases**: Include tests for boundary conditions and error cases
6. **Keep tests isolated**: Tests should not depend on each other
7. **Use markers**: Mark slow or integration tests appropriately

### Using Mocks

```python
from unittest.mock import Mock, patch

@patch('src.data.load_dataset.load_dataset')
def test_dataset_loading(mock_load_dataset):
    """Test dataset loading with mocked Hugging Face API."""
    # Setup mock
    mock_dataset = DatasetDict({"train": ...})
    mock_load_dataset.return_value = mock_dataset
    
    # Test code
    result = load_dataset_from_config()
    
    # Assertions
    assert result is not None
    mock_load_dataset.assert_called_once()
```

## Continuous Integration

### CI Configuration

Tests are designed to run in CI environments:

- **No GPU required**: All model tests use mocks
- **No external API calls**: All network calls are mocked
- **Fast execution**: Unit tests complete in seconds
- **Deterministic**: No random data generation

### Example CI Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with print statements visible
pytest -s

# Run with detailed traceback
pytest --tb=long

# Run with pdb debugger on failure
pytest --pdb

# Run specific test with debugging
pytest tests/unit/test_json_validator.py::test_valid_json --pdb
```

### Common Issues

#### Test Failures

1. **Import errors**: Ensure project root is in Python path
2. **Fixture not found**: Check fixture is defined in `conftest.py`
3. **Mock not working**: Verify patch path matches import path
4. **Path issues**: Use `temp_dir` fixture for file operations

#### Debugging Tips

```python
# Add print statements (use -s flag)
print(f"Debug: {variable}")

# Use pytest's built-in debugging
import pytest
pytest.set_trace()  # Breakpoint

# Check fixture values
def test_something(temp_dir):
    print(f"Temp dir: {temp_dir}")  # See what fixture provides
```

## Test Maintenance

### Adding New Tests

1. **Identify test location**: Unit test or integration test?
2. **Create test file**: Follow naming convention `test_*.py`
3. **Use existing fixtures**: Check `conftest.py` for available fixtures
4. **Write test cases**: Cover happy path, edge cases, and errors
5. **Run tests**: Verify tests pass locally
6. **Update coverage**: Check coverage report for gaps

### Updating Existing Tests

- **When code changes**: Update corresponding tests
- **When behavior changes**: Update test expectations
- **When bugs are found**: Add regression tests
- **When refactoring**: Ensure tests still pass

### Test Organization

- **Group related tests**: Use test classes for related functionality
- **Keep tests focused**: One test should verify one behavior
- **Use descriptive names**: Test names should be self-documenting
- **Add docstrings**: Explain what each test verifies

## Performance

### Test Execution Times

- **Unit tests**: < 1 second total
- **Integration tests**: < 5 seconds total
- **Full suite**: < 10 seconds total

### Optimization Tips

1. **Use fixtures efficiently**: Share expensive setup via fixtures
2. **Mock heavy operations**: Don't load real models or datasets
3. **Run tests in parallel**: Use `pytest-xdist` for parallel execution
4. **Skip slow tests**: Mark slow tests and skip in CI if needed

## Coverage Reports

### Generating Reports

```bash
# Terminal report
pytest --cov=src --cov-report=term

# HTML report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

### Interpreting Coverage

- **Line coverage**: Percentage of code lines executed
- **Branch coverage**: Percentage of code branches executed
- **Function coverage**: Percentage of functions called

### Coverage Goals by Module

| Module | Target Coverage |
|--------|----------------|
| `src/utils/` | >90% |
| `src/data/` | >85% |
| `src/model/` | >80% (mocked) |
| `src/training/` | >75% (mocked) |

## Troubleshooting

### Common Problems

#### Tests Pass Locally but Fail in CI

- Check Python version compatibility
- Verify all dependencies are installed
- Check for platform-specific code
- Ensure environment variables are set

#### Import Errors

```python
# Ensure project root is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### Mock Not Working

```python
# Patch the import path, not the definition
@patch('module.where.used.FunctionName')  # Correct
# Not: @patch('module.where.defined.FunctionName')
```

#### Fixture Scope Issues

```python
# Use appropriate fixture scope
@pytest.fixture(scope="module")  # Shared across module
@pytest.fixture(scope="session")  # Shared across session
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

## Next Steps

1. **Run the test suite**: `pytest` or `make test`
2. **Check coverage**: `pytest --cov=src --cov-report=html`
3. **Add tests for new features**: Follow the patterns in existing tests
4. **Maintain test quality**: Keep tests fast, isolated, and reliable

---

For more information, see [tests/README.md](../tests/README.md) for test structure details.
