# LLM Batch Tests

This directory contains comprehensive tests for the LLM batch processing project.

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_cli.py` - Tests for CLI commands and utilities
- `test_batch_openai.py` - Tests for OpenAI batch processing
- `test_batch_anthropic.py` - Tests for Anthropic batch processing
- `test_batch_gemini.py` - Tests for Gemini batch processing (minimal)
- `test_init.py` - Tests for package initialization and configuration
- `test_integration.py` - Integration tests for complete workflows

## Running Tests

### Install test dependencies
```bash
uv add --dev pytest pytest-cov pytest-mock
```

### Run all tests
```bash
pytest
```

### Run specific test files
```bash
pytest tests/test_cli.py
pytest tests/test_batch_openai.py
```

### Run with coverage
```bash
pytest --cov=src/llm_batch --cov-report=html
```

### Run specific test categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Unit Tests
- Individual function testing
- Mock external dependencies
- Fast execution

### Integration Tests
- Complete workflow testing
- End-to-end scenarios
- May use real API calls (with proper mocking)

### Fixtures
- `temp_dir` - Temporary directory for test files
- `sample_json_files` - Sample JSON request files
- `sample_batch_file` - Sample batch file
- `sample_template_file` - Sample Jinja2 template
- `sample_data_file` - Sample YAML data file
- `mock_openai_client` - Mocked OpenAI client
- `mock_anthropic_client` - Mocked Anthropic client
- `mock_litellm` - Mocked litellm completion

## Test Coverage

The tests cover:

1. **CLI Functions**
   - Batch file creation
   - Template processing
   - PDF text extraction
   - Configuration display
   - Error handling

2. **OpenAI Integration**
   - Batch upload
   - Batch status checking
   - Results fetching
   - Error handling

3. **Anthropic Integration**
   - Batch creation
   - Results retrieval
   - Status checking
   - API error handling

4. **Configuration**
   - Config file loading
   - Logging setup
   - Environment variables

5. **Integration Workflows**
   - Complete OpenAI workflow
   - Complete Anthropic workflow
   - Template to batch processing
   - Large batch processing
   - Error handling scenarios

## Adding New Tests

When adding new tests:

1. Use appropriate fixtures from `conftest.py`
2. Mock external dependencies
3. Test both success and error scenarios
4. Add appropriate markers for test categorization
5. Include docstrings explaining test purpose

## Mocking Strategy

- External APIs are mocked to avoid real API calls
- File system operations use temporary directories
- Console output is mocked to avoid cluttering test output
- Logging is captured and verified where relevant 