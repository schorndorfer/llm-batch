import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from llm_batch import CONFIG


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_json_files(temp_dir):
    """Create sample JSON files for testing."""
    files = []

    # Sample request file 1
    request1 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    file1 = temp_dir / "request1.json"
    file1.write_text(json.dumps({"request": request1}))
    files.append(file1)

    # Sample request file 2
    request2 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Test message"}],
        "max_tokens": 200,
    }
    file2 = temp_dir / "request2.json"
    file2.write_text(json.dumps(request2))
    files.append(file2)

    return files


@pytest.fixture
def sample_batch_file(temp_dir):
    """Create a sample batch file for testing."""
    batch_data = [
        {
            "custom_id": "id_request1.json",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        },
        {
            "custom_id": "id_request2.json",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Test message"}],
                "max_tokens": 200,
            },
        },
    ]

    batch_file = temp_dir / "test-batch.jsonl"
    batch_file.write_text("\n".join([json.dumps(item) for item in batch_data]))
    return batch_file


@pytest.fixture
def sample_template_file(temp_dir):
    """Create a sample Jinja2 template file."""
    template_content = """
Hello {{ name }},

This is a test message for {{ purpose }}.

Best regards,
{{ sender }}
"""
    template_file = temp_dir / "test_template.txt"
    template_file.write_text(template_content)
    return template_file


@pytest.fixture
def sample_data_file(temp_dir):
    """Create a sample YAML data file for template testing."""
    data = {"name": "John Doe", "purpose": "testing", "sender": "Test Team"}
    data_file = temp_dir / "test_data.yml"
    data_file.write_text(yaml.dump(data))
    return data_file


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a mock PDF file for testing."""
    # This is a mock since we can't easily create real PDFs in tests
    pdf_file = temp_dir / "test.pdf"
    pdf_file.write_text("Mock PDF content")
    return pdf_file


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    with patch("anthropic.Anthropic") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_litellm():
    """Mock litellm for testing."""
    with patch("litellm.completion") as mock_completion:
        yield mock_completion


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return CONFIG
