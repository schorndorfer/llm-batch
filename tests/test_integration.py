import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from llm_batch.cli import make, template
from llm_batch.batch_openai import send, fetch
from llm_batch.batch_anthropic import send as anthropic_send, fetch as anthropic_fetch


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_openai_workflow(
        self, temp_dir, sample_json_files, mock_openai_client
    ):
        """Test complete OpenAI batch workflow from JSON files to results."""
        # Step 1: Create batch file from JSON files
        in_dir = temp_dir
        out_dir = temp_dir / "output"
        batch_name = "integration_test"

        make(in_dir=in_dir, out=out_dir, batch_name=batch_name)

        # Verify batch file was created
        batch_file = out_dir / f"{batch_name}-requests.jsonl"
        assert batch_file.exists()

        # Step 2: Send batch to OpenAI
        mock_file_response = Mock()
        mock_file_response.id = "file_123"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_456"

        mock_openai_client.files.create.return_value = mock_file_response
        mock_openai_client.batches.create.return_value = mock_batch_response

        send(batch_file=batch_file, description="Integration test")

        # Verify batch was sent
        mock_openai_client.files.create.assert_called_once()
        mock_openai_client.batches.create.assert_called_once()

        # Step 3: Fetch results
        mock_batch_retrieve = Mock()
        mock_batch_retrieve.status = "completed"
        mock_batch_retrieve.output_file_id = "output_file_123"

        mock_file_content = Mock()
        mock_file_content.text = '{"result": "integration test response"}'

        mock_openai_client.batches.retrieve.return_value = mock_batch_retrieve
        mock_openai_client.files.content.return_value = mock_file_content

        results_dir = temp_dir / "results"
        fetch(batch_id="batch_456", out=results_dir, batch_name=batch_name)

        # Verify results were fetched
        results_file = results_dir / f"{batch_name}-responses.jsonl"
        assert results_file.exists()
        assert "integration test response" in results_file.read_text()

    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_complete_anthropic_workflow(
        self, mock_anthropic_class, temp_dir, sample_batch_file
    ):
        """Test complete Anthropic batch workflow."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Step 1: Send batch to Anthropic
        mock_batch_response = Mock()
        mock_batch_response.id = "anthropic_batch_123"

        mock_client.messages.batches.create.return_value = mock_batch_response

        anthropic_send(batch_file=sample_batch_file)

        # Verify batch was sent
        mock_anthropic_class.assert_called_once()
        mock_client.messages.batches.create.assert_called_once()

        # Step 2: Fetch results
        mock_result1 = Mock()
        mock_result1.to_json.return_value = (
            '{"custom_id": "id-0", "result": "anthropic response"}'
        )

        mock_client.messages.batches.results.return_value = [mock_result1]

        results_dir = temp_dir / "anthropic_results"
        anthropic_fetch(
            batch_id="msgbatch_123", out=results_dir, batch_name="anthropic_test"
        )

        # Verify results were fetched
        results_file = results_dir / "anthropic_test-responses.jsonl"
        assert results_file.exists()
        assert "anthropic response" in results_file.read_text()

    @patch("llm_batch.cli.console")
    def test_template_to_batch_workflow(self, mock_console, temp_dir):
        """Test workflow from template to batch processing."""
        # Step 1: Process template
        out_dir = temp_dir / "template_output"

        # Create a proper JSON template
        template_content = """
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Hello {{ name }}, this is a test for {{ purpose }}"
    }
  ],
  "max_tokens": 100
}
"""
        template_file = temp_dir / "test_template.json"
        template_file.write_text(template_content)

        # Create data file
        data_content = {"name": ["John", "Jane"], "purpose": ["testing", "development"]}
        data_file = temp_dir / "test_data.yml"
        import yaml

        data_file.write_text(yaml.dump(data_content))

        template(template=template_file, data=data_file, out=out_dir, execute=False)

        # Verify template was processed
        assert out_dir.exists()

        # Step 2: Create batch file from template output
        # This would typically involve creating JSON files from the template output
        # For this test, we'll simulate the process

        # Create sample JSON files from template output
        template_output_dir = temp_dir / "template_json"
        template_output_dir.mkdir()

        # Simulate creating JSON files from template
        request1 = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello John Doe"}],
            "max_tokens": 100,
        }
        file1 = template_output_dir / "request1.json"
        file1.write_text(json.dumps({"request": request1}))

        # Create batch file
        batch_out_dir = temp_dir / "batch_output"
        make(in_dir=template_output_dir, out=batch_out_dir, batch_name="template_batch")

        # Verify batch file was created
        batch_file = batch_out_dir / "template_batch-requests.jsonl"
        assert batch_file.exists()

        # Verify batch content
        content = batch_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1

        request = json.loads(lines[0])
        assert request["method"] == "POST"
        assert request["url"] == "/v1/chat/completions"
        assert "Hello John Doe" in request["body"]["messages"][0]["content"]

    def test_error_handling_integration(self, temp_dir):
        """Test error handling in integration scenarios."""
        # Test with invalid input files
        invalid_dir = temp_dir / "invalid"
        invalid_dir.mkdir()

        # Create invalid JSON file
        invalid_file = invalid_dir / "invalid.json"
        invalid_file.write_text("{ invalid json")

        out_dir = temp_dir / "output"

        # Should handle invalid JSON gracefully
        make(in_dir=invalid_dir, out=out_dir, batch_name="error_test")

        # Output file should be created but empty
        output_file = out_dir / "error_test-requests.jsonl"
        assert output_file.exists()
        assert output_file.read_text().strip() == ""

    def test_large_batch_processing(self, temp_dir):
        """Test processing of large batches."""
        # Create multiple JSON files
        json_dir = temp_dir / "large_batch"
        json_dir.mkdir()

        # Create 10 sample JSON files
        for i in range(10):
            request = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"Request {i}"}],
                "max_tokens": 100,
            }
            file_path = json_dir / f"request_{i}.json"
            file_path.write_text(json.dumps({"request": request}))

        # Create batch file
        out_dir = temp_dir / "large_output"
        make(in_dir=json_dir, out=out_dir, batch_name="large_batch")

        # Verify batch file was created with all requests
        batch_file = out_dir / "large_batch-requests.jsonl"
        assert batch_file.exists()

        content = batch_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 10

        # Verify each request has correct structure
        for line in lines:
            request = json.loads(line)
            assert "custom_id" in request
            assert request["method"] == "POST"
            assert request["url"] == "/v1/chat/completions"
            assert "body" in request

    def test_configuration_integration(self, sample_config):
        """Test that configuration is properly integrated throughout the system."""
        # Verify configuration is accessible
        assert sample_config is not None
        assert "logging" in sample_config

        # Test that logging configuration is properly structured
        logging_config = sample_config["logging"]
        assert "version" in logging_config
        assert "formatters" in logging_config
        assert "handlers" in logging_config
        assert "loggers" in logging_config
        assert "root" in logging_config

        # Verify that all required loggers are configured
        loggers = logging_config["loggers"]
        assert "openaihelper" in loggers

        # Verify handlers are properly configured
        handlers = logging_config["handlers"]
        assert "console" in handlers
        assert "file" in handlers
