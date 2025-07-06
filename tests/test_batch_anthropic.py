import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock
from llm_batch.batch_anthropic import send, fetch, check


class TestAnthropicBatch:
    """Test Anthropic batch processing functions."""

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.logger")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_send_batch_success(
        self, mock_anthropic_class, mock_logger, mock_console, sample_batch_file
    ):
        """Test successful batch upload."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock the batch response
        mock_batch_response = Mock()
        mock_batch_response.id = "batch_456"

        mock_client.messages.batches.create.return_value = mock_batch_response

        # Test the send function
        send(batch_file=sample_batch_file)

        # Verify Anthropic client was called correctly
        mock_anthropic_class.assert_called_once()
        mock_client.messages.batches.create.assert_called_once()

        # Verify the requests were created correctly
        call_args = mock_client.messages.batches.create.call_args
        requests = call_args[1]["requests"]

        assert len(requests) == 2

        # Check first request - verify it's a Request object with correct structure
        first_request = requests[0]
        assert "custom_id" in first_request
        assert "params" in first_request
        assert first_request["custom_id"] == "id-0"
        params = first_request["params"]
        assert params["model"] == "gpt-3.5-turbo"
        assert params["max_tokens"] == 100
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Hello"

        # Check second request
        second_request = requests[1]
        assert "custom_id" in second_request
        assert "params" in second_request
        assert second_request["custom_id"] == "id-1"
        params = second_request["params"]
        assert params["model"] == "gpt-4"
        assert params["max_tokens"] == 200
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Test message"

        # Verify console was called
        assert mock_console.print.call_count >= 1

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.logger")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_fetch_batch_success(
        self, mock_anthropic_class, mock_logger, mock_console, temp_dir
    ):
        """Test successful batch results fetching."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock the results response
        mock_result1 = Mock()
        mock_result1.to_json.return_value = (
            '{"custom_id": "id-0", "result": "response1"}'
        )

        mock_result2 = Mock()
        mock_result2.to_json.return_value = (
            '{"custom_id": "id-1", "result": "response2"}'
        )

        mock_client.messages.batches.results.return_value = [mock_result1, mock_result2]

        out_dir = temp_dir / "output"
        batch_id = "msgbatch_123"  # Use valid Anthropic batch ID format
        batch_name = "test_batch"

        # Test the fetch function
        fetch(batch_id=batch_id, out=out_dir, batch_name=batch_name)

        # Verify output file was created
        output_file = out_dir / f"{batch_name}-responses.jsonl"
        assert output_file.exists()

        # Check content
        content = output_file.read_text()
        assert "response1" in content
        assert "response2" in content

        # Verify Anthropic client was called correctly
        mock_anthropic_class.assert_called_once()
        mock_client.messages.batches.results.assert_called_once_with(
            message_batch_id=batch_id
        )

        # Verify console and logger were called
        assert mock_console.print.call_count >= 1
        assert mock_logger.info.call_count >= 1

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_check_batches(self, mock_anthropic_class, mock_console):
        """Test checking batch list."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock the batches response
        mock_batch1 = Mock()
        mock_batch1.id = "batch_1"
        mock_batch1.processing_status = "completed"
        mock_batch1.created_at = "2022-01-01T00:00:00Z"

        mock_batch2 = Mock()
        mock_batch2.id = "batch_2"
        mock_batch2.processing_status = "in_progress"
        mock_batch2.created_at = "2022-01-01T00:01:00Z"

        mock_client.messages.batches.list.return_value = [
            mock_batch2,
            mock_batch1,
        ]  # Unsorted

        # Test the check function
        check(limit=10)

        # Verify Anthropic client was called
        mock_anthropic_class.assert_called_once()
        mock_client.messages.batches.list.assert_called_once_with(limit=10)

        # Verify console was called for each batch
        assert mock_console.print.call_count == 2

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_check_batches_with_limit(self, mock_anthropic_class, mock_console):
        """Test checking batches with custom limit."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_client.messages.batches.list.return_value = []

        # Test with custom limit
        check(limit=5)

        # Verify the limit was passed correctly
        mock_anthropic_class.assert_called_once()
        mock_client.messages.batches.list.assert_called_once_with(limit=5)

    def test_send_batch_file_not_found(self):
        """Test sending batch with non-existent file."""
        non_existent_file = Path("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            send(batch_file=non_existent_file)

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.logger")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_send_batch_anthropic_error(
        self, mock_anthropic_class, mock_logger, mock_console, sample_batch_file
    ):
        """Test handling Anthropic API errors."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock Anthropic client to raise an exception
        mock_client.messages.batches.create.side_effect = Exception(
            "Anthropic API error"
        )

        # Should handle the exception gracefully
        with pytest.raises(Exception):
            send(batch_file=sample_batch_file)

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.logger")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_send_batch_invalid_json(
        self, mock_anthropic_class, mock_logger, mock_console, temp_dir
    ):
        """Test sending batch with invalid JSON in batch file."""
        # Create invalid batch file
        invalid_batch_file = temp_dir / "invalid-batch.jsonl"
        invalid_batch_file.write_text("invalid json content\n")

        with pytest.raises(json.JSONDecodeError):
            send(batch_file=invalid_batch_file)

    @patch("llm_batch.batch_anthropic.console")
    @patch("llm_batch.batch_anthropic.logger")
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_fetch_batch_no_results(
        self, mock_anthropic_class, mock_logger, mock_console, temp_dir
    ):
        """Test fetching batch with no results."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock empty results
        mock_client.messages.batches.results.return_value = []

        out_dir = temp_dir / "output"
        batch_id = "msgbatch_123"  # Use valid Anthropic batch ID format
        batch_name = "test_batch"

        # Test the fetch function
        fetch(batch_id=batch_id, out=out_dir, batch_name=batch_name)

        # Verify output file was created but empty
        output_file = out_dir / f"{batch_name}-responses.jsonl"
        assert output_file.exists()
        assert output_file.read_text() == ""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch("llm_batch.batch_anthropic.Anthropic")
    def test_anthropic_client_initialization(
        self, mock_anthropic_class, sample_batch_file
    ):
        """Test that Anthropic client is initialized with API key."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock successful batch creation
        mock_batch_response = Mock()
        mock_batch_response.id = "batch_123"
        mock_client.messages.batches.create.return_value = mock_batch_response

        send(batch_file=sample_batch_file)

        # Verify Anthropic client was initialized with API key
        mock_anthropic_class.assert_called_once_with(api_key="test_key")
