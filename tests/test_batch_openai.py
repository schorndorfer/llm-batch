import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock
from llm_batch.batch_openai import send, fetch, check


class TestOpenAIBatch:
    """Test OpenAI batch processing functions."""

    @patch("llm_batch.batch_openai.console")
    @patch("llm_batch.batch_openai.logger")
    def test_send_batch_success(
        self, mock_logger, mock_console, sample_batch_file, mock_openai_client
    ):
        """Test successful batch upload."""
        # Mock the OpenAI client responses
        mock_file_response = Mock()
        mock_file_response.id = "file_123"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_456"

        mock_openai_client.files.create.return_value = mock_file_response
        mock_openai_client.batches.create.return_value = mock_batch_response

        # Test the send function
        send(batch_file=sample_batch_file, description="Test batch")

        # Verify OpenAI client was called correctly
        mock_openai_client.files.create.assert_called_once()
        mock_openai_client.batches.create.assert_called_once_with(
            input_file_id="file_123",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Test batch: {sample_batch_file.name}"},
        )

        # Verify console and logger were called
        assert mock_console.print.call_count >= 2
        assert mock_logger.info.call_count >= 2

    @patch("llm_batch.batch_openai.console")
    @patch("llm_batch.batch_openai.logger")
    def test_fetch_batch_completed(
        self, mock_logger, mock_console, mock_openai_client, temp_dir
    ):
        """Test fetching completed batch results."""
        # Mock the batch response
        mock_batch_response = Mock()
        mock_batch_response.status = "completed"
        mock_batch_response.output_file_id = "output_file_123"

        # Mock the file content response
        mock_file_response = Mock()
        mock_file_response.text = '{"result": "test response"}'

        mock_openai_client.batches.retrieve.return_value = mock_batch_response
        mock_openai_client.files.content.return_value = mock_file_response

        out_dir = temp_dir / "output"
        batch_id = "batch_123"
        batch_name = "test_batch"

        # Test the fetch function
        fetch(batch_id=batch_id, out=out_dir, batch_name=batch_name)

        # Verify output file was created
        output_file = out_dir / f"{batch_name}-responses.jsonl"
        assert output_file.exists()
        assert output_file.read_text() == '{"result": "test response"}'

        # Verify OpenAI client was called correctly
        mock_openai_client.batches.retrieve.assert_called_once_with(batch_id)
        mock_openai_client.files.content.assert_called_once_with("output_file_123")

    @patch("llm_batch.batch_openai.console")
    @patch("llm_batch.batch_openai.logger")
    def test_fetch_batch_not_completed(
        self, mock_logger, mock_console, mock_openai_client
    ):
        """Test fetching batch that is not completed."""
        # Mock the batch response
        mock_batch_response = Mock()
        mock_batch_response.status = "in_progress"

        mock_openai_client.batches.retrieve.return_value = mock_batch_response

        batch_id = "batch_123"

        # Test the fetch function
        fetch(batch_id=batch_id)

        # Verify OpenAI client was called
        mock_openai_client.batches.retrieve.assert_called_once_with(batch_id)

        # Verify file content was not called (since batch is not completed)
        mock_openai_client.files.content.assert_not_called()

    @patch("llm_batch.batch_openai.console")
    def test_check_batches(self, mock_console, mock_openai_client):
        """Test checking batch list."""
        # Mock the batches response
        mock_batch1 = Mock()
        mock_batch1.id = "batch_1"
        mock_batch1.status = "completed"
        mock_batch1.created_at = 1640995200  # 2022-01-01 00:00:00

        mock_batch2 = Mock()
        mock_batch2.id = "batch_2"
        mock_batch2.status = "in_progress"
        mock_batch2.created_at = 1640995260  # 2022-01-01 00:01:00

        mock_openai_client.batches.list.return_value = [
            mock_batch2,
            mock_batch1,
        ]  # Unsorted

        # Test the check function
        check(limit=10)

        # Verify OpenAI client was called
        mock_openai_client.batches.list.assert_called_once_with(limit=10)

        # Verify console was called for each batch
        assert mock_console.print.call_count == 2

    @patch("llm_batch.batch_openai.console")
    def test_check_batches_with_limit(self, mock_console, mock_openai_client):
        """Test checking batches with custom limit."""
        mock_openai_client.batches.list.return_value = []

        # Test with custom limit
        check(limit=5)

        # Verify the limit was passed correctly
        mock_openai_client.batches.list.assert_called_once_with(limit=5)

    def test_send_batch_file_not_found(self):
        """Test sending batch with non-existent file."""
        non_existent_file = Path("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            send(batch_file=non_existent_file)

    @patch("llm_batch.batch_openai.console")
    @patch("llm_batch.batch_openai.logger")
    def test_send_batch_openai_error(
        self, mock_logger, mock_console, sample_batch_file, mock_openai_client
    ):
        """Test handling OpenAI API errors."""
        # Mock OpenAI client to raise an exception
        mock_openai_client.files.create.side_effect = Exception("OpenAI API error")

        # Should handle the exception gracefully
        with pytest.raises(Exception):
            send(batch_file=sample_batch_file)
