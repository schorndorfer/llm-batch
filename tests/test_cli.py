import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from llm_batch.cli import make, config, pdf2text, template, extract_combinations


class TestCLI:
    """Test CLI functions."""

    def test_extract_combinations(self):
        """Test the extract_combinations function."""
        dict_of_lists = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.1, 0.7],
            "max_tokens": [100, 200],
        }

        combinations = extract_combinations(dict_of_lists)

        assert len(combinations) == 8  # 2 * 2 * 2
        assert all(isinstance(combo, dict) for combo in combinations)
        assert all(len(combo) == 3 for combo in combinations)

        # Check that all combinations are unique
        combo_strings = [json.dumps(combo, sort_keys=True) for combo in combinations]
        assert len(set(combo_strings)) == 8

    def test_make_batch_file_success(self, temp_dir, sample_json_files):
        """Test successful batch file creation."""
        in_dir = temp_dir
        out_dir = temp_dir / "output"
        batch_name = "test_batch"

        # Create the batch file
        make(in_dir=in_dir, out=out_dir, batch_name=batch_name)

        # Check that output file was created
        output_file = out_dir / f"{batch_name}-requests.jsonl"
        assert output_file.exists()

        # Check content
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

        # Parse and verify structure
        for line in lines:
            request = json.loads(line)
            assert "custom_id" in request
            assert "method" in request
            assert "url" in request
            assert "body" in request
            assert request["method"] == "POST"
            assert request["url"] == "/v1/chat/completions"

    def test_make_batch_file_no_json_files(self, temp_dir):
        """Test batch file creation when no JSON files exist."""
        in_dir = temp_dir
        out_dir = temp_dir / "output"

        # Should not raise an exception, just return early
        make(in_dir=in_dir, out=out_dir, batch_name="test")

        # Output directory should not be created
        assert not out_dir.exists()

    def test_make_batch_file_invalid_json(self, temp_dir):
        """Test batch file creation with invalid JSON files."""
        # Create invalid JSON file
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json content")

        out_dir = temp_dir / "output"

        # Should handle the error gracefully
        make(in_dir=temp_dir, out=out_dir, batch_name="test")

        # Output file should still be created but without the invalid file
        output_file = out_dir / "test-requests.jsonl"
        assert output_file.exists()
        assert output_file.read_text().strip() == ""

    @patch("llm_batch.cli.console")
    def test_config_command(self, mock_console, sample_config):
        """Test the config command."""
        config()

        # Verify console.print was called
        assert mock_console.print.call_count >= 1

    @patch("fitz.open")
    @patch("llm_batch.cli.console")
    def test_pdf2text_success(self, mock_console, mock_fitz_open, temp_dir):
        """Test successful PDF text extraction."""
        # Create mock PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text("Mock PDF content")

        # Mock fitz document using MagicMock
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_page.number = 0
        mock_page.get_text.return_value = "Extracted text content"
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        out_dir = temp_dir / "output"

        pdf2text(in_dir=temp_dir, out=out_dir)

        # Check output file was created
        output_file = out_dir / "test.txt"
        assert output_file.exists()
        assert "Extracted text content" in output_file.read_text()

    @patch("fitz.open")
    @patch("llm_batch.cli.console")
    def test_pdf2text_with_page_range(self, mock_console, mock_fitz_open, temp_dir):
        """Test PDF text extraction with page range."""
        # Create mock PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text("Mock PDF content")

        # Mock fitz document with multiple pages using MagicMock
        mock_doc = MagicMock()
        mock_page1 = Mock()
        mock_page1.number = 0
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.number = 1
        mock_page2.get_text.return_value = "Page 2 content"
        mock_page3 = Mock()
        mock_page3.number = 2
        mock_page3.get_text.return_value = "Page 3 content"

        mock_doc.__iter__.return_value = [mock_page1, mock_page2, mock_page3]
        mock_fitz_open.return_value = mock_doc

        out_dir = temp_dir / "output"

        # Extract pages 1-2 (index 0-1)
        pdf2text(in_dir=temp_dir, out=out_dir, start=1, end=2)

        # Check output file
        output_file = out_dir / "test.txt"
        assert output_file.exists()
        content = output_file.read_text()
        assert "Page 2 content" in content
        assert "Page 3 content" in content
        assert "Page 1 content" not in content

    @patch("fitz.open")
    @patch("llm_batch.cli.console")
    def test_pdf2text_exception_handling(self, mock_console, mock_fitz_open, temp_dir):
        """Test PDF text extraction with exception handling."""
        # Create mock PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text("Mock PDF content")

        # Mock fitz to raise an exception
        mock_fitz_open.side_effect = Exception("PDF processing error")

        out_dir = temp_dir / "output"

        # Should handle the exception gracefully
        pdf2text(in_dir=temp_dir, out=out_dir)

        # Output directory should be created even with exception
        assert out_dir.exists()

    @patch("llm_batch.cli.console")
    def test_template_command_success(self, mock_console, temp_dir):
        """Test successful template processing."""
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

        out_dir = temp_dir / "output"

        template(template=template_file, data=data_file, out=out_dir, execute=False)

        # Check that output directory was created
        assert out_dir.exists()

        # Check that template was processed (dry run mode)
        assert mock_console.print.call_count >= 1

    def test_template_command_invalid_template(self, temp_dir, sample_data_file):
        """Test template command with invalid template file."""
        invalid_template = temp_dir / "nonexistent.txt"
        out_dir = temp_dir / "output"

        with pytest.raises(AssertionError):
            template(
                template=invalid_template,
                data=sample_data_file,
                out=out_dir,
                execute=False,
            )

    def test_template_command_invalid_data(self, temp_dir, sample_template_file):
        """Test template command with invalid data file."""
        invalid_data = temp_dir / "nonexistent.yml"
        out_dir = temp_dir / "output"

        with pytest.raises(AssertionError):
            template(
                template=sample_template_file,
                data=invalid_data,
                out=out_dir,
                execute=False,
            )

    def test_template_command_file_as_output(
        self, temp_dir, sample_template_file, sample_data_file
    ):
        """Test template command when output is a file instead of directory."""
        output_file = temp_dir / "output.txt"
        output_file.write_text("existing content")

        with pytest.raises(ValueError):
            template(
                template=sample_template_file,
                data=sample_data_file,
                out=output_file,
                execute=False,
            )
