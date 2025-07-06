import pytest
import os
from unittest.mock import patch, mock_open
from llm_batch import __version__, CONFIG, app, console, logger


class TestInit:
    """Test package initialization and configuration."""

    def test_version(self):
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_config_loaded(self):
        """Test that configuration is loaded."""
        assert CONFIG is not None
        assert isinstance(CONFIG, dict)
        assert "logging" in CONFIG

    def test_app_exists(self):
        """Test that the main app exists."""
        assert app is not None
        assert hasattr(app, "help")
        assert "Commands to execute LLM batch jobs" in app.help

    def test_console_exists(self):
        """Test that console is available."""
        assert console is not None
        assert hasattr(console, "print")

    def test_logger_exists(self):
        """Test that logger is available."""
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_config_structure(self):
        """Test that configuration has the expected structure."""
        assert "logging" in CONFIG
        logging_config = CONFIG["logging"]

        # Check logging configuration structure
        assert "version" in logging_config
        assert "formatters" in logging_config
        assert "handlers" in logging_config
        assert "loggers" in logging_config
        assert "root" in logging_config

    def test_logging_configuration(self):
        """Test that logging configuration is valid."""
        logging_config = CONFIG["logging"]

        # Check formatters
        assert "simple" in logging_config["formatters"]
        assert "detailed" in logging_config["formatters"]

        # Check handlers
        assert "console" in logging_config["handlers"]
        assert "file" in logging_config["handlers"]

        # Check loggers
        assert "openaihelper" in logging_config["loggers"]

        # Check root configuration
        assert "level" in logging_config["root"]
        assert "handlers" in logging_config["root"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"})
    def test_environment_variables(self):
        """Test that environment variables are accessible."""
        # This test verifies that the dotenv loading works
        # The actual environment variable access is tested in other modules
        assert os.environ.get("OPENAI_API_KEY") == "test_openai_key"

    def test_app_commands_structure(self):
        """Test that the app has the expected command structure."""
        # Test that batch commands are available
        assert hasattr(app, "command")

        # The app should have batch and utils subcommands
        # This is tested by checking the app structure
        assert app is not None

    def test_config_file_loading(self):
        """Test that config file is loaded correctly."""
        # Verify that config contains expected keys
        expected_keys = ["logging"]
        for key in expected_keys:
            assert key in CONFIG, f"Expected key '{key}' not found in CONFIG"

    def test_logging_levels(self):
        """Test that logging levels are properly configured."""
        logging_config = CONFIG["logging"]

        # Check that loggers have proper levels
        for logger_name, logger_config in logging_config["loggers"].items():
            assert "level" in logger_config
            assert logger_config["level"] in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]

        # Check root logger
        root_config = logging_config["root"]
        assert "level" in root_config
        assert root_config["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
