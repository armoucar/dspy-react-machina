"""Tests for tools module."""

from dspy_react_machina.tools import (
    FINISH_TOOL_MESSAGE,
    HANDLE_ERROR_TOOL_MESSAGE,
    finish_tool_func,
    format_tool_parameters,
    handle_error_tool_func,
    SpecialTools,
)


class TestSpecialTools:
    """Test SpecialTools enum."""

    def test_special_tools_values(self):
        """Test that all special tools have correct values."""
        assert SpecialTools.FINISH == "finish"
        assert SpecialTools.ERROR == "error"
        assert SpecialTools.TIMEOUT == "timeout"


class TestFormatToolParameters:
    """Test format_tool_parameters function."""

    def test_empty_args(self):
        """Test formatting with empty arguments."""
        result = format_tool_parameters({})
        assert result == ""

    def test_single_arg(self):
        """Test formatting with single argument."""
        result = format_tool_parameters({"query": "test"})
        assert result == "query='test'"

    def test_multiple_args(self):
        """Test formatting with multiple arguments."""
        result = format_tool_parameters({"city": "Paris", "units": "metric"})
        # Order might vary, so check both formats
        assert result in ["city='Paris', units='metric'", "units='metric', city='Paris'"]

    def test_different_types(self):
        """Test formatting with different value types."""
        result = format_tool_parameters({"count": 42, "enabled": True, "name": "test"})
        assert "count=42" in result
        assert "enabled=True" in result
        assert "name='test'" in result


class TestFinishToolFunc:
    """Test finish_tool_func function."""

    def test_returns_message(self):
        """Test that finish_tool_func returns the correct message."""
        result = finish_tool_func()
        assert result == FINISH_TOOL_MESSAGE
        assert result == "Task complete. Ready to provide final outputs."


class TestHandleErrorToolFunc:
    """Test handle_error_tool_func function."""

    def test_returns_message(self):
        """Test that handle_error_tool_func returns the correct message."""
        result = handle_error_tool_func()
        assert result == HANDLE_ERROR_TOOL_MESSAGE
        assert result == "There was an error processing the last request."
