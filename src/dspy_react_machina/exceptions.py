#!/usr/bin/env python3
"""
Custom exceptions for dspy-react-machina.

This module defines custom exception classes for error handling in the ReActMachina module.
These exceptions provide better error context and allow for more precise error handling.
"""


class ToolNotFoundError(KeyError):
    """Raised when a requested tool is not found in the tool registry.

    This exception is raised when attempting to execute a tool that hasn't been
    registered with the ReActMachina module.

    Inherits from KeyError for compatibility with existing error handling patterns.
    """

    def __init__(self, tool_name: str, available_tools: list[str] | None = None) -> None:
        """Initialize ToolNotFoundError.

        Args:
            tool_name: Name of the tool that was not found
            available_tools: Optional list of available tool names for context
        """
        self.tool_name = tool_name
        self.available_tools = available_tools

        message = f"Tool '{tool_name}' not found in registry"
        if available_tools:
            message += f". Available tools: {', '.join(available_tools)}"

        super().__init__(message)


class ToolExecutionError(RuntimeError):
    """Raised when a tool execution fails.

    This exception wraps errors that occur during tool execution, providing
    additional context about which tool failed and with what arguments.

    Inherits from RuntimeError as tool execution failures are runtime issues.
    """

    def __init__(self, tool_name: str, tool_args: dict, original_error: Exception) -> None:
        """Initialize ToolExecutionError.

        Args:
            tool_name: Name of the tool that failed
            tool_args: Arguments that were passed to the tool
            original_error: The original exception that was raised
        """
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.original_error = original_error

        message = f"Error executing tool '{tool_name}' with args {tool_args}: {original_error!s}"
        super().__init__(message)
