#!/usr/bin/env python3
"""
Tool Management for ReActMachina

This module provides tool execution and management utilities for the ReActMachina module.
It includes special tool definitions, tool parameter formatting, and the finish tool callable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

# UI String Constants
FINISH_TOOL_MESSAGE = "Task complete. Ready to provide final outputs."
HANDLE_ERROR_TOOL_MESSAGE = "There was an error processing the last request."


class SpecialTools(StrEnum):
    """Constants for special tool names.

    These are internal control signals used by ReActMachina to manage the agent's behavior.
    """

    FINISH = "finish"
    ERROR = "error"
    TIMEOUT = "timeout"  # TODO: Implement timeout handling for long-running tool executions


# Internal-only tools that should not be exposed to the LLM
# These tools are used for internal flow control and error handling only
INTERNAL_ONLY_TOOLS: frozenset[str] = frozenset([SpecialTools.ERROR])


def format_tool_parameters(tool_args: dict[str, Any]) -> str:
    """Format tool arguments as a parameter string.

    Args:
        tool_args: Dictionary of tool arguments

    Returns:
        Formatted string like "key1='value1', key2='value2'"
    """
    return ", ".join(f"{k}={v!r}" for k, v in tool_args.items()) if tool_args else ""


def finish_tool_func() -> str:
    """Return the finish tool message.

    This is used as the callable for the finish tool, which signals task completion.

    Returns:
        The finish tool message string
    """
    return FINISH_TOOL_MESSAGE


def handle_error_tool_func() -> str:
    """Return the handle error tool message.

    This is used as the callable for the handle_error tool, which signals a processing error.

    Returns:
        The handle error tool message string
    """
    return HANDLE_ERROR_TOOL_MESSAGE
