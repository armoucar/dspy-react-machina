#!/usr/bin/env python3
"""
Conversation and History Management for ReActMachina

This module provides types and functions for managing conversation history
and interaction records in the ReActMachina module.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import dspy

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

# UI String Constants
FORMAT_ERROR_RESPONSE = "Retrying with correct field format."

FORMAT_ERROR_REASONING = (
    "I was not able to process the last request. I recommend retrying the last query "
    "while I will pay extra attention to formatting the response the way it is expected."
)


class Fields(StrEnum):
    """Constants for signature field names used throughout ReActMachina."""

    HISTORY = "history"
    INTERRUPTION_INSTRUCTIONS = "interruption_instructions"
    MACHINE_STATE = "machine_state"
    REASONING = "reasoning"
    RESPONSE = "response"
    TOOL_ARGS = "tool_args"
    TOOL_NAME = "tool_name"
    TOOL_RESULT = "tool_result"


class InteractionRecord(TypedDict, total=False):
    """Type definition for interaction record in history.

    All fields are optional (total=False) since different interaction types
    have different required fields.
    """

    machine_state: str
    tool_name: str
    tool_args: dict[str, Any]
    response: str
    reasoning: str  # Optional: only with ChainOfThought
    tool_result: str  # Optional: only in some states


class ValidationResult(NamedTuple):
    """Result of prediction validation."""

    is_valid: bool
    missing_fields: list[str]


class StateProcessingResult(NamedTuple):
    """Result of state processing."""

    prediction: dspy.Prediction
    updated_history: dspy.History


def create_interaction_record(
    prediction: dspy.Prediction,
    state: str,
    first_input_field: str,
    original_inputs: dict[str, Any],
    tool_result: str | None,
) -> InteractionRecord:
    """Create an interaction record for the history.

    Args:
        prediction: The prediction from the LLM
        state: Current machine state
        first_input_field: Name of the first input field
        original_inputs: Original user inputs
        tool_result: Tool execution result

    Returns:
        InteractionRecord with all relevant fields for this interaction
    """
    from dspy_react_machina.state_machine import MachineStates

    # Get input fields based on state
    input_fields: dict[str, Any] = {}
    if state == MachineStates.USER_QUERY:
        input_fields = {first_input_field: original_inputs[first_input_field]}
    else:
        input_fields = {Fields.TOOL_RESULT: tool_result}

    interaction_record: dict[str, Any] = {
        Fields.MACHINE_STATE: state,
        Fields.TOOL_NAME: prediction.tool_name,
        Fields.TOOL_ARGS: prediction.tool_args,
        Fields.RESPONSE: prediction.response,
        **input_fields,
    }

    # Add reasoning if present (only with ChainOfThought)
    if hasattr(prediction, Fields.REASONING):
        interaction_record[Fields.REASONING] = prediction.reasoning

    return interaction_record  # type: ignore[return-value]


def update_history(history: dspy.History, interaction_record: dict[str, Any] | InteractionRecord) -> dspy.History:
    """Update history with a new interaction record.

    Args:
        history: Current conversation history
        interaction_record: New interaction to add

    Returns:
        Updated history with the new interaction

    Raises:
        AssertionError: If the updated history doesn't have exactly one more message
    """
    updated_history = dspy.History(messages=[*history.messages, interaction_record])  # type: ignore[list-item]

    # Post-condition: new history should have exactly one more message
    assert len(updated_history.messages) == len(history.messages) + 1, (
        "Updated history should have exactly one more message"
    )

    return updated_history


def extract_output_values(
    prediction: dspy.Prediction, signature: type[Signature], only_existing: bool = False
) -> dict[str, Any]:
    """Extract output field values from a prediction.

    Args:
        prediction: The prediction to extract values from
        signature: The signature defining output fields
        only_existing: If True, only include fields that exist in prediction.
                      If False, include all with None defaults.

    Returns:
        Dictionary mapping field names to their values
    """
    output_values = {}
    for name in signature.output_fields:
        if only_existing:
            if hasattr(prediction, name):
                output_values[name] = getattr(prediction, name)
        else:
            output_values[name] = getattr(prediction, name, None)
    return output_values


def get_last_tool_result_message(
    trajectory: dict[str, Any],
    step: int,
) -> str:
    """Get a formatted message about the last successful tool result.

    Args:
        trajectory: The current trajectory dictionary
        step: The current step number

    Returns:
        Formatted message with last tool result, or empty string if no previous result
    """
    from dspy_react_machina.tools import SpecialTools

    if step == 0:
        return ""

    # Look backwards for the last non-error tool call
    for i in range(step - 1, -1, -1):
        tool_name = trajectory.get(f"tool_name_{i}")
        tool_args = trajectory.get(f"tool_args_{i}")
        observation = trajectory.get(f"observation_{i}")

        # Skip error states
        if tool_name in (SpecialTools.ERROR, SpecialTools.TIMEOUT):
            continue

        if tool_name and observation is not None and tool_args is not None:
            obs_str = str(observation)
            return f"The last tool we were able to get a result from was: {obs_str}"

    return ""


def create_format_error_response(
    trajectory: dict[str, Any],
    step: int,
    has_reasoning: bool,
) -> dict[str, Any]:
    """Create a standardized error response for format failures.

    Args:
        trajectory: The current trajectory dictionary
        step: The current step number
        has_reasoning: Whether reasoning field should be included

    Returns:
        Dictionary with error response fields
    """
    from dspy_react_machina.tools import SpecialTools

    # Get last tool result context if available
    last_result_msg = get_last_tool_result_message(trajectory, step)

    response: dict[str, Any] = {
        Fields.TOOL_NAME: SpecialTools.ERROR,
        Fields.TOOL_ARGS: {},
        Fields.RESPONSE: last_result_msg if last_result_msg else FORMAT_ERROR_RESPONSE,
    }

    if has_reasoning:
        response[Fields.REASONING] = FORMAT_ERROR_REASONING

    return response
