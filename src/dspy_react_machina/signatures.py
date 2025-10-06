#!/usr/bin/env python3
"""
Signature Construction for ReActMachina

This module provides functions for building DSPy signatures for different machine states
in the ReActMachina module. Each state has its own signature with specific input and output fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

import dspy
from dspy_react_machina.conversation import Fields
from dspy_react_machina.state_machine import MachineStates
from dspy_react_machina.tools import finish_tool_func, handle_error_tool_func, INTERNAL_ONLY_TOOLS, SpecialTools
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class StateFieldSpec(TypedDict):
    """Type definition for state field specification.

    Defines which input and output fields are expected for each machine state.
    Used for both field masking and validation.
    """

    inputs: list[str]
    outputs: list[str]


def create_state_field_specs(
    original_signature_output_fields: list[str] | None = None,
    original_signature_input_fields: list[str] | None = None,
    predictor_class: type[dspy.Predict] | type[dspy.ChainOfThought] = dspy.ChainOfThought,
) -> dict[str, StateFieldSpec]:
    """Create the state field specifications for ReActMachina module.

    Defines which input and output fields are expected for each machine state.
    These specifications are used for both field masking and validation.

    Args:
        original_signature_output_fields: List of output field names from the original signature
        original_signature_input_fields: List of input field names from the original signature
        predictor_class: The predictor class to use (dspy.Predict or dspy.ChainOfThought)

    Returns:
        Dict mapping state names to StateFieldSpec configurations
    """
    if original_signature_output_fields is None:
        original_signature_output_fields = []
    if original_signature_input_fields is None:
        original_signature_input_fields = []

    # Build outputs based on predictor class
    react_outputs = _build_output_fields(predictor_class, ["tool_name", "tool_args", "response"])
    finish_outputs = _build_output_fields(predictor_class, original_signature_output_fields)

    return {
        MachineStates.USER_QUERY: {
            "inputs": ["machine_state"] + original_signature_input_fields + ["history"],
            "outputs": react_outputs,
        },
        MachineStates.TOOL_RESULT: {
            "inputs": ["machine_state", "tool_result", "history"],
            "outputs": react_outputs,
        },
        MachineStates.INTERRUPTED: {
            "inputs": ["machine_state", "tool_result", "interruption_instructions", "history"],
            "outputs": finish_outputs,  # INTERRUPTED produces final outputs (like FINISH) but for interrupted execution
        },
        MachineStates.FINISH: {
            "inputs": ["machine_state", "tool_result", "history"],
            "outputs": finish_outputs,
        },
    }


def create_finish_tool(signature: type[Signature]) -> dspy.Tool:
    """Create the finish tool for task completion.

    Args:
        signature: The original signature defining the task outputs

    Returns:
        A dspy.Tool for finishing the task
    """
    outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])

    return dspy.Tool(
        func=finish_tool_func,
        name=SpecialTools.FINISH,
        desc=f"Signal task completion when ready to produce the final outputs: {outputs}",
        args={},
    )


def create_handle_error_tool() -> dspy.Tool:
    """Create the handle_error tool for error handling.

    Returns:
        A dspy.Tool for handling errors
    """
    return dspy.Tool(
        func=handle_error_tool_func,
        name=SpecialTools.ERROR,
        desc="Signal that an error occurred during processing. Used internally for error recovery.",
        args={},
    )


def build_instructions(signature: type[Signature], tools: dict[str, dspy.Tool]) -> list[str]:
    """Build the instructions for the unified signature.

    Args:
        signature: The original signature
        tools: Dictionary of available tools

    Returns:
        List of instruction strings
    """
    instr = [f"{signature.instructions}\n"] if signature.instructions else []

    instr.append("---\n")
    instr.append("You can use the following tools to assist you:\n")

    # Add tool descriptions to instructions, excluding internal-only tools
    # Internal-only tools (like 'error') are used for flow control and should not be exposed to the LLM
    idx = 1
    for tool_name, tool in tools.items():
        if tool_name not in INTERNAL_ONLY_TOOLS:
            instr.append(f"({idx}) {tool}\n")
            idx += 1

    instr.append("When providing `tool_args`, the value must be in JSON format.")
    return instr


def build_state_signatures(
    signature: type[Signature],
    tools: dict[str, dspy.Tool],
    instr: list[str],
    predictor_class: type[dspy.Predict] | type[dspy.ChainOfThought],
) -> dict[str, type[Signature]]:
    """Build separate signatures for each machine state.

    Args:
        signature: The original signature
        tools: Dictionary of available tools
        instr: List of instruction strings
        predictor_class: The predictor class to use

    Returns:
        Dictionary mapping state names to their signatures
    """

    # fmt: off
    return {
        MachineStates.USER_QUERY: _build_state_signature(MachineStates.USER_QUERY, signature, tools, instr, predictor_class),
        MachineStates.TOOL_RESULT: _build_state_signature(MachineStates.TOOL_RESULT, signature, tools, instr, predictor_class),
        MachineStates.INTERRUPTED: _build_state_signature(MachineStates.INTERRUPTED, signature, tools, instr, predictor_class),
        MachineStates.FINISH: _build_state_signature(MachineStates.FINISH, signature, tools, instr, predictor_class),
    }


def _get_field_description(field, name: str) -> str:
    """Extract field description from multiple sources or use default.

    Tries to extract description from:
    1. field.json_schema_extra["desc"]
    2. field.description
    3. Falls back to placeholder: ${name}

    Args:
        field: The field to extract description from
        name: Field name to use in fallback

    Returns:
        Field description string
    """
    # Try json_schema_extra first
    if hasattr(field, "json_schema_extra") and field.json_schema_extra:
        if isinstance(field.json_schema_extra, dict):
            desc = field.json_schema_extra.get("desc")
            if desc:
                return desc

    # Try description attribute
    if hasattr(field, "description") and field.description:
        return field.description

    # Fall back to placeholder
    return f"${{{name}}}"


def _build_output_fields(
    predictor_class: type[dspy.Predict] | type[dspy.ChainOfThought], additional_fields: list[str]
) -> list[str]:
    """Build output field list with optional reasoning field.

    Args:
        predictor_class: The predictor class (determines if reasoning is included)
        additional_fields: Additional fields to include after reasoning

    Returns:
        List of output field names
    """
    output_fields = []
    if predictor_class == dspy.ChainOfThought:
        output_fields.append("reasoning")
    output_fields.extend(additional_fields)
    return output_fields


def _build_input_fields_for_state(state: str, signature: type[Signature]) -> dict[str, tuple[type, FieldInfo]]:
    """Build input fields for a specific machine state.

    Args:
        state: The machine state
        signature: The original signature

    Returns:
        Dictionary of input field definitions
    """
    input_fields = {}

    # All states have machine_state
    input_fields[Fields.MACHINE_STATE] = (
        Literal["user_query", "tool_result", "interrupted", "finish"],
        dspy.InputField(desc="Current machine state"),
    )

    # State-specific input fields
    if state == MachineStates.USER_QUERY:
        # User query state uses original signature inputs
        for name, field in signature.input_fields.items():
            input_fields[name] = (field.annotation, dspy.InputField(desc=_get_field_description(field, name)))
    elif state == MachineStates.INTERRUPTED:
        # Interrupted state uses both tool_result and interruption_instructions as inputs
        input_fields[Fields.TOOL_RESULT] = (
            str,
            dspy.InputField(desc="Tool execution observation or result"),
        )
        input_fields[Fields.INTERRUPTION_INSTRUCTIONS] = (
            str,
            dspy.InputField(desc="Interruption message with guidance for handling the interruption"),
        )
    else:
        # All other states use tool_result as input
        input_fields[Fields.TOOL_RESULT] = (
            str,
            dspy.InputField(desc="Tool execution observation or result"),
        )

    # All states have history
    input_fields[Fields.HISTORY] = (
        dspy.History,
        dspy.InputField(desc="Full conversation and tool interaction history"),
    )

    return input_fields


def _build_output_fields_for_state(
    state: str,
    signature: type[Signature],
    tools: dict[str, dspy.Tool],
    predictor_class: type[dspy.Predict] | type[dspy.ChainOfThought],
) -> dict[str, tuple[type, FieldInfo]]:
    """Build output fields for a specific machine state.

    Args:
        state: The machine state
        signature: The original signature
        tools: Dictionary of available tools
        predictor_class: The predictor class to use

    Returns:
        Dictionary of output field definitions
    """
    output_fields = {}

    # Add reasoning field if using ChainOfThought
    if predictor_class == dspy.ChainOfThought:
        output_fields[Fields.REASONING] = (str, dspy.OutputField(desc="Step-by-step reasoning process"))

    # State-specific output fields
    if state in (MachineStates.FINISH, MachineStates.INTERRUPTED):
        # FINISH and INTERRUPTED states output the original signature outputs
        for name, field in signature.output_fields.items():
            output_fields[name] = (field.annotation, dspy.OutputField(desc=_get_field_description(field, name)))
    else:
        # All other states output ReAct fields
        output_fields[Fields.TOOL_NAME] = (
            str,
            dspy.OutputField(desc=f"Name of tool to call. Must be one of: {', '.join(list(tools.keys()))}"),
        )
        output_fields[Fields.TOOL_ARGS] = (
            dict,
            dspy.OutputField(desc="Arguments for tool call in JSON format"),
        )
        output_fields[Fields.RESPONSE] = (
            str,
            dspy.OutputField(desc="Tool call description or final answer to user"),
        )

    return output_fields


def _build_state_signature(
    state: str,
    signature: type[Signature],
    tools: dict[str, dspy.Tool],
    instr: list[str],
    predictor_class: type[dspy.Predict] | type[dspy.ChainOfThought],
) -> type[Signature]:
    """Build a signature for a specific machine state.

    Args:
        state: The machine state (user_query, tool_result, interrupted, finish)
        signature: The original signature
        tools: Dictionary of available tools
        instr: List of instruction strings
        predictor_class: The predictor class to use

    Returns:
        Signature for the specified state
    """
    # Build input and output fields for this state
    input_fields = _build_input_fields_for_state(state, signature)
    output_fields = _build_output_fields_for_state(state, signature, tools, predictor_class)

    return dspy.Signature({**input_fields, **output_fields}, "\n".join(instr))  # type: ignore[call-arg, return-value]
