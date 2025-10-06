#!/usr/bin/env python3
"""
Custom adapter that maintains a unified system prompt while dynamically masking fields
based on the current state.

The ReActMachina module uses 4 states (USER_QUERY, TOOL_RESULT, CONTINUE, FINISH),
each with its own signature. This adapter ensures all states receive the same system
prompt, preventing inconsistent prompts across state transitions.

It uses field masking to show only state-relevant fields in user messages while keeping
the system prompt constant across all LLM calls. This prevents the LLM from generating
irrelevant fields while maintaining full context awareness.
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Any, TypedDict

import dspy
from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
)
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError
from dspy_react_machina.signatures import create_state_field_specs
from dspy_react_machina.state_machine import MachineStates

logger = logging.getLogger(__name__)


# UI String Constants
MODULE_PROMPT = textwrap.dedent("""
    You are a ReAct (Reasoning and Acting) agent that solves tasks to completion by progressing through a state machine.
    Each state represents a function with specific inputs and outputs, and the `machine_state` field determines which function is active.
    You progress through states by reasoning step by step and using available tools to gather information.
    When calling tools, provide the tool name and arguments. When ready to answer, ensure all required outputs are provided.

    Your objective is:
""")

FIELD_DESCRIPTION_PREFIX = (
    "This agent operates as a state machine. "
    "The `machine_state` field determines which function (inputs → outputs) is active."
)

OUTPUT_REQUIREMENTS_FORMAT = (
    "Respond using the exact field format `[[ ## field_name ## ]]`. "
    "Required fields in order: {fields}, ending with `[[ ## completed ## ]]`. "
    "Format: field marker on one line, value on next line, blank line between fields."
)

OUTPUT_EXCLUSION_WARNING = "Do NOT generate the following fields for this state: {excluded_fields}."


# TypedDict for state field specification
class StateFieldSpec(TypedDict):
    """Type definition for state field specification.

    Defines which input and output fields are expected for each machine state.
    Used for both field masking and validation.
    """

    inputs: list[str]
    outputs: list[str]


class ReActMachinaAdapter(ChatAdapter):
    """ReActMachina adapter that maintains unified system prompt while masking user message fields based on state."""

    # Class constants
    FIELD_HEADER_PATTERN = re.compile(r"\[\[ ## (\w+) ## \]\]")

    # Initialization

    def __init__(
        self,
        original_signature: type[Signature],
        state_signatures: dict[str, type[Signature]],
        predictor_class: type["dspy.Predict"] | type["dspy.ChainOfThought"] = dspy.ChainOfThought,
        _debug_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the adapter with original and state-specific signatures.

        Args:
            original_signature: The original signature before ReAct fields were added
            state_signatures: Dictionary mapping state names to their signatures
            predictor_class: The predictor class to use (dspy.Predict or dspy.ChainOfThought)
            _debug_config: Internal debug config dict with keys: fail_mode, fail_step (NOT INTENDED FOR PRODUCTION)
        """
        super().__init__(**kwargs)
        self.state_signatures = state_signatures
        self._debug_config = _debug_config or {}
        self._current_step: int | None = None  # Track current step for failure injection

        # Create unified view of all fields for system prompt generation
        self.all_input_fields = self._collect_all_fields([sig.input_fields for sig in state_signatures.values()])
        self.all_output_fields = self._collect_all_fields([sig.output_fields for sig in state_signatures.values()])

        # Extract field names from original signature
        original_output_fields = list(original_signature.output_fields.keys())
        original_input_fields = list(original_signature.input_fields.keys())

        self.state_field_specs = create_state_field_specs(
            original_output_fields, original_input_fields, predictor_class
        )

    # Private Utility Methods

    def _debug_set_current_step(self, step: int) -> None:
        """Set current step for debug failure injection (INTERNAL DEBUG ONLY)."""
        self._current_step = step

    @staticmethod
    def _collect_all_fields(field_dicts: list[dict]) -> dict:
        """Collect all unique fields from multiple field dictionaries.

        Args:
            field_dicts: List of field dictionaries from different signatures

        Returns:
            Dictionary containing all unique fields (by name)
        """
        all_fields = {}
        for field_dict in field_dicts:
            for name, field in field_dict.items():
                if name not in all_fields:
                    all_fields[name] = field
        return all_fields

    @staticmethod
    def _is_history_field(field) -> bool:
        """Check if a field is a History field.

        Args:
            field: Field to check

        Returns:
            True if the field is a History field, False otherwise
        """
        return hasattr(field, "annotation") and field.annotation == dspy.History

    @staticmethod
    def _select_fields(field_names: tuple[str, ...], fields_dict: dict) -> dict:
        """Select specific fields from a field dictionary.

        Args:
            field_names: Names of fields to select
            fields_dict: Dictionary of available fields

        Returns:
            Dictionary containing only the selected fields
        """
        return {k: fields_dict[k] for k in field_names}

    # Override Methods from ChatAdapter

    def format_field_description(self, signature: type[Signature]) -> str:
        """Use all fields from all state signatures for field descriptions."""
        # Exclude History field from descriptions (it's internal state, not user input)
        input_fields_without_history = {k: v for k, v in self.all_input_fields.items() if not self._is_history_field(v)}

        return (
            f"{FIELD_DESCRIPTION_PREFIX}\n\n"
            f"These are possible input fields:\n\n{get_field_description_string(input_fields_without_history)}"
            f"\n\nThese are possible output fields:\n\n{get_field_description_string(self.all_output_fields)}\n"
        )

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Always use the unified signature for field structure."""

        parts = []

        # Group states with identical structures to reduce redundancy
        structure_groups = self._group_states_by_structure()

        # Document each unique structure with grouped state names
        for structure_key, state_names in structure_groups.items():
            input_field_names, output_field_names = structure_key
            state_display = self._format_state_names(state_names)

            parts.append("\n---\n")
            parts.append(f"For the {state_display} state, messages are structured as:")

            # Add input and output field documentation
            parts.extend(self._format_field_documentation(input_field_names, output_field_names))

        parts.append("\n---")
        parts.append("\nEvery output message completes with: [[ ## completed ## ]]")
        parts.append("\n---")

        return "\n".join(parts).strip()

    def format_task_description(self, signature: type[Signature]) -> str:
        """Use task description from state signatures (all have same instructions)."""
        # All state signatures share the same instructions, so we can use any of them
        first_signature = next(iter(self.state_signatures.values()))
        instructions = textwrap.dedent(first_signature.instructions).strip()

        return MODULE_PROMPT + "\n" + instructions

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format user message content with state-based field masking."""
        # Pre-conditions
        assert isinstance(inputs, dict), f"Inputs must be dict, got {type(inputs)}"
        assert "machine_state" in inputs, "Machine state must be present in inputs"

        messages = [prefix]

        # Determine the current machine state from inputs
        machine_state = inputs.get("machine_state", MachineStates.USER_QUERY)

        # Get allowed input fields for this machine state based on field specification
        field_spec = self.state_field_specs.get(machine_state, {})
        allowed_input_fields = field_spec.get("inputs", list(self.all_input_fields.keys()))

        # Add machine_state field first
        messages.append(self._format_machine_state_field(inputs))

        # Add other allowed fields
        messages.extend(self._format_allowed_fields(inputs, allowed_input_fields))

        # Add output requirements if this is the main request
        if main_request:
            output_requirements = self.user_message_output_requirements_for_state(machine_state)
            messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format assistant message content using only the provided outputs."""
        # Format each field with proper line breaks
        formatted_fields = []
        for k, v in self.all_output_fields.items():
            if k in outputs:
                field_info = v
                value = outputs.get(k, missing_field_message)
                formatted_field_value = format_field_value(field_info=field_info, value=value)
                formatted_fields.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        assistant_message_content = "\n\n".join(formatted_fields)
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output, extracting only fields that are present.

        Args:
            signature: The signature type
            completion: The LLM completion string to parse

        Returns:
            Dictionary of parsed field values

        Raises:
            AdapterParseError: If parsing fails for any field
        """
        # Pre-condition: validate completion is not None/empty
        assert completion is not None, "Completion cannot be None"
        assert isinstance(completion, str), f"Completion must be str, got {type(completion)}"

        # Extract sections from completion
        sections = self._extract_sections(completion)

        # Parse each section into fields
        fields = {}
        for k, v in sections:
            if k and k not in fields and k in self.all_output_fields:
                try:
                    fields[k] = parse_value(v, self.all_output_fields[k].annotation)
                except (TypeError, ValueError, KeyError, AttributeError) as e:
                    # Common parsing errors - re-raise with context
                    self._raise_parse_error(signature, completion, k, v, e)
                except Exception as e:
                    # Unexpected parsing errors - log and re-raise with context
                    logger.exception(
                        f"Unexpected error parsing field {k}",
                        extra={"field": k, "value": v, "error_type": type(e).__name__},
                    )
                    self._raise_parse_error(signature, completion, k, v, e)

        return fields

    # Extended Public Methods

    def user_message_output_requirements_for_state(self, machine_state: str) -> str:
        """Generate state-specific output requirements for the user message."""
        field_spec = self.state_field_specs.get(machine_state, {})
        allowed_output_fields = field_spec.get("outputs", list(self.all_output_fields.keys()))

        # Build fields string with type information (handles debug injection internally)
        fields_str = self._build_field_parts(allowed_output_fields)

        # Build main message
        message = OUTPUT_REQUIREMENTS_FORMAT.format(fields=fields_str)

        # Add exclusion warning
        exclusion_warning = self._build_exclusion_warning(allowed_output_fields)
        message += f" {exclusion_warning}"

        return message

    # Structure Analysis

    def _group_states_by_structure(self) -> dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]]:
        """Group states with identical input/output structures."""
        structure_groups = {}
        for state_name, state_signature in self.state_signatures.items():
            # Get input fields (excluding History)
            state_input_fields = [
                name for name, field in state_signature.input_fields.items() if not self._is_history_field(field)
            ]

            # Get output fields
            state_output_fields = list(state_signature.output_fields.keys())

            # Create structure key for grouping
            structure_key = (tuple(state_input_fields), tuple(state_output_fields))

            if structure_key not in structure_groups:
                structure_groups[structure_key] = []
            structure_groups[structure_key].append(state_name)

        return structure_groups

    def _format_state_names(self, state_names: list[str]) -> str:
        """Format state names for display."""
        if len(state_names) == 1:
            return f"`{state_names[0]}`"
        else:
            names_except_last = ", ".join(f"`{name}`" for name in state_names[:-1])
            return f"{names_except_last}, or `{state_names[-1]}`"

    def _format_field_documentation(
        self, input_field_names: tuple[str, ...], output_field_names: tuple[str, ...]
    ) -> list[str]:
        """Format field documentation for input and output fields."""
        documentation_parts = []

        # Input fields
        input_fields = self._select_fields(input_field_names, self.all_input_fields)
        if input_fields:
            documentation_parts.append("\nInput fields:\n")
            documentation_parts.append(
                self._format_signature_fields_for_instructions(input_fields, list(input_field_names))
            )

        # Output fields
        output_fields = self._select_fields(output_field_names, self.all_output_fields)
        if output_fields:
            documentation_parts.append("\nOutput fields:\n")
            documentation_parts.append(
                self._format_signature_fields_for_instructions(output_fields, list(output_field_names))
            )

        return documentation_parts

    def _format_signature_fields_for_instructions(self, fields_dict: dict, field_order: list[str]) -> str:
        """Format signature fields for instructions with optional ordering.

        Args:
            fields_dict: Dictionary of fields to format
            field_order: Optional list specifying the order of fields

        Returns:
            Formatted string representation of fields
        """
        ordered_fields = [(name, fields_dict[name]) for name in field_order if name in fields_dict]

        return self.format_field_with_value(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                for field_name, field_info in ordered_fields
            },
        )

    # User Message Construction

    def _format_machine_state_field(self, inputs: dict[str, Any]) -> str:
        """Format machine_state field."""
        field_info = self.all_input_fields["machine_state"]
        value = inputs["machine_state"]
        formatted_field_value = format_field_value(field_info=field_info, value=value)
        return f"[[ ## machine_state ## ]]\n{formatted_field_value}"

    def _format_allowed_fields(self, inputs: dict[str, Any], allowed_input_fields: list[str]) -> list[str]:
        """Format allowed input fields."""
        formatted_fields = []
        for field_name in allowed_input_fields:
            should_format_field = (
                field_name != "machine_state"  # Skip machine_state since we already handled it
                and field_name in inputs
                and inputs[field_name] is not None
                and field_name in self.all_input_fields
            )
            if should_format_field:
                field_info = self.all_input_fields[field_name]
                value = inputs.get(field_name)
                formatted_field_value = format_field_value(field_info=field_info, value=value)
                formatted_fields.append(f"[[ ## {field_name} ## ]]\n{formatted_field_value}")
        return formatted_fields

    # Output Requirements Construction

    def _build_field_parts(self, allowed_output_fields: list[str]) -> str:
        """Build field parts with type information."""

        def type_info(field_name: str) -> str:
            field = self.all_output_fields[field_name]
            if field.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(field.annotation)})"
            return ""

        # Build field parts list with type information
        field_parts = []
        for field_name in allowed_output_fields:
            if field_name in self.all_output_fields:
                field_parts.append(f"`[[ ## {field_name} ## ]]`{type_info(field_name)}")

        # Check if debug mode is active
        if self._debug_config.get("fail_mode"):
            # Debug mode: inject failures and add debug notice
            from dspy_react_machina._debug_utils import inject_debug_field_parts

            return inject_debug_field_parts(field_parts, self._debug_config, self._current_step)

        # Normal mode: just join field parts
        return ", then ".join(field_parts)

    def _build_exclusion_warning(self, allowed_output_fields: list[str]) -> str:
        """Build warning about excluded fields."""
        all_output_field_names = set(self.all_output_fields.keys())
        excluded_fields = all_output_field_names - set(allowed_output_fields)
        excluded_field_list = ", ".join(f"`[[ ## {field_name} ## ]]`" for field_name in sorted(excluded_fields))
        return OUTPUT_EXCLUSION_WARNING.format(excluded_fields=excluded_field_list)

    # Parsing & Error Handling

    def _extract_sections(self, completion: str) -> list[tuple[str | None, str]]:
        """Extract field sections from completion text.

        Parses LLM output that uses field markers like "[[ ## field_name ## ]]" to separate different fields.
        Each section is extracted with its header (field name) and content.

        Returns:
            List of tuples: (field_name or None, field_content)
        """
        # Start with one section for content before any headers
        sections: list[tuple[str | None, list[str]]] = [(None, [])]

        for line in completion.splitlines():
            match = self.FIELD_HEADER_PATTERN.match(line.strip())
            if match:
                # Found a field header - start a new section
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                # Regular line - add to current section
                sections[-1][1].append(line)

        # Join lines within each section and strip whitespace
        return [(k, "\n".join(v).strip()) for k, v in sections]

    def _raise_parse_error(
        self, signature: type[Signature], completion: str, field_name: str, field_value: str, error: Exception
    ) -> None:
        """Raise a parse error with detailed information."""
        raise AdapterParseError(
            adapter_name="ReActMachinaAdapter",
            signature=signature,  # type: ignore[arg-type]  # dspy expects instance but we have type
            lm_response=completion,
            message=f"Failed to parse field {field_name} with value {field_value} from the LM response. Error message: {error}",
        )
