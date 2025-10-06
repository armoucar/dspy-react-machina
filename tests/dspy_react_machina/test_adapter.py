"""Tests for ReActMachinaAdapter."""

from textwrap import dedent
from typing import Literal

import dspy
from dspy.signatures.field import InputField, OutputField
from dspy.utils.exceptions import AdapterParseError

from dspy_react_machina import ReActMachina
from dspy_react_machina.adapter import ReActMachinaAdapter, create_state_field_specs
from dspy_react_machina.state_machine import MachineStates


def create_test_state_signatures(original_signature, predictor_class=dspy.ChainOfThought):
    """Helper to create state signatures for testing."""

    # Create a minimal ReActMachina instance to generate signatures
    def dummy_tool():
        """Dummy tool for testing."""
        return "result"

    agent = ReActMachina(signature=original_signature, tools=[dummy_tool], predictor_class=predictor_class)
    return agent.state_signatures


class TestAdapterFoundation:
    """Phase 1: Test adapter initialization and foundational methods."""

    def test_adapter_init(self):
        """Test adapter initialization with valid signatures."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, predictor_class=dspy.ChainOfThought)

        # Act
        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Assert
        assert adapter.state_signatures is not None
        assert len(adapter.state_signatures) == 4
        assert MachineStates.USER_QUERY in adapter.state_signatures
        assert MachineStates.TOOL_RESULT in adapter.state_signatures
        assert MachineStates.INTERRUPTED in adapter.state_signatures
        assert MachineStates.FINISH in adapter.state_signatures
        assert adapter.state_field_specs is not None
        assert len(adapter.state_field_specs) == 4

    def test_create_state_field_specs_with_cot(self):
        """Test state field specs creation with ChainOfThought predictor."""
        # Arrange
        original_output_fields = ["response"]
        original_input_fields = ["context", "question"]
        predictor_class = dspy.ChainOfThought

        # Act
        state_field_specs = create_state_field_specs(
            original_signature_output_fields=original_output_fields,
            original_signature_input_fields=original_input_fields,
            predictor_class=predictor_class,
        )

        # Assert - USER_QUERY state
        assert MachineStates.USER_QUERY in state_field_specs
        user_query_inputs = state_field_specs[MachineStates.USER_QUERY]["inputs"]
        assert "machine_state" in user_query_inputs
        assert "context" in user_query_inputs
        assert "question" in user_query_inputs
        assert "history" in user_query_inputs

        user_query_outputs = state_field_specs[MachineStates.USER_QUERY]["outputs"]
        assert "reasoning" in user_query_outputs  # ChainOfThought adds reasoning
        assert "tool_name" in user_query_outputs
        assert "tool_args" in user_query_outputs
        assert "response" in user_query_outputs

        # Assert - TOOL_RESULT state
        assert MachineStates.TOOL_RESULT in state_field_specs
        tool_result_inputs = state_field_specs[MachineStates.TOOL_RESULT]["inputs"]
        assert "machine_state" in tool_result_inputs
        assert "tool_result" in tool_result_inputs
        assert "history" in tool_result_inputs

        tool_result_outputs = state_field_specs[MachineStates.TOOL_RESULT]["outputs"]
        assert "reasoning" in tool_result_outputs
        assert "tool_name" in tool_result_outputs
        assert "tool_args" in tool_result_outputs
        assert "response" in tool_result_outputs

        # Assert - INTERRUPTED state
        assert MachineStates.INTERRUPTED in state_field_specs
        interrupted_inputs = state_field_specs[MachineStates.INTERRUPTED]["inputs"]
        assert "machine_state" in interrupted_inputs
        assert "tool_result" in interrupted_inputs
        assert "interruption_instructions" in interrupted_inputs
        assert "history" in interrupted_inputs

        # Assert - FINISH state
        assert MachineStates.FINISH in state_field_specs
        finish_outputs = state_field_specs[MachineStates.FINISH]["outputs"]
        assert "reasoning" in finish_outputs  # ChainOfThought adds reasoning
        assert "response" in finish_outputs  # Original signature output field

    def test_create_state_field_specs_with_predict(self):
        """Test state field specs creation with Predict predictor (no reasoning)."""
        # Arrange
        original_output_fields = ["is_valid"]
        original_input_fields = ["statement"]  # Bool output signature
        predictor_class = dspy.Predict

        # Act
        state_field_specs = create_state_field_specs(
            original_signature_output_fields=original_output_fields,
            original_signature_input_fields=original_input_fields,
            predictor_class=predictor_class,
        )

        # Assert - outputs should NOT include reasoning
        user_query_outputs = state_field_specs[MachineStates.USER_QUERY]["outputs"]
        assert "reasoning" not in user_query_outputs
        assert "tool_name" in user_query_outputs
        assert "tool_args" in user_query_outputs
        assert "response" in user_query_outputs

        finish_outputs = state_field_specs[MachineStates.FINISH]["outputs"]
        assert "reasoning" not in finish_outputs
        assert "is_valid" in finish_outputs

    def test_create_state_field_specs_empty_fields(self):
        """Test state field specs creation with no original fields."""
        # Arrange
        original_output_fields = ["result"]
        original_input_fields = ["value"]  # Typed input signature
        predictor_class = dspy.Predict

        # Act
        state_field_specs = create_state_field_specs(
            original_signature_output_fields=original_output_fields,
            original_signature_input_fields=original_input_fields,
            predictor_class=predictor_class,
        )

        # Assert
        assert len(state_field_specs) == 4
        user_query_inputs = state_field_specs[MachineStates.USER_QUERY]["inputs"]
        assert "machine_state" in user_query_inputs
        assert "history" in user_query_inputs
        assert "value" in user_query_inputs
        finish_outputs = state_field_specs[MachineStates.FINISH]["outputs"]
        assert "result" in finish_outputs


class TestAdapterFormatting:
    """Phase 2: Test adapter formatting methods."""

    def test_format_field_description(self):
        """Test format_field_description includes state machine explanation and fields - multi-output."""
        # Arrange
        original_signature = dspy.Signature("text -> category: str, confidence: float")  # Multi-output
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Act - use user_query state signature (method uses self.all_input_fields internally)
        result = adapter.format_field_description(state_signatures[MachineStates.USER_QUERY])

        # Assert
        # Note: Some field descriptions have trailing spaces (e.g., text, category)
        expected = (
            "This agent operates as a state machine. The `machine_state` field determines which function (inputs → outputs) is active.\n"
            "\n"
            "These are possible input fields:\n"
            "\n"
            "1. `machine_state` (Literal['user_query', 'tool_result', 'interrupted', 'finish']): Current machine state\n"
            "2. `text` (str): \n"
            "3. `tool_result` (str): Tool execution observation or result\n"
            "4. `interruption_instructions` (str): Interruption message with guidance for handling the interruption\n"
            "\n"
            "These are possible output fields:\n"
            "\n"
            "1. `reasoning` (str): Step-by-step reasoning process\n"
            "2. `tool_name` (str): Name of tool to call. Must be one of: dummy_tool, finish, error\n"
            "3. `tool_args` (dict): Arguments for tool call in JSON format\n"
            "4. `response` (str): Tool call description or final answer to user\n"
            "5. `category` (str): \n"
            "6. `confidence` (float):\n"
        )
        assert result == expected

    def test_format_field_structure(self):
        """Test format_field_structure creates state-specific documentation - complex types."""
        # Arrange
        original_signature = dspy.Signature("data: str -> analysis: dict[str, int]")  # Complex types
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Act - use user_query state signature (method uses self.state_signatures internally)
        result = adapter.format_field_structure(state_signatures[MachineStates.USER_QUERY])

        # Assert
        expected = dedent("""\
            ---

            For the `user_query` state, messages are structured as:

            Input fields:

            [[ ## machine_state ## ]]
            {machine_state}

            [[ ## data ## ]]
            {data}

            Output fields:

            [[ ## reasoning ## ]]
            {reasoning}

            [[ ## tool_name ## ]]
            {tool_name}

            [[ ## tool_args ## ]]
            {tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

            [[ ## response ## ]]
            {response}

            ---

            For the `tool_result` state, messages are structured as:

            Input fields:

            [[ ## machine_state ## ]]
            {machine_state}

            [[ ## tool_result ## ]]
            {tool_result}

            Output fields:

            [[ ## reasoning ## ]]
            {reasoning}

            [[ ## tool_name ## ]]
            {tool_name}

            [[ ## tool_args ## ]]
            {tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

            [[ ## response ## ]]
            {response}

            ---

            For the `interrupted` state, messages are structured as:

            Input fields:

            [[ ## machine_state ## ]]
            {machine_state}

            [[ ## tool_result ## ]]
            {tool_result}

            [[ ## interruption_instructions ## ]]
            {interruption_instructions}

            Output fields:

            [[ ## reasoning ## ]]
            {reasoning}

            [[ ## analysis ## ]]
            {analysis}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": {"type": "integer"}}

            ---

            For the `finish` state, messages are structured as:

            Input fields:

            [[ ## machine_state ## ]]
            {machine_state}

            [[ ## tool_result ## ]]
            {tool_result}

            Output fields:

            [[ ## reasoning ## ]]
            {reasoning}

            [[ ## analysis ## ]]
            {analysis}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": {"type": "integer"}}

            ---

            Every output message completes with: [[ ## completed ## ]]

            ---""")
        assert result == expected

    def test_format_task_description(self):
        """Test format_task_description includes ReAct agent description."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        # Act
        result = adapter.format_task_description(state_signatures["user_query"])

        # Assert
        expected = dedent("""\

            You are a ReAct (Reasoning and Acting) agent that solves tasks to completion by progressing through a state machine.
            Each state represents a function with specific inputs and outputs, and the `machine_state` field determines which function is active.
            You progress through states by reasoning step by step and using available tools to gather information.
            When calling tools, provide the tool name and arguments. When ready to answer, ensure all required outputs are provided.

            Your objective is:

            Given the fields `question`, produce the fields `answer`.

            ---

            You can use the following tools to assist you:

            (1) dummy_tool, whose description is <desc>Dummy tool for testing.</desc>. It takes arguments {}.

            (2) finish, whose description is <desc>Signal task completion when ready to produce the final outputs: `answer`</desc>. It takes arguments {}.

            When providing `tool_args`, the value must be in JSON format.""")
        assert result == expected

    def test_user_message_output_requirements_for_state_user_query(self):
        """Test output requirements for USER_QUERY state."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Act
        result = adapter.user_message_output_requirements_for_state(MachineStates.USER_QUERY)

        # Assert
        expected = (
            "Respond using the exact field format `[[ ## field_name ## ]]`. "
            "Required fields in order: `[[ ## reasoning ## ]]`, then `[[ ## tool_name ## ]]`, "
            "then `[[ ## tool_args ## ]]` (must be formatted as a valid Python dict), "
            "then `[[ ## response ## ]]`, ending with `[[ ## completed ## ]]`. "
            "Format: field marker on one line, value on next line, blank line between fields. "
            "Do NOT generate the following fields for this state: `[[ ## answer ## ]]`."
        )
        assert result == expected

    def test_user_message_output_requirements_for_state_finish(self):
        """Test output requirements for FINISH state."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Act
        result = adapter.user_message_output_requirements_for_state(MachineStates.FINISH)

        # Assert
        expected = (
            "Respond using the exact field format `[[ ## field_name ## ]]`. "
            "Required fields in order: `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, "
            "ending with `[[ ## completed ## ]]`. "
            "Format: field marker on one line, value on next line, blank line between fields. "
            "Do NOT generate the following fields for this state: `[[ ## response ## ]]`, "
            "`[[ ## tool_args ## ]]`, `[[ ## tool_name ## ]]`."
        )
        assert result == expected

    def test_group_states_by_structure(self):
        """Test states are grouped by identical input/output structure."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Act
        structure_groups = adapter._group_states_by_structure()

        # Assert
        assert isinstance(structure_groups, dict)
        # Each structure key is a tuple of (input_fields_tuple, output_fields_tuple)
        for structure_key, state_names in structure_groups.items():
            assert isinstance(structure_key, tuple)
            assert len(structure_key) == 2  # (inputs, outputs)
            assert isinstance(state_names, list)
            assert len(state_names) > 0

        # TOOL_RESULT and FINISH should have same structure (both use tool_result input)
        tool_result_states = [
            states
            for states in structure_groups.values()
            if MachineStates.TOOL_RESULT in states or MachineStates.FINISH in states
        ]
        # They might be grouped together
        assert len(tool_result_states) > 0


class TestAdapterParsing:
    """Phase 3: Test adapter parsing methods."""

    def test_parse_valid_completion(self):
        """Test parsing well-formed LLM output with multiple fields."""

        # Arrange
        class AnalysisSignature(dspy.Signature):
            """Analyze data and provide insights."""

            data: str = InputField(desc="Data to analyze")
            insight: str = OutputField(desc="Key insight")

        original_signature = AnalysisSignature
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        completion = dedent("""\
            [[ ## reasoning ## ]]
            Let me analyze this data.

            [[ ## tool_name ## ]]
            analyze

            [[ ## completed ## ]]""")

        # Act - parse using user_query state signature
        result = adapter.parse(state_signatures[MachineStates.USER_QUERY], completion)

        # Assert
        assert "reasoning" in result
        assert result["reasoning"] == "Let me analyze this data."
        assert "tool_name" in result
        assert result["tool_name"] == "analyze"

    def test_parse_partial_completion(self):
        """Test parsing completion with only some fields present."""

        # Arrange
        class SentimentSignature(dspy.Signature):
            """Classify sentiment."""

            text: str = InputField()
            sentiment: Literal["positive", "negative", "neutral"] = OutputField()

        original_signature = SentimentSignature
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Completion with only reasoning field
        completion = dedent("""\
            [[ ## reasoning ## ]]
            Just reasoning here.

            [[ ## completed ## ]]""")

        # Act - parse using user_query state signature
        result = adapter.parse(state_signatures[MachineStates.USER_QUERY], completion)

        # Assert
        assert "reasoning" in result
        assert result["reasoning"] == "Just reasoning here."
        assert "tool_name" not in result
        assert "tool_args" not in result
        assert "sentiment" not in result

    def test_parse_with_type_error(self):
        """Test parsing raises error when value type doesn't match annotation - many outputs."""
        # Arrange
        original_signature = dspy.Signature("input -> output1: str, output2: bool, output3: list[str]")  # Many outputs
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        # Completion with invalid JSON for dict field
        completion = dedent("""\
            [[ ## tool_args ## ]]
            this is not valid json for a dict

            [[ ## completed ## ]]""")

        # Act & Assert - parse using user_query state signature
        try:
            adapter.parse(state_signatures[MachineStates.USER_QUERY], completion)
            assert False, "Expected AdapterParseError to be raised"
        except AdapterParseError as e:
            assert "tool_args" in str(e)
            assert "Failed to parse field" in str(e)

    def test_extract_sections_basic(self):
        """Test extracting sections from basic completion."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        completion = dedent("""\
            [[ ## reasoning ## ]]
            Thinking step by step.

            [[ ## answer ## ]]
            The final answer is 42.

            [[ ## completed ## ]]""")

        # Act
        sections = adapter._extract_sections(completion)

        # Assert
        # Should have sections: (None, content_before_first_header), (reasoning, ...), (answer, ...), (completed, ...)
        section_dict = {k: v for k, v in sections if k is not None}
        assert "reasoning" in section_dict
        assert "Thinking step by step." in section_dict["reasoning"]
        assert "answer" in section_dict
        assert "The final answer is 42." in section_dict["answer"]
        assert "completed" in section_dict

    def test_extract_sections_multiline_content(self):
        """Test extracting sections with multi-line content."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        completion = dedent("""\
            [[ ## reasoning ## ]]
            First line of reasoning.
            Second line of reasoning.
            Third line of reasoning.

            [[ ## answer ## ]]
            Multi-line
            answer
            here.

            [[ ## completed ## ]]""")

        # Act
        sections = adapter._extract_sections(completion)

        # Assert
        section_dict = {k: v for k, v in sections if k is not None}
        assert "reasoning" in section_dict
        reasoning_content = section_dict["reasoning"]
        assert "First line of reasoning." in reasoning_content
        assert "Second line of reasoning." in reasoning_content
        assert "Third line of reasoning." in reasoning_content

        assert "answer" in section_dict
        answer_content = section_dict["answer"]
        assert "Multi-line" in answer_content
        assert "answer" in answer_content
        assert "here." in answer_content

    def test_extract_sections_no_headers(self):
        """Test extracting sections from completion with no field markers."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        completion = "Just some plain text without any field markers."

        # Act
        sections = adapter._extract_sections(completion)

        # Assert
        # Should have one section with None as header
        assert len(sections) >= 1
        assert sections[0][0] is None
        assert "Just some plain text" in sections[0][1]


class TestAdapterUserMessages:
    """Phase 4: Test adapter user message formatting with state-based masking."""

    def test_format_user_message_content_user_query_state(self):
        """Test formatting user message for USER_QUERY state - baseline signature."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")  # Baseline
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        inputs = {
            "machine_state": MachineStates.USER_QUERY,
            "question": "What is 2+2?",
        }

        # Act - use user_query state signature
        result = adapter.format_user_message_content(
            signature=state_signatures[MachineStates.USER_QUERY], inputs=inputs, main_request=False
        )

        # Assert
        expected = dedent("""\
            [[ ## machine_state ## ]]
            user_query

            [[ ## question ## ]]
            What is 2+2?""")
        assert result == expected

    def test_format_user_message_content_with_main_request(self):
        """Test formatting user message with main_request=True includes output requirements."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        inputs = {
            "machine_state": MachineStates.USER_QUERY,
            "question": "What is 2+2?",
        }

        # Act - use user_query state signature
        result = adapter.format_user_message_content(
            signature=state_signatures[MachineStates.USER_QUERY], inputs=inputs, main_request=True
        )

        # Assert
        expected = (
            "[[ ## machine_state ## ]]\n"
            "user_query\n"
            "\n"
            "[[ ## question ## ]]\n"
            "What is 2+2?\n"
            "\n"
            "Respond using the exact field format `[[ ## field_name ## ]]`. "
            "Required fields in order: `[[ ## reasoning ## ]]`, then `[[ ## tool_name ## ]]`, "
            "then `[[ ## tool_args ## ]]` (must be formatted as a valid Python dict), "
            "then `[[ ## response ## ]]`, ending with `[[ ## completed ## ]]`. "
            "Format: field marker on one line, value on next line, blank line between fields. "
            "Do NOT generate the following fields for this state: `[[ ## answer ## ]]`."
        )
        assert result == expected

    def test_format_user_message_content_tool_result_state(self):
        """Test formatting user message for TOOL_RESULT state."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        inputs = {
            "machine_state": MachineStates.TOOL_RESULT,
            "tool_result": "Search found: The answer is 4",
        }

        # Act - use tool_result state signature
        result = adapter.format_user_message_content(
            signature=state_signatures[MachineStates.TOOL_RESULT], inputs=inputs, main_request=True
        )

        # Assert
        expected = (
            "[[ ## machine_state ## ]]\n"
            "tool_result\n"
            "\n"
            "[[ ## tool_result ## ]]\n"
            "Search found: The answer is 4\n"
            "\n"
            "Respond using the exact field format `[[ ## field_name ## ]]`. "
            "Required fields in order: `[[ ## reasoning ## ]]`, then `[[ ## tool_name ## ]]`, "
            "then `[[ ## tool_args ## ]]` (must be formatted as a valid Python dict), "
            "then `[[ ## response ## ]]`, ending with `[[ ## completed ## ]]`. "
            "Format: field marker on one line, value on next line, blank line between fields. "
            "Do NOT generate the following fields for this state: `[[ ## answer ## ]]`."
        )
        assert result == expected

    def test_format_user_message_content_with_none_values(self):
        """Test _should_include_field excludes None values."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        inputs = {
            "machine_state": MachineStates.USER_QUERY,
            "question": "What is 2+2?",
            "context": None,  # Should be excluded
        }

        # Act - use user_query state signature
        result = adapter.format_user_message_content(
            signature=state_signatures[MachineStates.USER_QUERY], inputs=inputs, main_request=False
        )

        # Assert
        expected = dedent("""\
            [[ ## machine_state ## ]]
            user_query

            [[ ## question ## ]]
            What is 2+2?""")
        assert result == expected

    def test_format_user_message_content_with_prefix_suffix(self):
        """Test formatting user message with prefix and suffix."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        inputs = {
            "machine_state": MachineStates.USER_QUERY,
            "question": "Test question",
        }

        # Act - use user_query state signature
        result = adapter.format_user_message_content(
            signature=state_signatures[MachineStates.USER_QUERY],
            inputs=inputs,
            prefix="PREFIX TEXT",
            suffix="SUFFIX TEXT",
            main_request=False,
        )

        # Assert
        expected = dedent("""\
            PREFIX TEXT

            [[ ## machine_state ## ]]
            user_query

            [[ ## question ## ]]
            Test question

            SUFFIX TEXT""")
        assert result == expected

    def test_format_allowed_fields_multiple_fields(self):
        """Test _format_allowed_fields with multiple allowed fields."""
        # Arrange
        original_signature = dspy.Signature("question, context -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        inputs = {
            "machine_state": MachineStates.USER_QUERY,
            "question": "What is AI?",
            "context": "Artificial Intelligence context",
        }

        allowed_fields = ["question", "context"]

        # Act
        result = adapter._format_allowed_fields(inputs, allowed_fields)

        # Assert
        expected = ["[[ ## question ## ]]\nWhat is AI?", "[[ ## context ## ]]\nArtificial Intelligence context"]
        assert result == expected

    def test_format_allowed_fields_empty_list(self):
        """Test _format_allowed_fields with no allowed fields."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        inputs = {"machine_state": MachineStates.USER_QUERY, "question": "Test"}
        allowed_fields = []

        # Act
        result = adapter._format_allowed_fields(inputs, allowed_fields)

        # Assert
        assert len(result) == 0


class TestAdapterAssistantMessages:
    """Phase 5: Test adapter assistant message formatting."""

    def test_format_assistant_message_content_single_field(self):
        """Test formatting assistant message with single output field - multi-input."""
        # Arrange
        original_signature = dspy.Signature("context, question -> response")  # Multi-input
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        outputs = {"response": "The response is 42"}

        # Act - use finish state signature which has original outputs
        result = adapter.format_assistant_message_content(
            signature=state_signatures[MachineStates.FINISH], outputs=outputs
        )

        # Assert
        expected = dedent("""\
            [[ ## response ## ]]
            The response is 42

            [[ ## completed ## ]]
            """)
        assert result == expected

    def test_format_assistant_message_content_multiple_fields(self):
        """Test formatting assistant message with multiple output fields."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        outputs = {
            "reasoning": "Let me search for this information",
            "tool_name": "search",
            "tool_args": {"query": "test"},
            "response": "Searching...",
        }

        # Act - use user_query state signature which has ReAct outputs
        result = adapter.format_assistant_message_content(
            signature=state_signatures[MachineStates.USER_QUERY], outputs=outputs
        )

        # Assert
        expected = dedent("""\
            [[ ## reasoning ## ]]
            Let me search for this information

            [[ ## tool_name ## ]]
            search

            [[ ## tool_args ## ]]
            {"query": "test"}

            [[ ## response ## ]]
            Searching...

            [[ ## completed ## ]]
            """)
        assert result == expected

    def test_format_assistant_message_content_with_missing_field_message(self):
        """Test formatting assistant message with missing_field_message parameter."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.Predict)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.Predict,
        )

        outputs = {"answer": "Test answer"}

        # Act - use finish state signature which has original outputs
        result = adapter.format_assistant_message_content(
            signature=state_signatures[MachineStates.FINISH], outputs=outputs, missing_field_message="[MISSING]"
        )

        # Assert
        expected = dedent("""\
            [[ ## answer ## ]]
            Test answer

            [[ ## completed ## ]]
            """)
        assert result == expected

    def test_format_assistant_message_content_partial_outputs(self):
        """Test formatting assistant message with only some output fields present."""
        # Arrange
        original_signature = dspy.Signature("question -> answer")
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Only provide reasoning and tool_name, not answer
        outputs = {
            "reasoning": "Thinking about this...",
            "tool_name": "finish",
        }

        # Act - use user_query state signature which has ReAct outputs
        result = adapter.format_assistant_message_content(
            signature=state_signatures[MachineStates.USER_QUERY], outputs=outputs
        )

        # Assert
        expected = dedent("""\
            [[ ## reasoning ## ]]
            Thinking about this...

            [[ ## tool_name ## ]]
            finish

            [[ ## completed ## ]]
            """)
        assert result == expected


class TestAdapterEdgeCases:
    """Phase 6: Test adapter edge cases and remaining helper methods."""

    def test_build_exclusion_warning_with_exclusions(self):
        """Test _build_exclusion_warning generates warning when fields are excluded - bool output."""
        # Arrange
        original_signature = dspy.Signature("statement -> is_valid: bool")  # Bool output
        state_signatures = create_test_state_signatures(original_signature, dspy.ChainOfThought)

        adapter = ReActMachinaAdapter(
            original_signature=original_signature,
            state_signatures=state_signatures,
            predictor_class=dspy.ChainOfThought,
        )

        # Only allow reasoning and tool_name (exclude is_valid)
        allowed_output_fields = ["reasoning", "tool_name"]

        # Act
        result = adapter._build_exclusion_warning(allowed_output_fields)

        # Assert
        expected = (
            "Do NOT generate the following fields for this state: "
            "`[[ ## is_valid ## ]]`, `[[ ## response ## ]]`, `[[ ## tool_args ## ]]`."
        )
        assert result == expected

    def test_create_state_field_specs_with_none_defaults(self):
        """Test _create_state_field_specs handles None default parameters - list I/O."""
        # Arrange & Act
        state_field_specs = create_state_field_specs(
            original_signature_output_fields=["summary"],  # List I/O signature
            original_signature_input_fields=["items"],
            predictor_class=dspy.Predict,
        )

        # Assert
        assert len(state_field_specs) == 4
        # USER_QUERY should have machine_state, history, and items
        user_query_inputs = state_field_specs[MachineStates.USER_QUERY]["inputs"]
        assert "machine_state" in user_query_inputs
        assert "history" in user_query_inputs
        assert "items" in user_query_inputs
        # Should not have reasoning in outputs (using Predict)
        user_query_outputs = state_field_specs[MachineStates.USER_QUERY]["outputs"]
        assert "reasoning" not in user_query_outputs  # Predict doesn't have reasoning
        assert "tool_name" in user_query_outputs
        assert "tool_args" in user_query_outputs
        assert "response" in user_query_outputs
