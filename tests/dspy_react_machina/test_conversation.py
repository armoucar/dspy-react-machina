"""Tests for conversation module - validates conversation history management."""

from typing import Literal

import dspy

from dspy_react_machina import ReActMachina
from dspy_react_machina.conversation import (
    create_format_error_response,
    create_interaction_record,
    extract_output_values,
    Fields,
)
from dspy_react_machina.state_machine import MachineStates
from dspy_react_machina.tools import SpecialTools


class TestExtractOutputValues:
    """Test extract_output_values function."""

    def test_extract_output_values_all_fields(self):
        """Test extracting output values from prediction (only original signature fields) - multi-output."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "text -> category: str, confidence: float"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Create a prediction with category and confidence (from original signature)
        prediction = dspy.Prediction(
            reasoning="Some reasoning",  # ReAct field - not extracted
            tool_name="example_tool",  # ReAct field - not extracted
            category="positive",  # Original signature field - extracted
            confidence=0.95,  # Original signature field - extracted
        )

        # Act
        result = extract_output_values(prediction, agent.signature, only_existing=False)

        # Assert - only extracts original signature output fields
        assert "category" in result
        assert result["category"] == "positive"
        assert "confidence" in result
        assert result["confidence"] == 0.95
        # ReAct fields are not extracted by this method
        assert "reasoning" not in result
        assert "tool_name" not in result
        assert len(result) == 2  # Only "category" and "confidence" from original signature

    def test_extract_output_values_only_existing(self):
        """Test extracting only existing output values from prediction"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        # Many output fields signature
        signature = "input -> output1: str, output2: bool, output3: list[str]"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Create a prediction with only some output fields
        prediction = dspy.Prediction(
            tool_name="example_tool",  # ReAct field - not extracted
            tool_args={"query": "test"},  # ReAct field - not extracted
            output1="test",  # Original field - extracted
            output2=True,  # Original field - extracted
            # output3 is missing
        )

        # Act - only_existing=True means don't include fields not in prediction
        result = extract_output_values(prediction, agent.signature, only_existing=True)

        # Assert - only output1 and output2 are extracted
        assert len(result) == 2
        assert "output1" in result
        assert "output2" in result
        assert "output3" not in result  # Missing, so not extracted with only_existing=True

    def test_extract_output_values_with_none_defaults(self):
        """Test extracting output values includes None for missing fields when only_existing=False - complex types."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "data: str -> analysis: dict[str, int]"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Create a prediction without analysis field
        prediction = dspy.Prediction(tool_name="example_tool")

        # Act - only_existing=False means include all fields with None defaults
        result = extract_output_values(prediction, agent.signature, only_existing=False)

        # Assert
        assert "analysis" in result
        assert result["analysis"] is None  # Field not in prediction, so None default


class TestCreateInteractionRecord:
    """Test create_interaction_record function."""

    def test_create_interaction_record_user_query_state(self):
        """Test _create_interaction_record for USER_QUERY state"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        class QuerySignature(dspy.Signature):
            """Process queries."""

            user_query: str = dspy.InputField()
            response_text: str = dspy.OutputField()

        tools = [example_tool]
        agent = ReActMachina(signature=QuerySignature, tools=tools, predictor_class=dspy.ChainOfThought)

        prediction = dspy.Prediction(
            reasoning="Let me search for this",
            tool_name="example_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )
        state = MachineStates.USER_QUERY
        original_inputs = {"user_query": "What is the answer?"}
        tool_result = None

        # Act
        record = create_interaction_record(prediction, state, agent._first_input_field, original_inputs, tool_result)

        # Assert
        assert record[Fields.MACHINE_STATE] == MachineStates.USER_QUERY
        assert record[Fields.TOOL_NAME] == "example_tool"
        assert record[Fields.TOOL_ARGS] == {"query": "test"}
        assert record[Fields.RESPONSE] == "Searching..."
        assert record[Fields.REASONING] == "Let me search for this"
        assert "user_query" in record
        assert record["user_query"] == "What is the answer?"

    def test_create_interaction_record_tool_result_state(self):
        """Test _create_interaction_record for TOOL_RESULT state"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        class SentimentSignature(dspy.Signature):
            """Classify sentiment."""

            text: str = dspy.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

        tools = [example_tool]
        agent = ReActMachina(signature=SentimentSignature, tools=tools, predictor_class=dspy.Predict)

        prediction = dspy.Prediction(
            tool_name="example_tool", tool_args={"query": "test"}, response="Continuing search..."
        )
        state = MachineStates.TOOL_RESULT
        original_inputs = {"text": "What is the answer?"}
        tool_result = "[example_tool(query='test')] Result for test"

        # Act
        record = create_interaction_record(prediction, state, agent._first_input_field, original_inputs, tool_result)

        # Assert
        assert record[Fields.MACHINE_STATE] == MachineStates.TOOL_RESULT
        assert record[Fields.TOOL_RESULT] == tool_result
        # Should NOT have reasoning since using Predict
        assert Fields.REASONING not in record

    def test_create_interaction_record_finish_action(self):
        """Test _create_interaction_record with finish tool."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        prediction = dspy.Prediction(
            reasoning="I have the answer now",
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to provide answer",
        )
        state = MachineStates.TOOL_RESULT
        original_inputs = {"question": "What is 2+2?"}
        tool_result = "Previous tool result"

        # Act
        record = create_interaction_record(prediction, state, agent._first_input_field, original_inputs, tool_result)

        # Assert
        assert record[Fields.TOOL_NAME] == SpecialTools.FINISH


class TestCreateFormatErrorResponse:
    """Test create_format_error_response function."""

    def test_create_format_error_response_with_cot(self):
        """Test _create_format_error_response with ChainOfThought predictor at step 0."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)
        trajectory = {}
        step = 0

        # Act
        response = create_format_error_response(trajectory, step, agent._has_reasoning)

        # Assert
        assert Fields.TOOL_NAME in response
        assert response[Fields.TOOL_NAME] == SpecialTools.ERROR
        assert Fields.TOOL_ARGS in response
        assert response[Fields.TOOL_ARGS] == {}
        assert Fields.RESPONSE in response
        # At step 0, should use fallback message
        assert "retrying with correct field format" in response[Fields.RESPONSE].lower()
        # Should include reasoning for ChainOfThought
        assert Fields.REASONING in response
        assert "not able to process the last request" in response[Fields.REASONING].lower()

    def test_create_format_error_response_with_predict(self):
        """Test _create_format_error_response with Predict predictor at step 0."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.Predict)
        trajectory = {}
        step = 0

        # Act
        response = create_format_error_response(trajectory, step, agent._has_reasoning)

        # Assert
        assert Fields.TOOL_NAME in response
        assert response[Fields.TOOL_NAME] == SpecialTools.ERROR
        assert Fields.TOOL_ARGS in response
        assert response[Fields.TOOL_ARGS] == {}
        assert Fields.RESPONSE in response
        # At step 0, should use fallback message
        assert "retrying with correct field format" in response[Fields.RESPONSE].lower()
        # Should NOT include reasoning for Predict
        assert Fields.REASONING not in response

    def test_create_format_error_response_with_last_tool_result(self):
        """Test _create_format_error_response includes last tool result at step > 0."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Build trajectory with a previous tool call
        # Note: observation includes the formatted tool call (as produced by _execute_tool)
        trajectory = {
            "tool_name_0": "example_tool",
            "tool_args_0": {"query": "test"},
            "observation_0": "[example_tool(query='test')] Result for test",
        }
        step = 1

        # Act
        response = create_format_error_response(trajectory, step, agent._has_reasoning)

        # Assert
        assert Fields.TOOL_NAME in response
        assert response[Fields.TOOL_NAME] == SpecialTools.ERROR
        assert Fields.RESPONSE in response
        # Should include last tool result
        assert "The last tool we were able to get a result from was:" in response[Fields.RESPONSE]
        assert "example_tool" in response[Fields.RESPONSE]
        assert "query='test'" in response[Fields.RESPONSE]
        assert "Result for test" in response[Fields.RESPONSE]

    def test_create_format_error_response_skips_error_tools(self):
        """Test _create_format_error_response skips error/timeout when looking for last result."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Build trajectory with error and success
        # Note: observations include the formatted tool call (as produced by _execute_tool)
        trajectory = {
            "tool_name_0": "example_tool",
            "tool_args_0": {"query": "test"},
            "observation_0": "[example_tool(query='test')] Result for test",
            "tool_name_1": SpecialTools.ERROR,
            "tool_args_1": {},
            "observation_1": "[error()] There was an error...",
        }
        step = 2

        # Act
        response = create_format_error_response(trajectory, step, agent._has_reasoning)

        # Assert
        # Should skip error and find example_tool
        assert "The last tool we were able to get a result from was: " in response[Fields.RESPONSE]
        assert "example_tool" in response[Fields.RESPONSE]
        assert "Result for test" in response[Fields.RESPONSE]
        # Should NOT include error tool name in response (just the last successful tool)
        assert (
            "error" not in response[Fields.RESPONSE].lower()
            or "there was an error" in response[Fields.RESPONSE].lower()
        )
