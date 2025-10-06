"""Tests for ReActMachina module."""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

import dspy

from dspy_react_machina import ReActMachina
from dspy_react_machina.conversation import Fields
from dspy_react_machina.exceptions import ToolExecutionError, ToolNotFoundError
from dspy_react_machina.react_machina import INTERRUPTION_INSTRUCTIONS
from dspy_react_machina.state_machine import MachineStates
from dspy_react_machina.tools import SpecialTools


class TestModuleBasics:
    """Phase 4: Test module initialization and basic methods."""

    def test_module_init_valid(self):
        """Test module initialization with valid parameters"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool for testing."""
            return f"Result for {query}"

        signature = "context, question -> response"
        tools = [example_tool]

        # Act
        agent = ReActMachina(
            signature=signature,
            tools=tools,
            max_steps=10,
            predictor_class=dspy.ChainOfThought,
        )

        # Assert
        assert agent.signature is not None
        assert agent.max_steps == 10
        assert len(agent.tool_registry) > 0
        assert "example_tool" in agent.tool_registry
        assert SpecialTools.FINISH in agent.tool_registry

    def test_module_init_with_predict(self):
        """Test module initialization with Predict predictor class"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool for testing."""
            return f"Result for {query}"

        signature = "statement -> is_valid: bool"
        tools = [example_tool]

        # Act
        agent = ReActMachina(
            signature=signature,
            tools=tools,
            max_steps=5,
            predictor_class=dspy.Predict,
        )

        # Assert
        assert agent.signature is not None
        assert agent.max_steps == 5
        assert len(agent.tool_registry) > 0

    def test_module_init_invalid_predictor(self):
        """Test module initialization with invalid predictor class raises error"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool for testing."""
            return f"Result for {query}"

        signature = "value: int -> result: str"
        tools = [example_tool]

        # Act & Assert
        try:
            ReActMachina(
                signature=signature,
                tools=tools,
                max_steps=10,
                predictor_class=str,  # Invalid predictor class
            )
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "predictor_class must be either dspy.Predict or dspy.ChainOfThought" in str(e)

    def test_tool_registration(self):
        """Test tools are converted to dspy.Tool and registered"""

        # Arrange
        def tool_one(x: int) -> int:
            """First tool."""
            return x * 2

        def tool_two(y: str) -> str:
            """Second tool."""
            return y.upper()

        signature = "items: list[str] -> summary: str"
        tools = [tool_one, tool_two]

        # Act
        agent = ReActMachina(signature=signature, tools=tools)

        # Assert
        assert "tool_one" in agent.tool_registry
        assert "tool_two" in agent.tool_registry
        assert isinstance(agent.tool_registry["tool_one"], dspy.Tool)
        assert isinstance(agent.tool_registry["tool_two"], dspy.Tool)

    def test_tool_registration_with_dspy_tool(self):
        """Test registering dspy.Tool objects directly"""

        # Arrange
        def my_func(x: int) -> int:
            """My function."""
            return x * 2

        dspy_tool = dspy.Tool(my_func, name="my_custom_tool", desc="Custom tool description")
        signature = "text -> category: str, confidence: float"
        tools = [dspy_tool]

        # Act
        agent = ReActMachina(signature=signature, tools=tools)

        # Assert
        assert "my_custom_tool" in agent.tool_registry
        assert agent.tool_registry["my_custom_tool"] == dspy_tool

    def test_finish_tool_creation(self):
        """Test finish tool is automatically created and registered"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "data: str -> analysis: dict[str, int]"
        tools = [example_tool]

        # Act
        agent = ReActMachina(signature=signature, tools=tools)

        # Assert
        assert SpecialTools.FINISH in agent.tool_registry
        finish_tool = agent.tool_registry[SpecialTools.FINISH]
        assert isinstance(finish_tool, dspy.Tool)
        assert finish_tool.name == SpecialTools.FINISH

    def test_build_unified_signature_contains_react_fields(self):
        """Test state signatures contain all ReAct state machine fields"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        class QuerySignature(dspy.Signature):
            """Process a query and return a response."""

            query: str = dspy.InputField(desc="The input query to process")
            result: str = dspy.OutputField(desc="The processed result")

        tools = [example_tool]

        # Act
        agent = ReActMachina(signature=QuerySignature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Assert - check state signatures are created
        assert len(agent.state_signatures) == 4

        # Check user_query state has original inputs + machine_state + history
        user_query_sig = agent.state_signatures["user_query"]
        assert Fields.MACHINE_STATE in user_query_sig.input_fields
        assert Fields.HISTORY in user_query_sig.input_fields
        assert "query" in user_query_sig.input_fields
        assert Fields.REASONING in user_query_sig.output_fields
        assert Fields.TOOL_NAME in user_query_sig.output_fields
        assert Fields.TOOL_ARGS in user_query_sig.output_fields
        assert Fields.RESPONSE in user_query_sig.output_fields

        # Check tool_result state
        tool_result_sig = agent.state_signatures["tool_result"]
        assert Fields.TOOL_RESULT in tool_result_sig.input_fields
        assert Fields.HISTORY in tool_result_sig.input_fields

        # Check finish state has original outputs
        finish_sig = agent.state_signatures["finish"]
        assert "result" in finish_sig.output_fields

    def test_build_instructions_includes_tools(self):
        """Test instructions include tool descriptions"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search for information."""
            return f"Results for {query}"

        def calculate_tool(expression: str) -> str:
            """Calculate a mathematical expression."""
            return f"Result of {expression}"

        class ClassificationSignature(dspy.Signature):
            """Classify the sentiment of text."""

            text: str = dspy.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

        tools = [search_tool, calculate_tool]

        # Act
        agent = ReActMachina(signature=ClassificationSignature, tools=tools)

        # Assert - check state signature instructions (all state signatures have same instructions)
        instructions = agent.state_signatures["user_query"].instructions

        # User tools should be included
        assert "search_tool" in instructions
        assert "calculate_tool" in instructions
        assert "Search for information" in instructions
        assert "Calculate a mathematical expression" in instructions

        # Special tool 'finish' should be included (LLM needs to know it can call this)
        assert "finish" in instructions

        # Internal-only tool 'error' should NOT be included (used only for internal flow control)
        assert "error" not in instructions


class TestModuleToolExecution:
    """Phase 5: Test module tool execution and error handling methods."""

    def test_execute_tool_success(self):
        """Test successful tool execution with return value"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search for information."""
            return f"Found results for {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Act
        result = agent._execute_tool("search_tool", {"query": "test"})

        # Assert
        assert "search_tool(query='test')" in result
        assert "Found results for test" in result
        assert result.startswith("[search_tool")

    def test_execute_tool_with_multiple_args(self):
        """Test tool execution with multiple arguments"""

        # Arrange
        def calculate_tool(operation: str, x: int, y: int) -> int:
            """Perform calculation."""
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            return 0

        signature = "context, question -> response"
        tools = [calculate_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Act
        result = agent._execute_tool("calculate_tool", {"operation": "add", "x": 5, "y": 3})

        # Assert
        assert "calculate_tool(" in result
        assert "operation='add'" in result
        assert "x=5" in result
        assert "y=3" in result
        assert "8" in result  # Result of 5 + 3

    def test_execute_tool_not_found(self):
        """Test execution of unknown tool name raises ToolNotFoundError"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "statement -> is_valid: bool"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Act & Assert
        with pytest.raises(ToolNotFoundError) as exc_info:
            agent._execute_tool("nonexistent_tool", {"query": "test"})

        # Verify exception details
        assert "nonexistent_tool" in str(exc_info.value)
        assert exc_info.value.tool_name == "nonexistent_tool"
        assert exc_info.value.available_tools is not None

    def test_execute_tool_error(self):
        """Test tool execution when tool raises exception raises ToolExecutionError"""

        # Arrange
        def failing_tool(query: str) -> str:
            """Tool that raises an error."""
            raise ValueError("Tool execution failed")

        signature = "value: int -> result: str"
        tools = [failing_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Act & Assert
        with pytest.raises(ToolExecutionError) as exc_info:
            agent._execute_tool("failing_tool", {"query": "test"})

        # Verify exception details
        assert exc_info.value.tool_name == "failing_tool"
        assert exc_info.value.tool_args == {"query": "test"}
        assert isinstance(exc_info.value.original_error, ValueError)
        assert "Tool execution failed" in str(exc_info.value.original_error)

    def test_handle_error_with_cot(self):
        """Test error handling creates proper trajectory with ChainOfThought"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "items: list[str] -> summary: str"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        error = ValueError("Test error message")
        state = "user_query"
        original_inputs = {"items": ["item1", "item2"]}
        tool_result = None
        history = dspy.History(messages=[])
        step = 0
        trajectory = {}

        # Act
        result = agent._handle_error(error, state, original_inputs, tool_result, history, step, trajectory)

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert result.summary is None  # Original output fields should be None on error
        assert result.steps == 1  # step + 1
        assert result.trajectory == trajectory
        # Verify history was updated with error record
        assert len(result.history.messages) == 1
        error_record = result.history.messages[0]
        assert error_record[Fields.TOOL_NAME] == SpecialTools.ERROR
        assert "Test error message" in error_record[Fields.RESPONSE]
        # Should have reasoning since using ChainOfThought
        assert Fields.REASONING in error_record

    def test_handle_error_with_predict(self):
        """Test error handling creates proper trajectory with Predict (no reasoning) - multi-output."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "text -> category: str, confidence: float"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.Predict)

        error = RuntimeError("Runtime error occurred")
        state = "tool_result"
        original_inputs = {"text": "Some input text"}
        tool_result = "Some tool result"
        history = dspy.History(messages=[{"previous": "message"}])
        step = 2
        trajectory = {"state_0": "user_query"}

        # Act
        result = agent._handle_error(error, state, original_inputs, tool_result, history, step, trajectory)

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert result.category is None
        assert result.confidence is None
        assert result.steps == 3  # step + 1
        # Verify history includes previous messages + error record
        assert len(result.history.messages) == 2
        error_record = result.history.messages[1]
        assert error_record[Fields.TOOL_NAME] == SpecialTools.ERROR
        # Should NOT have reasoning since using Predict
        assert Fields.REASONING not in error_record

    def test_handle_max_steps_with_cot(self):
        """Test max steps handling with ChainOfThought - should call INTERRUPTED state"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "data: str -> analysis: dict[str, int]"
        tools = [example_tool]
        max_steps = 5
        agent = ReActMachina(signature=signature, tools=tools, max_steps=max_steps, predictor_class=dspy.ChainOfThought)

        history = dspy.History(messages=[{"msg": "1"}, {"msg": "2"}])
        trajectory = {"state_0": "user_query", "state_1": "tool_result", "observation_4": "Last tool result"}

        # Mock the INTERRUPTED state predictor to return a final answer
        interrupted_prediction = dspy.Prediction(
            reasoning="Based on the information gathered, here's my analysis",
            analysis={"total": 42, "count": 10},
        )
        agent.state_predictors[MachineStates.INTERRUPTED] = MagicMock(return_value=interrupted_prediction)

        # Act
        result = agent._handle_max_steps(history, trajectory)

        # Assert
        assert isinstance(result, dspy.Prediction)
        # Now outputs should NOT be None - they should come from the INTERRUPTED state
        assert result.analysis is not None
        assert result.analysis == {"total": 42, "count": 10}
        assert result.steps == max_steps
        assert result.trajectory == trajectory
        # Verify INTERRUPTED state was called (not FINISH)
        agent.state_predictors[MachineStates.INTERRUPTED].assert_called_once()
        # Verify the INTERRUPTED predictor was called with correct inputs
        call_args = agent.state_predictors[MachineStates.INTERRUPTED].call_args[1]
        assert call_args[Fields.INTERRUPTION_INSTRUCTIONS] == INTERRUPTION_INSTRUCTIONS
        assert call_args[Fields.MACHINE_STATE] == MachineStates.INTERRUPTED
        assert call_args[Fields.TOOL_RESULT] == "Last tool result"
        # Verify history contains tool_result
        final_message = result.history.messages[-1]
        assert Fields.TOOL_RESULT in final_message
        assert final_message[Fields.TOOL_RESULT] == "Last tool result"

    def test_handle_max_steps_with_predict(self):
        """Test max steps handling with Predict (no reasoning) - should call INTERRUPTED state"""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        max_steps = 10
        agent = ReActMachina(signature=signature, tools=tools, max_steps=max_steps, predictor_class=dspy.Predict)

        history = dspy.History(messages=[])
        trajectory = {"observation_9": "Weather data retrieved"}

        # Mock the INTERRUPTED state predictor to return a final answer
        interrupted_prediction = dspy.Prediction(answer="Based on available information, the answer is 42")
        agent.state_predictors[MachineStates.INTERRUPTED] = MagicMock(return_value=interrupted_prediction)

        # Act
        result = agent._handle_max_steps(history, trajectory)

        # Assert
        assert isinstance(result, dspy.Prediction)
        # Now outputs should NOT be None - they should come from the INTERRUPTED state
        assert result.answer is not None
        assert result.answer == "Based on available information, the answer is 42"
        assert result.steps == max_steps
        # Verify INTERRUPTED state was called (not FINISH)
        agent.state_predictors[MachineStates.INTERRUPTED].assert_called_once()
        # Verify the INTERRUPTED predictor was called with correct inputs
        call_args = agent.state_predictors[MachineStates.INTERRUPTED].call_args[1]
        assert call_args[Fields.INTERRUPTION_INSTRUCTIONS] == INTERRUPTION_INSTRUCTIONS
        assert call_args[Fields.MACHINE_STATE] == MachineStates.INTERRUPTED
        assert call_args[Fields.TOOL_RESULT] == "Weather data retrieved"
        # Verify history contains tool_result
        final_message = result.history.messages[-1]
        assert Fields.TOOL_RESULT in final_message
        assert final_message[Fields.TOOL_RESULT] == "Weather data retrieved"
        # Should NOT have reasoning since using Predict (reasoning is only in trajectory, not in final record)
        assert not hasattr(interrupted_prediction, Fields.REASONING)


class TestModuleProcessMethods:
    """Phase 6: Test module internal process methods."""

    def test_get_input_fields_for_state_user_query_state(self):
        """Test _get_input_fields_for_state returns first input field for USER_QUERY."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        state = MachineStates.USER_QUERY
        original_inputs = {"question": "What is AI?"}
        tool_result = None

        # Act
        result = agent._get_input_fields_for_state(state, original_inputs, tool_result)

        # Assert
        assert "question" in result
        assert result["question"] == "What is AI?"
        assert Fields.TOOL_RESULT not in result

    def test_get_input_fields_for_state_tool_result_state(self):
        """Test _get_input_fields_for_state returns tool_result for non-USER_QUERY states."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        state = MachineStates.TOOL_RESULT
        original_inputs = {"question": "What is AI?"}
        tool_result = "[search(query='AI')] Found information about AI"

        # Act
        result = agent._get_input_fields_for_state(state, original_inputs, tool_result)

        # Assert
        assert Fields.TOOL_RESULT in result
        assert result[Fields.TOOL_RESULT] == tool_result
        # Should not have question field for TOOL_RESULT state
        assert "question" not in result


class TestModuleValidation:
    """Phase 6.5: Test module validation methods for format error handling."""

    def test_validate_react_prediction_valid(self):
        """Test _validate_react_prediction with valid prediction."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        prediction = dspy.Prediction(
            tool_name="example_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )
        state = MachineStates.USER_QUERY

        # Act
        is_valid, missing_fields = agent._validate_react_prediction(prediction, state)

        # Assert
        assert is_valid is True
        assert missing_fields == []

    def test_validate_react_prediction_missing_tool_name(self):
        """Test _validate_react_prediction with missing tool_name."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        prediction = dspy.Prediction(
            tool_args={"query": "test"},
            response="Searching...",
        )
        state = MachineStates.USER_QUERY

        # Act
        is_valid, missing_fields = agent._validate_react_prediction(prediction, state)

        # Assert
        assert is_valid is False
        assert Fields.TOOL_NAME in missing_fields

    def test_validate_react_prediction_missing_multiple_fields(self):
        """Test _validate_react_prediction with multiple missing fields."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        prediction = dspy.Prediction(tool_name="example_tool")
        state = MachineStates.USER_QUERY

        # Act
        is_valid, missing_fields = agent._validate_react_prediction(prediction, state)

        # Assert
        assert is_valid is False
        assert Fields.TOOL_ARGS in missing_fields
        assert Fields.RESPONSE in missing_fields

    def test_validate_react_prediction_none_values(self):
        """Test _validate_react_prediction with None values for required fields."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        prediction = dspy.Prediction(
            tool_name=None,
            tool_args=None,
            response=None,
        )
        state = MachineStates.USER_QUERY

        # Act
        is_valid, missing_fields = agent._validate_react_prediction(prediction, state)

        # Assert
        assert is_valid is False
        assert len(missing_fields) == 3

    def test_validate_react_prediction_finish_state_always_valid(self):
        """Test _validate_react_prediction with FINISH state (always valid)."""

        # Arrange
        def example_tool(query: str) -> str:
            """Example tool."""
            return f"Result for {query}"

        signature = "question -> answer"
        tools = [example_tool]
        agent = ReActMachina(signature=signature, tools=tools)

        # Even with missing fields, FINISH state should be valid
        prediction = dspy.Prediction()
        state = MachineStates.FINISH

        # Act
        is_valid, missing_fields = agent._validate_react_prediction(prediction, state)

        # Assert
        assert is_valid is True
        assert missing_fields == []


class TestModuleIntegration:
    """Phase 7: Test full forward() execution flow with integration."""

    def test_forward_simple_execution_with_mock(self):
        """Test forward() method with mocked predictor for simple tool execution"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found information about {query}"

        signature = "context, question -> response"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Mock the predictor to control outputs
        # First call: tool execution
        first_prediction = dspy.Prediction(
            reasoning="I should search for this",
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )

        # Second call: finish
        second_prediction = dspy.Prediction(
            reasoning="Now I can answer",
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to answer",
        )

        # Third call: final answer
        final_prediction = dspy.Prediction(
            reasoning="Based on the search results",
            response="The answer based on search results",
        )

        mock_predictor = MagicMock(side_effect=[first_prediction, second_prediction, final_prediction])
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(context="Some context", question="What is the answer?")

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "response")
        assert result.response == "The answer based on search results"
        assert hasattr(result, "history")
        assert len(result.history.messages) > 0
        assert hasattr(result, "steps")
        assert hasattr(result, "trajectory")

    def test_forward_max_steps_reached(self):
        """Test forward() calls INTERRUPTED state when max steps reached"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "input -> output1: str, output2: bool, output3: list[str]"  # Many outputs
        tools = [search_tool]
        max_steps = 2
        agent = ReActMachina(signature=signature, tools=tools, max_steps=max_steps, predictor_class=dspy.Predict)

        # Mock predictor to return tool calls for USER_QUERY, TOOL_RESULT states
        # but return final outputs for INTERRUPTED state
        tool_prediction = dspy.Prediction(
            tool_name="search_tool", tool_args={"query": "test"}, response="Searching more..."
        )
        interrupted_prediction = dspy.Prediction(
            output1="Final answer based on gathered info",
            output2=True,
            output3=["result1", "result2"],
        )

        def mock_side_effect(**kwargs):
            if kwargs.get(Fields.MACHINE_STATE) == MachineStates.INTERRUPTED:
                return interrupted_prediction
            return tool_prediction

        mock_predictor = MagicMock(side_effect=mock_side_effect)
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(input="What is the answer?")

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert result.steps == max_steps
        # Should have real outputs from INTERRUPTED state (not None)
        assert result.output1 == "Final answer based on gathered info"
        assert result.output2 is True
        assert result.output3 == ["result1", "result2"]
        # Verify INTERRUPTED state was called (not FINISH)
        interrupted_calls = [
            call
            for call in mock_predictor.call_args_list
            if call[1].get(Fields.MACHINE_STATE) == MachineStates.INTERRUPTED
        ]
        assert len(interrupted_calls) == 1
        # Verify INTERRUPTED was called with INTERRUPTION_INSTRUCTIONS
        assert interrupted_calls[0][1][Fields.INTERRUPTION_INSTRUCTIONS] == INTERRUPTION_INSTRUCTIONS

    def test_forward_with_error_in_loop(self):
        """Test forward() re-raises unexpected errors during execution."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Mock predictor to raise an unexpected error (RuntimeError)
        mock_predictor = MagicMock(side_effect=RuntimeError("LLM API error"))
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act & Assert - unexpected errors should be re-raised
        with pytest.raises(RuntimeError) as exc_info:
            agent.forward(question="What is the answer?")

        assert "LLM API error" in str(exc_info.value)

    def test_forward_with_existing_history(self):
        """Test forward() continues from existing history."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.Predict)

        # Create existing history
        existing_history = dspy.History(messages=[{"previous": "interaction"}])

        # Mock predictor to return finish immediately
        # With new flow: USER_QUERY → finish → TOOL_RESULT → finish → FINISH
        finish_prediction = dspy.Prediction(tool_name=SpecialTools.FINISH, tool_args={}, response="Ready to answer")

        final_prediction = dspy.Prediction(answer="The answer")

        # Need 3 predictions: USER_QUERY state, TOOL_RESULT state, FINISH state
        mock_predictor = MagicMock(side_effect=[finish_prediction, finish_prediction, final_prediction])
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(question="What is the answer?", history=existing_history)

        # Assert
        assert isinstance(result, dspy.Prediction)
        # History should include previous message
        assert len(result.history.messages) > 1
        assert result.history.messages[0] == {"previous": "interaction"}

    def test_process_state_user_query(self):
        """Test _process_state for USER_QUERY state."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Mock predictor
        mock_prediction = dspy.Prediction(
            reasoning="Let me search",
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )

        mock_predictor = MagicMock(return_value=mock_prediction)
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        history = dspy.History(messages=[])
        state = MachineStates.USER_QUERY
        original_inputs = {"question": "What is AI?"}
        tool_result = None
        trajectory = {}
        step = 0

        # Act
        prediction, updated_history = agent._process_state(
            state, original_inputs, tool_result, history, trajectory, step
        )

        # Assert
        assert isinstance(prediction, dspy.Prediction)
        assert prediction.tool_name == "search_tool"
        assert isinstance(updated_history, dspy.History)
        assert len(updated_history.messages) == 1

    def test_process_state_tool_result(self):
        """Test _process_state for TOOL_RESULT state."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.Predict)

        # Mock predictor
        mock_prediction = dspy.Prediction(tool_name=SpecialTools.FINISH, tool_args={}, response="Ready to answer")

        mock_predictor = MagicMock(return_value=mock_prediction)
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        history = dspy.History(messages=[{"previous": "message"}])
        state = MachineStates.TOOL_RESULT
        original_inputs = {"question": "What is AI?"}
        tool_result = "[search_tool(query='AI')] Found information about AI"
        trajectory = {}
        step = 1

        # Act
        prediction, updated_history = agent._process_state(
            state, original_inputs, tool_result, history, trajectory, step
        )

        # Assert
        assert isinstance(prediction, dspy.Prediction)
        assert prediction.tool_name == SpecialTools.FINISH
        assert len(updated_history.messages) == 2

    def test_process_finish(self):
        """Test _process_finish creates final prediction with outputs."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Mock predictor for finish state
        final_prediction = dspy.Prediction(
            reasoning="Based on my research",
            answer="The answer is 42",
        )

        mock_predictor = MagicMock(return_value=final_prediction)
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        history = dspy.History(messages=[{"step": "1"}])
        observation = "[finish()] Task complete. Ready to provide final outputs."
        step = 2
        trajectory = {"state_0": "user_query", "state_1": "tool_result"}

        # Act
        result = agent._process_finish(observation, history, step, trajectory)

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert result.answer == "The answer is 42"
        assert result.steps == 3  # step + 1
        assert result.trajectory == trajectory
        # History should include finish interaction
        assert len(result.history.messages) == 2

    def test_process_state_with_malformed_prediction(self):
        """Test _process_state handles malformed predictions with fallback."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Mock predictor to return malformed prediction (missing tool_name)
        malformed_prediction = dspy.Prediction(
            reasoning="Some reasoning",
            tool_args={"query": "test"},
            response="Searching...",
        )

        mock_predictor = MagicMock(return_value=malformed_prediction)
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        history = dspy.History(messages=[])
        state = MachineStates.USER_QUERY
        original_inputs = {"question": "What is AI?"}
        tool_result = None
        trajectory = {}
        step = 0

        # Act
        prediction, updated_history = agent._process_state(
            state, original_inputs, tool_result, history, trajectory, step
        )

        # Assert
        # Should have fallback prediction with handle_error tool_name
        assert prediction.tool_name == SpecialTools.ERROR
        assert prediction.tool_args == {}
        # At step 0, should use fallback message
        assert "retrying with correct field format" in prediction.response.lower()
        # History should contain the fallback prediction
        assert len(updated_history.messages) == 1
        assert updated_history.messages[0][Fields.TOOL_NAME] == SpecialTools.ERROR

    def test_process_state_with_valid_prediction(self):
        """Test _process_state passes through valid predictions unchanged."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, predictor_class=dspy.ChainOfThought)

        # Mock predictor to return valid prediction
        valid_prediction = dspy.Prediction(
            reasoning="Let me search",
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )

        mock_predictor = MagicMock(return_value=valid_prediction)
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        history = dspy.History(messages=[])
        state = MachineStates.USER_QUERY
        original_inputs = {"question": "What is AI?"}
        tool_result = None
        trajectory = {}
        step = 0

        # Act
        prediction, updated_history = agent._process_state(
            state, original_inputs, tool_result, history, trajectory, step
        )

        # Assert
        # Should have original prediction unchanged
        assert prediction.tool_name == "search_tool"
        assert prediction.tool_args == {"query": "test"}
        assert prediction.response == "Searching..."
        # History should contain the valid prediction
        assert len(updated_history.messages) == 1
        assert updated_history.messages[0][Fields.TOOL_NAME] == "search_tool"

    def test_forward_with_malformed_prediction_recovery(self):
        """Test forward() recovers from malformed predictions and continues execution."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found information about {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Mock the predictor to control outputs
        # First call: malformed (missing tool_name) - should trigger fallback
        malformed_prediction = dspy.Prediction(
            reasoning="I should search",
            tool_args={"query": "test"},
            response="Searching...",
        )

        # Second call: valid tool call after receiving error
        recovery_prediction = dspy.Prediction(
            reasoning="Let me try again with correct format",
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching with correct format...",
        )

        # Third call: finish
        finish_prediction = dspy.Prediction(
            reasoning="Now I can answer",
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to answer",
        )

        # Fourth call: final answer
        final_prediction = dspy.Prediction(
            reasoning="Based on the search results",
            answer="The answer based on search results",
        )

        mock_predictor = MagicMock(
            side_effect=[malformed_prediction, recovery_prediction, finish_prediction, final_prediction]
        )
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(question="What is the answer?")

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "answer")
        assert result.answer == "The answer based on search results"
        # History should show the handle_error and recovery
        assert len(result.history.messages) > 0
        # First message should have handle_error tool_name (fallback)
        assert result.history.messages[0][Fields.TOOL_NAME] == SpecialTools.ERROR
        # Second message should have valid tool_name
        assert result.history.messages[1][Fields.TOOL_NAME] == "search_tool"

    def test_trajectory_error_flag_on_success(self):
        """Test trajectory includes error flag set to False for successful tool calls."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.Predict)

        # Mock predictor for successful execution
        tool_prediction = dspy.Prediction(
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )

        finish_prediction = dspy.Prediction(
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to answer",
        )

        final_prediction = dspy.Prediction(answer="The answer")

        mock_predictor = MagicMock(side_effect=[tool_prediction, finish_prediction, final_prediction])
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(question="What is the answer?")

        # Assert
        assert hasattr(result, "trajectory")
        # Error flag should be False for successful tool call
        assert "error_0" in result.trajectory
        assert result.trajectory["error_0"] is False
        # Observation should not start with "Error:"
        assert not result.trajectory["observation_0"].startswith("Error:")

    def test_trajectory_error_flag_on_error(self):
        """Test trajectory includes error flag set to True when tool execution fails."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.Predict)

        # Mock predictor to return malformed prediction (missing tool_name)
        malformed_prediction = dspy.Prediction(
            tool_args={"query": "test"},
            response="Searching...",
        )

        finish_prediction = dspy.Prediction(
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to answer",
        )

        final_prediction = dspy.Prediction(answer="The answer")

        mock_predictor = MagicMock(side_effect=[malformed_prediction, finish_prediction, final_prediction])
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(question="What is the answer?")

        # Assert
        assert hasattr(result, "trajectory")
        # handle_error tool is not marked as error (it's a recovery mechanism)
        assert "error_0" in result.trajectory
        assert result.trajectory["error_0"] is False
        # Observation should be the handle_error message
        assert "error" in result.trajectory["observation_0"].lower()
        # But we can detect handle_error by checking tool_name
        assert result.trajectory["tool_name_0"] == SpecialTools.ERROR

    def test_trajectory_error_flags_multiple_steps(self):
        """Test trajectory tracks error flags correctly across multiple steps."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.Predict)

        # First call: success
        success_prediction = dspy.Prediction(
            tool_name="search_tool",
            tool_args={"query": "test1"},
            response="First search...",
        )

        # Second call: error (malformed)
        error_prediction = dspy.Prediction(
            tool_args={"query": "test2"},
            response="Second search...",
        )

        # Third call: success again
        recovery_prediction = dspy.Prediction(
            tool_name="search_tool",
            tool_args={"query": "test3"},
            response="Third search...",
        )

        # Fourth call: finish
        finish_prediction = dspy.Prediction(
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to answer",
        )

        final_prediction = dspy.Prediction(answer="The answer")

        mock_predictor = MagicMock(
            side_effect=[success_prediction, error_prediction, recovery_prediction, finish_prediction, final_prediction]
        )
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(question="What is the answer?")

        # Assert
        assert hasattr(result, "trajectory")
        # First step: success (error_0 = False)
        assert result.trajectory["error_0"] is False
        # Second step: handle_error (error_1 = False, but tool_name is error)
        assert result.trajectory["error_1"] is False
        assert result.trajectory["tool_name_1"] == SpecialTools.ERROR
        # Third step: success (error_2 = False)
        assert result.trajectory["error_2"] is False
        # Can easily find handle_error steps by checking tool_name
        handle_error_steps = [
            int(k.replace("tool_name_", ""))
            for k in result.trajectory.keys()
            if k.startswith("tool_name_") and result.trajectory[k] == SpecialTools.ERROR
        ]
        assert handle_error_steps == [1]

    def test_trajectory_easy_error_detection(self):
        """Test that clients can easily detect and extract errors from trajectory."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.Predict)

        # Create scenario with one error
        success_prediction = dspy.Prediction(
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )

        error_prediction = dspy.Prediction(
            tool_args={},
            response="Error attempt...",
        )

        finish_prediction = dspy.Prediction(
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Done",
        )

        final_prediction = dspy.Prediction(answer="Answer")

        mock_predictor = MagicMock(
            side_effect=[success_prediction, error_prediction, finish_prediction, final_prediction]
        )
        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = agent.forward(question="Test")
        trajectory = result.trajectory

        # Assert - demonstrate easy handle_error detection patterns
        # Pattern 1: Check if any handle_error occurred
        has_handle_error = any(
            trajectory.get(f"tool_name_{i}") == SpecialTools.ERROR for i in range(10) if f"tool_name_{i}" in trajectory
        )
        assert has_handle_error is True

        # Pattern 2: Get all handle_error steps
        handle_error_steps = [
            int(k.replace("tool_name_", ""))
            for k in trajectory.keys()
            if k.startswith("tool_name_") and trajectory[k] == SpecialTools.ERROR
        ]
        assert len(handle_error_steps) == 1
        assert handle_error_steps[0] == 1

        # Pattern 3: Get handle_error details
        handle_error_details = [
            {
                "step": i,
                "observation": trajectory[f"observation_{i}"],
                "tool_name": trajectory[f"tool_name_{i}"],
            }
            for i in handle_error_steps
        ]
        assert len(handle_error_details) == 1
        assert handle_error_details[0]["step"] == 1
        assert handle_error_details[0]["tool_name"] == SpecialTools.ERROR
        assert "error" in handle_error_details[0]["observation"].lower()


class TestAsyncMethods:
    """Test async methods and execution flow."""

    @pytest.mark.asyncio
    async def test_aexecute_tool_success(self):
        """Test async tool execution succeeds"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Act
        result = await agent._aexecute_tool("search_tool", {"query": "test"})

        # Assert
        assert result == "[search_tool(query='test')] Found: test"

    @pytest.mark.asyncio
    async def test_aexecute_tool_not_found(self):
        """Test async tool execution with non-existent tool raises ToolNotFoundError"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Act & Assert
        with pytest.raises(ToolNotFoundError) as exc_info:
            await agent._aexecute_tool("non_existent_tool", {"query": "test"})

        assert "non_existent_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_aforward_simple_execution(self):
        """Test aforward() method with mocked predictor for simple async execution"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found information about {query}"

        signature = "context, question -> response"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Mock the predictor to control outputs
        # First call: tool execution
        first_prediction = dspy.Prediction(
            reasoning="I should search for this",
            tool_name="search_tool",
            tool_args={"query": "test"},
            response="Searching...",
        )

        # Second call: finish
        second_prediction = dspy.Prediction(
            reasoning="Now I can answer",
            tool_name=SpecialTools.FINISH,
            tool_args={},
            response="Ready to answer",
        )

        # Third call: final answer
        final_prediction = dspy.Prediction(
            reasoning="Based on the search results",
            response="The answer based on search results",
        )

        # Create async mock
        mock_predictor = MagicMock()
        mock_predictor.acall = AsyncMock(side_effect=[first_prediction, second_prediction, final_prediction])

        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = await agent.acall(context="Some context", question="What is the answer?")

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "response")
        assert result.response == "The answer based on search results"
        assert hasattr(result, "history")
        assert len(result.history.messages) > 0
        assert hasattr(result, "steps")
        assert hasattr(result, "trajectory")

    @pytest.mark.asyncio
    async def test_aforward_max_steps_reached(self):
        """Test aforward() calls INTERRUPTED state when max steps reached"""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "input -> output"
        tools = [search_tool]
        max_steps = 2
        agent = ReActMachina(signature=signature, tools=tools, max_steps=max_steps, predictor_class=dspy.Predict)

        # Mock predictor to return tool calls for USER_QUERY, TOOL_RESULT states
        # but return final outputs for INTERRUPTED state
        tool_prediction = dspy.Prediction(
            tool_name="search_tool", tool_args={"query": "test"}, response="Searching more..."
        )
        interrupted_prediction = dspy.Prediction(output="Final answer based on gathered info")

        async def mock_side_effect(**kwargs):
            if kwargs.get(Fields.MACHINE_STATE) == MachineStates.INTERRUPTED:
                return interrupted_prediction
            return tool_prediction

        # Create async mock
        mock_predictor = MagicMock()
        mock_predictor.acall = AsyncMock(side_effect=mock_side_effect)

        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = await agent.acall(input="What is the answer?")

        # Assert
        assert isinstance(result, dspy.Prediction)
        assert result.steps == max_steps
        # Should have real output from INTERRUPTED state (not None)
        assert result.output == "Final answer based on gathered info"
        # Verify INTERRUPTED state was called (not FINISH)
        interrupted_calls = [
            call
            for call in mock_predictor.acall.call_args_list
            if call[1].get(Fields.MACHINE_STATE) == MachineStates.INTERRUPTED
        ]
        assert len(interrupted_calls) == 1
        # Verify INTERRUPTED was called with INTERRUPTION_INSTRUCTIONS
        assert interrupted_calls[0][1][Fields.INTERRUPTION_INSTRUCTIONS] == INTERRUPTION_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_aforward_with_error_in_loop(self):
        """Test aforward() re-raises unexpected errors during execution."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Mock predictor to raise an unexpected error (RuntimeError)
        mock_predictor = MagicMock()
        mock_predictor.acall = AsyncMock(side_effect=RuntimeError("LLM API error"))

        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act & Assert - unexpected errors should be re-raised
        with pytest.raises(RuntimeError) as exc_info:
            await agent.acall(question="What is the answer?")

        assert "LLM API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_aforward_with_existing_history(self):
        """Test aforward() continues from existing history."""

        # Arrange
        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        signature = "question -> answer"
        tools = [search_tool]
        agent = ReActMachina(signature=signature, tools=tools, max_steps=5, predictor_class=dspy.ChainOfThought)

        # Create existing history with one interaction
        existing_history = dspy.History(
            messages=[
                {
                    Fields.MACHINE_STATE: MachineStates.USER_QUERY,
                    Fields.TOOL_NAME: "search_tool",
                    Fields.TOOL_ARGS: {"query": "previous"},
                    Fields.RESPONSE: "Previous response",
                    "question": "Previous question",
                }
            ]
        )

        # Mock predictor
        first_prediction = dspy.Prediction(
            reasoning="Searching again",
            tool_name="search_tool",
            tool_args={"query": "new"},
            response="Searching...",
        )

        second_prediction = dspy.Prediction(
            reasoning="Now finishing", tool_name=SpecialTools.FINISH, tool_args={}, response="Done"
        )

        final_prediction = dspy.Prediction(reasoning="Final answer", answer="Combined answer")

        # Create async mock
        mock_predictor = MagicMock()
        mock_predictor.acall = AsyncMock(side_effect=[first_prediction, second_prediction, final_prediction])

        for state in agent.state_predictors:
            agent.state_predictors[state] = mock_predictor

        # Act
        result = await agent.acall(question="New question", history=existing_history)

        # Assert
        assert isinstance(result, dspy.Prediction)
        # History should contain both old and new messages
        assert len(result.history.messages) > 1
        # First message should be from existing history
        assert result.history.messages[0]["question"] == "Previous question"
