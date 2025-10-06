#!/usr/bin/env python3
"""
A ReAct (Reasoning and Acting) module that maintains all interactions in a single,
ever-growing context. Every user query, tool call, tool result, and agent response
is stored in the conversation history, providing complete transparency and enabling
true multi-turn conversations.

The module uses a simple state machine (see state_machine.py) to control flow between
4 states:
- USER_QUERY: Initial state when processing user input
- TOOL_RESULT: State after a tool has been executed
- INTERRUPTED: State for forced completion when max_steps is reached (interrupted execution)
- FINISH: State for normal completion when agent calls finish tool (voluntary completion)

Each state has its own DSPy signature with appropriate input/output fields. A single
predictor type (dspy.Predict or dspy.ChainOfThought) operates on these different
signatures based on the current state.

The custom ReActMachinaAdapter maintains a unified system prompt across all LLM calls
while dynamically masking which fields appear in user messages based on the active state.
This prevents the LLM from generating irrelevant fields while maintaining full context.

All conversation history (user interactions, tool calls, observations, agent responses)
persists in a single dspy.History object that grows with each interaction.

This module is generic and should work with any signature, similar to the standard dspy.ReAct.

Key Features:
- Single unified context containing full interaction history
- Multi-turn conversation support
- Simple state-based flow control with clear distinction between voluntary and forced completion
- Consistent system prompts via custom adapter
- Complete trajectory preservation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import dspy
from dspy.signatures.signature import ensure_signature
from dspy_react_machina.adapter import ReActMachinaAdapter
from dspy_react_machina.conversation import (
    Fields,
    StateProcessingResult,
    create_format_error_response,
    create_interaction_record,
    extract_output_values,
    get_last_tool_result_message,
    update_history,
    ValidationResult,
)
from dspy_react_machina.exceptions import ToolExecutionError, ToolNotFoundError
from dspy_react_machina.signatures import (
    build_instructions,
    build_state_signatures,
    create_finish_tool,
    create_handle_error_tool,
)
from dspy_react_machina.state_machine import MachineStates, StateMachine
from dspy_react_machina.tools import format_tool_parameters, SpecialTools

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


# UI String Constants
INTERRUPTION_INSTRUCTIONS = (
    "Process interrupted — the last tool call was executed, but the maximum step limit was reached. "
    "Use the data gathered so far to compose a response. "
    "Inform the user that the process stopped early and that they can ask you to continue if they'd like."
)


class ReActMachina(dspy.Module):
    """ReAct module with state machine architecture and full conversation history."""

    # Initialization

    def __init__(
        self,
        signature: type[Signature],
        tools: list,
        max_steps: int = 10,
        predictor_class: type[dspy.Predict] | type[dspy.ChainOfThought] = dspy.ChainOfThought,
        _debug_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the ReActMachina module.

        Args:
            signature: The signature of the module, which defines the input and output
            tools: List of callable functions or dspy.Tool instances
            max_steps: Maximum number of agent steps before forcing completion
            predictor_class: The predictor class to use (dspy.Predict or dspy.ChainOfThought)
            _debug_config: Internal debug config dict with keys: fail_mode, fail_step (DO NOT USE IN PRODUCTION)

        Raises:
            ValueError: If max_steps is not positive, tools is empty, or predictor_class is invalid
        """
        # Validate inputs
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        if not tools:
            raise ValueError("At least one tool must be provided")

        if predictor_class not in (dspy.Predict, dspy.ChainOfThought):
            raise ValueError(
                f"predictor_class must be either dspy.Predict or dspy.ChainOfThought, got {predictor_class}"
            )

        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_steps = max_steps

        # Convert functions to Tool instances and create tool registry
        tools_list = [t if isinstance(t, dspy.Tool) else dspy.Tool(t) for t in tools]
        tools_dict: dict[str, dspy.Tool] = {tool.name: tool for tool in tools_list}  # type: ignore[misc]

        # Add special tools
        tools_dict[SpecialTools.FINISH] = create_finish_tool(self.signature)
        tools_dict[SpecialTools.ERROR] = create_handle_error_tool()
        self.tool_registry = tools_dict

        # Build state-specific signatures
        instr = build_instructions(self.signature, tools_dict)
        self.state_signatures = build_state_signatures(self.signature, tools_dict, instr, predictor_class)

        # Create the custom adapter with state-specific signatures
        self.custom_adapter = ReActMachinaAdapter(
            self.signature,
            self.state_signatures,
            predictor_class=predictor_class,
            _debug_config=_debug_config,
        )

        # Create predictors for each state
        self.state_predictors = {state: predictor_class(sig) for state, sig in self.state_signatures.items()}

    # Override Methods (Public API)

    def forward(self, **input_args) -> dspy.Prediction:
        """Process user input using ReAct with history.

        Args:
            **input_args: Arguments matching the original signature input fields, plus optional history

        Returns:
            dspy.Prediction with original signature outputs and updated history
        """
        # Pre-condition: validate history type if present
        if Fields.HISTORY in input_args and input_args[Fields.HISTORY] is not None:
            assert isinstance(input_args[Fields.HISTORY], dspy.History), (
                f"History must be dspy.History, got {type(input_args[Fields.HISTORY])}"
            )

        # Extract and prepare inputs (without mutating input_args)
        history = input_args.get(Fields.HISTORY) or dspy.History(messages=[])
        original_inputs = {
            k: v for k, v in input_args.items() if k in self.signature.input_fields and k != Fields.HISTORY
        }

        # Start the ReAct loop with user query
        state_machine = StateMachine(MachineStates.USER_QUERY)
        updated_history = dspy.History(messages=list(history.messages))
        tool_result = None
        trajectory = {}

        for step in range(self.max_steps):
            try:
                # Execute one step of the ReAct loop
                final_prediction, updated_history, tool_result = self._execute_step(
                    state_machine,
                    original_inputs,
                    tool_result,
                    updated_history,
                    trajectory,
                    step,
                )

                # If we got a final prediction, return it immediately
                if final_prediction is not None:
                    return final_prediction

            except (ToolNotFoundError, ToolExecutionError) as e:
                # Handle tool-related errors - these are expected and should be handled gracefully
                logger.warning(
                    f"Tool error in step {step}: {e}",
                    extra={
                        "step": step,
                        "state": state_machine.current_state,
                        "error_type": type(e).__name__,
                    },
                )
                # Mark error in trajectory
                trajectory[f"error_{step}"] = True
                trajectory[f"error_message_{step}"] = str(e)

                return self._handle_error(
                    e,
                    state_machine.current_state,
                    original_inputs,
                    tool_result,
                    updated_history,
                    step,
                    trajectory,
                )
            except (TypeError, ValueError, KeyError, AttributeError) as e:
                # Handle common errors that might occur during processing
                logger.error(
                    f"Processing error in step {step}: {e}",
                    extra={
                        "step": step,
                        "state": state_machine.current_state,
                        "error_type": type(e).__name__,
                    },
                )
                trajectory[f"error_{step}"] = True
                trajectory[f"error_message_{step}"] = str(e)

                return self._handle_error(
                    e,
                    state_machine.current_state,
                    original_inputs,
                    tool_result,
                    updated_history,
                    step,
                    trajectory,
                )
            except Exception as e:
                # Unexpected errors - log and re-raise
                logger.exception(
                    f"Unexpected error in step {step}: {e}",
                    extra={
                        "step": step,
                        "state": state_machine.current_state,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        # Max steps reached, force completion
        return self._handle_max_steps(updated_history, trajectory)

    async def _ahandle_max_steps(self, history: dspy.History, trajectory: dict[str, Any]) -> dspy.Prediction:
        """Handle max steps reached scenario asynchronously by triggering the INTERRUPTED state.

        This allows the LLM to synthesize a final answer based on the information
        gathered during the max_steps iterations, while acknowledging the interruption.
        """
        # Use max_steps - 1 as step number so final prediction shows steps = max_steps
        step = self.max_steps - 1

        # Call _aprocess_interrupted to let LLM generate final answer acknowledging interruption
        return await self._aprocess_interrupted(INTERRUPTION_INSTRUCTIONS, history, step, trajectory)

    async def aforward(self, **input_args) -> dspy.Prediction:
        """Process user input using ReAct with history asynchronously.

        Args:
            **input_args: Arguments matching the original signature input fields, plus optional history

        Returns:
            dspy.Prediction with original signature outputs and updated history
        """
        # Pre-condition: validate history type if present
        if Fields.HISTORY in input_args and input_args[Fields.HISTORY] is not None:
            assert isinstance(input_args[Fields.HISTORY], dspy.History), (
                f"History must be dspy.History, got {type(input_args[Fields.HISTORY])}"
            )

        # Extract and prepare inputs (without mutating input_args)
        history = input_args.get(Fields.HISTORY) or dspy.History(messages=[])
        original_inputs = {
            k: v for k, v in input_args.items() if k in self.signature.input_fields and k != Fields.HISTORY
        }

        # Start the ReAct loop with user query
        state_machine = StateMachine(MachineStates.USER_QUERY)
        updated_history = dspy.History(messages=list(history.messages))
        tool_result = None
        trajectory = {}

        for step in range(self.max_steps):
            try:
                # Execute one step of the ReAct loop
                final_prediction, updated_history, tool_result = await self._aexecute_step(
                    state_machine,
                    original_inputs,
                    tool_result,
                    updated_history,
                    trajectory,
                    step,
                )

                # If we got a final prediction, return it immediately
                if final_prediction is not None:
                    return final_prediction

            except (ToolNotFoundError, ToolExecutionError) as e:
                # Handle tool-related errors - these are expected and should be handled gracefully
                logger.warning(
                    f"Tool error in step {step}: {e}",
                    extra={
                        "step": step,
                        "state": state_machine.current_state,
                        "error_type": type(e).__name__,
                    },
                )
                # Mark error in trajectory
                trajectory[f"error_{step}"] = True
                trajectory[f"error_message_{step}"] = str(e)

                return self._handle_error(
                    e,
                    state_machine.current_state,
                    original_inputs,
                    tool_result,
                    updated_history,
                    step,
                    trajectory,
                )
            except (TypeError, ValueError, KeyError, AttributeError) as e:
                # Handle common errors that might occur during processing
                logger.error(
                    f"Processing error in step {step}: {e}",
                    extra={
                        "step": step,
                        "state": state_machine.current_state,
                        "error_type": type(e).__name__,
                    },
                )
                trajectory[f"error_{step}"] = True
                trajectory[f"error_message_{step}"] = str(e)

                return self._handle_error(
                    e,
                    state_machine.current_state,
                    original_inputs,
                    tool_result,
                    updated_history,
                    step,
                    trajectory,
                )
            except Exception as e:
                # Unexpected errors - log and re-raise
                logger.exception(
                    f"Unexpected error in step {step}: {e}",
                    extra={
                        "step": step,
                        "state": state_machine.current_state,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        # Max steps reached, force completion
        return await self._ahandle_max_steps(updated_history, trajectory)

    # Properties

    @property
    def _first_input_field(self) -> str:
        """Get the first input field name from the original signature."""
        return next(iter(self.signature.input_fields.keys()))

    @property
    def _has_reasoning(self) -> bool:
        """Check if reasoning field is in state signatures outputs."""
        # All state signatures have the same reasoning field presence (based on predictor_class)
        # Check any state signature (e.g., user_query)
        return Fields.REASONING in self.state_signatures[MachineStates.USER_QUERY].output_fields

    # Core State Machine Logic

    def _execute_step(
        self,
        state_machine: StateMachine,
        original_inputs: dict[str, Any],
        tool_result: str | None,
        updated_history: dspy.History,
        trajectory: dict[str, Any],
        step: int,
    ) -> tuple[dspy.Prediction | None, dspy.History, str | None]:
        """Execute a single ReAct step.

        Args:
            state_machine: Current state machine instance
            original_inputs: Original input arguments
            tool_result: Result from previous tool execution
            updated_history: Current conversation history
            trajectory: Current trajectory dictionary
            step: Current step number

        Returns:
            Tuple of (final_prediction or None, updated_history, next_tool_result or None)
            If final_prediction is not None, the loop should terminate
        """
        # Pre-conditions
        assert step >= 0, f"Invalid step: {step}"
        assert state_machine is not None, "State machine cannot be None"

        # Notify adapter of current step (for debug failure injection)
        self.custom_adapter._debug_set_current_step(step)

        # Process current state
        logger.debug(f"Processing state {state_machine.current_state} at step {step}")
        processing_result = self._process_state(
            state_machine.current_state,
            original_inputs,
            tool_result,
            updated_history,
            trajectory,
            step,
        )
        prediction = processing_result.prediction
        updated_history = processing_result.updated_history

        # Execute tool and get observation
        logger.debug(f"Executing tool {prediction.tool_name} with args {prediction.tool_args}")
        observation = self._execute_tool(prediction.tool_name, prediction.tool_args, trajectory, step)

        # Build trajectory for this step
        trajectory[f"state_{step}"] = state_machine.current_state
        if hasattr(prediction, Fields.REASONING):
            trajectory[f"reasoning_{step}"] = prediction.reasoning
        trajectory[f"tool_name_{step}"] = prediction.tool_name
        trajectory[f"tool_args_{step}"] = prediction.tool_args
        trajectory[f"observation_{step}"] = observation
        trajectory[f"error_{step}"] = False

        # Handle state transitions
        current_state = state_machine.current_state

        if current_state == MachineStates.USER_QUERY:
            # From USER_QUERY, always transition to TOOL_RESULT
            state_machine.transition(MachineStates.TOOL_RESULT)
            return (None, updated_history, observation)
        elif current_state == MachineStates.TOOL_RESULT:
            # From TOOL_RESULT, check if we should finish or continue
            if prediction.tool_name == SpecialTools.FINISH:
                state_machine.transition(MachineStates.FINISH)
                final_prediction = self._process_finish(observation, updated_history, step, trajectory)
                return (final_prediction, updated_history, None)
            else:
                # Continue with next tool
                state_machine.transition(MachineStates.TOOL_RESULT)
                return (None, updated_history, observation)
        else:
            # This should not happen - safety check
            raise ValueError(f"Unexpected state in _execute_step: {current_state}")

    async def _aexecute_step(
        self,
        state_machine: StateMachine,
        original_inputs: dict[str, Any],
        tool_result: str | None,
        updated_history: dspy.History,
        trajectory: dict[str, Any],
        step: int,
    ) -> tuple[dspy.Prediction | None, dspy.History, str | None]:
        """Execute a single ReAct step asynchronously.

        Args:
            state_machine: Current state machine instance
            original_inputs: Original input arguments
            tool_result: Result from previous tool execution
            updated_history: Current conversation history
            trajectory: Current trajectory dictionary
            step: Current step number

        Returns:
            Tuple of (final_prediction or None, updated_history, next_tool_result or None)
            If final_prediction is not None, the loop should terminate
        """
        # Pre-conditions
        assert step >= 0, f"Invalid step: {step}"
        assert state_machine is not None, "State machine cannot be None"

        # Notify adapter of current step (for debug failure injection)
        self.custom_adapter._debug_set_current_step(step)

        # Process current state
        logger.debug(f"Processing state {state_machine.current_state} at step {step}")
        processing_result = await self._aprocess_state(
            state_machine.current_state,
            original_inputs,
            tool_result,
            updated_history,
            trajectory,
            step,
        )
        prediction = processing_result.prediction
        updated_history = processing_result.updated_history

        # Execute tool and get observation
        logger.debug(f"Executing tool {prediction.tool_name} with args {prediction.tool_args}")
        observation = await self._aexecute_tool(prediction.tool_name, prediction.tool_args, trajectory, step)

        # Build trajectory for this step
        trajectory[f"state_{step}"] = state_machine.current_state
        if hasattr(prediction, Fields.REASONING):
            trajectory[f"reasoning_{step}"] = prediction.reasoning
        trajectory[f"tool_name_{step}"] = prediction.tool_name
        trajectory[f"tool_args_{step}"] = prediction.tool_args
        trajectory[f"observation_{step}"] = observation
        trajectory[f"error_{step}"] = False

        # Handle state transitions
        current_state = state_machine.current_state

        if current_state == MachineStates.USER_QUERY:
            # From USER_QUERY, always transition to TOOL_RESULT
            state_machine.transition(MachineStates.TOOL_RESULT)
            return (None, updated_history, observation)
        elif current_state == MachineStates.TOOL_RESULT:
            # From TOOL_RESULT, check if we should finish or continue
            if prediction.tool_name == SpecialTools.FINISH:
                state_machine.transition(MachineStates.FINISH)
                final_prediction = await self._aprocess_finish(observation, updated_history, step, trajectory)
                return (final_prediction, updated_history, None)
            else:
                # Continue with next tool
                state_machine.transition(MachineStates.TOOL_RESULT)
                return (None, updated_history, observation)
        else:
            # This should not happen - safety check
            raise ValueError(f"Unexpected state in _aexecute_step: {current_state}")

    def _process_state(
        self,
        state: str,
        original_inputs: dict[str, Any],
        tool_result: str | None,
        history: dspy.History,
        trajectory: dict[str, Any],
        step: int,
    ) -> StateProcessingResult:
        """Process a single state transition in the ReAct loop.

        Args:
            state: Current machine state
            original_inputs: Original input arguments
            tool_result: Result from previous tool execution
            history: Conversation history
            trajectory: Current trajectory dictionary
            step: Current step number

        Returns:
            StateProcessingResult with prediction and updated history
        """
        # Pre-conditions
        assert state in MachineStates, f"Invalid state: {state}"
        assert step >= 0, f"Invalid step: {step}"
        assert state in self.state_predictors, f"No predictor for state: {state}"

        # Prepare inputs for the unified predictor based on machine state
        predictor_inputs: dict[str, Any] = {Fields.MACHINE_STATE: state, Fields.HISTORY: history}

        if state == MachineStates.USER_QUERY:
            predictor_inputs.update(original_inputs)
        else:
            predictor_inputs[Fields.TOOL_RESULT] = tool_result

        # Call state-specific predictor with custom adapter
        with dspy.settings.context(adapter=self.custom_adapter):
            prediction = self.state_predictors[state](**predictor_inputs)

        # Validate prediction has required fields
        validation_result = self._validate_react_prediction(prediction, state)

        if not validation_result.is_valid:
            logger.warning(
                f"Prediction missing required fields: {validation_result.missing_fields}. "
                f"Using fallback error response."
            )
            # Create fallback prediction with context from trajectory
            fallback_fields = create_format_error_response(trajectory, step, self._has_reasoning)
            prediction = dspy.Prediction(**fallback_fields)

        # Create interaction record and update history
        interaction_record = create_interaction_record(
            prediction, state, self._first_input_field, original_inputs, tool_result
        )
        updated_history = update_history(history, interaction_record)

        # Post-condition: history should grow after processing state
        assert len(updated_history.messages) > len(history.messages), "History should grow after processing state"

        return StateProcessingResult(prediction=prediction, updated_history=updated_history)

    async def _aprocess_state(
        self,
        state: str,
        original_inputs: dict[str, Any],
        tool_result: str | None,
        history: dspy.History,
        trajectory: dict[str, Any],
        step: int,
    ) -> StateProcessingResult:
        """Process a single state transition in the ReAct loop asynchronously.

        Args:
            state: Current machine state
            original_inputs: Original input arguments
            tool_result: Result from previous tool execution
            history: Conversation history
            trajectory: Current trajectory dictionary
            step: Current step number

        Returns:
            StateProcessingResult with prediction and updated history
        """
        # Pre-conditions
        assert state in MachineStates, f"Invalid state: {state}"
        assert step >= 0, f"Invalid step: {step}"
        assert state in self.state_predictors, f"No predictor for state: {state}"

        # Prepare inputs for the unified predictor based on machine state
        predictor_inputs: dict[str, Any] = {Fields.MACHINE_STATE: state, Fields.HISTORY: history}

        if state == MachineStates.USER_QUERY:
            predictor_inputs.update(original_inputs)
        else:
            predictor_inputs[Fields.TOOL_RESULT] = tool_result

        # Call state-specific predictor with custom adapter
        with dspy.settings.context(adapter=self.custom_adapter):
            prediction = await self.state_predictors[state].acall(**predictor_inputs)

        # Validate prediction has required fields
        validation_result = self._validate_react_prediction(prediction, state)

        if not validation_result.is_valid:
            logger.warning(
                f"Prediction missing required fields: {validation_result.missing_fields}. Using fallback error response."
            )
            # Create fallback prediction with context from trajectory
            fallback_fields = create_format_error_response(trajectory, step, self._has_reasoning)
            prediction = dspy.Prediction(**fallback_fields)

        # Create interaction record and update history
        interaction_record = create_interaction_record(
            prediction, state, self._first_input_field, original_inputs, tool_result
        )
        updated_history = update_history(history, interaction_record)

        # Post-condition: history should grow after processing state
        assert len(updated_history.messages) > len(history.messages), "History should grow after processing state"

        return StateProcessingResult(prediction=prediction, updated_history=updated_history)

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        trajectory: dict[str, Any] | None = None,
        step: int = 0,
    ) -> str:
        """Execute a tool and return the formatted observation.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool
            trajectory: Current trajectory dictionary (needed for error handling)
            step: Current step number (needed for error handling)

        Returns:
            Formatted observation string

        Raises:
            ToolNotFoundError: If tool is not found in registry
            ToolExecutionError: If tool execution fails
        """
        # Pre-conditions
        assert tool_name, "Tool name cannot be empty"
        assert isinstance(tool_args, dict), f"Tool args must be dict, got {type(tool_args)}"

        # Handle special ERROR control signal (handle_error tool)
        # This is used internally for error recovery when format validation fails
        if tool_name == SpecialTools.ERROR:
            tool = self.tool_registry[tool_name]
            base_msg = tool()
            last_result = get_last_tool_result_message(trajectory or {}, step)
            full_msg = f"{base_msg}\n{last_result}" if last_result else base_msg
            return f"[{tool_name}()] {full_msg}"

        if tool_name not in self.tool_registry:
            available_tools = [t for t in self.tool_registry.keys() if t not in (SpecialTools.FINISH,)]
            raise ToolNotFoundError(tool_name, available_tools)

        try:
            tool = self.tool_registry[tool_name]
            result = tool(**tool_args)
            return f"[{tool_name}({format_tool_parameters(tool_args)})] {result}"
        except ToolNotFoundError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Wrap other exceptions in ToolExecutionError
            raise ToolExecutionError(tool_name, tool_args, e) from e

    async def _aexecute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        trajectory: dict[str, Any] | None = None,
        step: int = 0,
    ) -> str:
        """Execute a tool asynchronously and return the formatted observation.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool
            trajectory: Current trajectory dictionary (needed for error handling)
            step: Current step number (needed for error handling)

        Returns:
            Formatted observation string

        Raises:
            ToolNotFoundError: If tool is not found in registry
            ToolExecutionError: If tool execution fails
        """
        # Pre-conditions
        assert tool_name, "Tool name cannot be empty"
        assert isinstance(tool_args, dict), f"Tool args must be dict, got {type(tool_args)}"

        # Handle special ERROR control signal (handle_error tool)
        # This is used internally for error recovery when format validation fails
        if tool_name == SpecialTools.ERROR:
            tool = self.tool_registry[tool_name]
            base_msg = await tool.acall()
            last_result = get_last_tool_result_message(trajectory or {}, step)
            full_msg = f"{base_msg}\n{last_result}" if last_result else base_msg
            return f"[{tool_name}()] {full_msg}"

        if tool_name not in self.tool_registry:
            available_tools = [t for t in self.tool_registry.keys() if t not in (SpecialTools.FINISH,)]
            raise ToolNotFoundError(tool_name, available_tools)

        try:
            tool = self.tool_registry[tool_name]
            result = await tool.acall(**tool_args)
            return f"[{tool_name}({format_tool_parameters(tool_args)})] {result}"
        except ToolNotFoundError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Wrap other exceptions in ToolExecutionError
            raise ToolExecutionError(tool_name, tool_args, e) from e

    def _process_finish(
        self,
        observation: str,
        history: dspy.History,
        step: int,
        trajectory: dict[str, Any],
    ) -> dspy.Prediction:
        """Process the finish state and return final prediction."""
        finish_inputs = {
            Fields.TOOL_RESULT: observation,
            Fields.MACHINE_STATE: MachineStates.FINISH,
            Fields.HISTORY: history,
        }

        with dspy.settings.context(adapter=self.custom_adapter):
            final_prediction = self.state_predictors[MachineStates.FINISH](**finish_inputs)

        # Extract original signature outputs
        output_values = extract_output_values(final_prediction, self.signature)

        # Create final interaction record
        final_interaction_record: dict[str, Any] = {
            Fields.TOOL_RESULT: observation,
            Fields.MACHINE_STATE: MachineStates.FINISH,
        }

        # Add reasoning if present (only with ChainOfThought)
        if hasattr(final_prediction, Fields.REASONING):
            final_interaction_record[Fields.REASONING] = final_prediction.reasoning

        # Add final outputs to history record
        final_interaction_record.update(extract_output_values(final_prediction, self.signature, only_existing=True))

        final_history = update_history(history, final_interaction_record)

        return dspy.Prediction(
            **output_values,
            history=final_history,
            steps=step + 1,
            trajectory=trajectory,
        )

    def _process_interrupted(
        self,
        interruption_instructions: str,
        history: dspy.History,
        step: int,
        trajectory: dict[str, Any],
    ) -> dspy.Prediction:
        """Process the interrupted state for max_steps interruption and return final prediction.

        This state is used when max_steps is reached, allowing the LLM to synthesize
        a final answer based on gathered information while acknowledging the interruption.
        """
        # Get the last tool result from trajectory
        last_tool_result = trajectory.get(f"observation_{step}", "")

        interrupted_inputs = {
            Fields.TOOL_RESULT: last_tool_result,
            Fields.INTERRUPTION_INSTRUCTIONS: interruption_instructions,
            Fields.MACHINE_STATE: MachineStates.INTERRUPTED,
            Fields.HISTORY: history,
        }

        with dspy.settings.context(adapter=self.custom_adapter):
            final_prediction = self.state_predictors[MachineStates.INTERRUPTED](**interrupted_inputs)

        # Extract original signature outputs
        output_values = extract_output_values(final_prediction, self.signature)

        # Create final interaction record
        final_interaction_record: dict[str, Any] = {
            Fields.TOOL_RESULT: last_tool_result,
            Fields.INTERRUPTION_INSTRUCTIONS: interruption_instructions,
            Fields.MACHINE_STATE: MachineStates.INTERRUPTED,
        }

        # Add reasoning if present (only with ChainOfThought)
        if hasattr(final_prediction, Fields.REASONING):
            final_interaction_record[Fields.REASONING] = final_prediction.reasoning

        # Add final outputs to history record
        final_interaction_record.update(extract_output_values(final_prediction, self.signature, only_existing=True))

        final_history = update_history(history, final_interaction_record)

        return dspy.Prediction(
            **output_values,
            history=final_history,
            steps=step + 1,
            trajectory=trajectory,
        )

    async def _aprocess_finish(
        self,
        observation: str,
        history: dspy.History,
        step: int,
        trajectory: dict[str, Any],
    ) -> dspy.Prediction:
        """Process the finish state asynchronously and return final prediction."""
        finish_inputs = {
            Fields.TOOL_RESULT: observation,
            Fields.MACHINE_STATE: MachineStates.FINISH,
            Fields.HISTORY: history,
        }

        with dspy.settings.context(adapter=self.custom_adapter):
            final_prediction = await self.state_predictors[MachineStates.FINISH].acall(**finish_inputs)

        # Extract original signature outputs
        output_values = extract_output_values(final_prediction, self.signature)

        # Create final interaction record
        final_interaction_record: dict[str, Any] = {
            Fields.TOOL_RESULT: observation,
            Fields.MACHINE_STATE: MachineStates.FINISH,
        }

        # Add reasoning if present (only with ChainOfThought)
        if hasattr(final_prediction, Fields.REASONING):
            final_interaction_record[Fields.REASONING] = final_prediction.reasoning

        # Add final outputs to history record
        final_interaction_record.update(extract_output_values(final_prediction, self.signature, only_existing=True))

        final_history = update_history(history, final_interaction_record)

        return dspy.Prediction(
            **output_values,
            history=final_history,
            steps=step + 1,
            trajectory=trajectory,
        )

    async def _aprocess_interrupted(
        self,
        interruption_instructions: str,
        history: dspy.History,
        step: int,
        trajectory: dict[str, Any],
    ) -> dspy.Prediction:
        """Process the interrupted state asynchronously for max_steps interruption and return final prediction.

        This state is used when max_steps is reached, allowing the LLM to synthesize
        a final answer based on gathered information while acknowledging the interruption.
        """
        # Get the last tool result from trajectory
        last_tool_result = trajectory.get(f"observation_{step}", "")

        interrupted_inputs = {
            Fields.TOOL_RESULT: last_tool_result,
            Fields.INTERRUPTION_INSTRUCTIONS: interruption_instructions,
            Fields.MACHINE_STATE: MachineStates.INTERRUPTED,
            Fields.HISTORY: history,
        }

        with dspy.settings.context(adapter=self.custom_adapter):
            final_prediction = await self.state_predictors[MachineStates.INTERRUPTED].acall(**interrupted_inputs)

        # Extract original signature outputs
        output_values = extract_output_values(final_prediction, self.signature)

        # Create final interaction record
        final_interaction_record: dict[str, Any] = {
            Fields.TOOL_RESULT: last_tool_result,
            Fields.INTERRUPTION_INSTRUCTIONS: interruption_instructions,
            Fields.MACHINE_STATE: MachineStates.INTERRUPTED,
        }

        # Add reasoning if present (only with ChainOfThought)
        if hasattr(final_prediction, Fields.REASONING):
            final_interaction_record[Fields.REASONING] = final_prediction.reasoning

        # Add final outputs to history record
        final_interaction_record.update(extract_output_values(final_prediction, self.signature, only_existing=True))

        final_history = update_history(history, final_interaction_record)

        return dspy.Prediction(
            **output_values,
            history=final_history,
            steps=step + 1,
            trajectory=trajectory,
        )

    def _handle_error(
        self,
        error: Exception,
        state: str,
        original_inputs: dict[str, Any],
        tool_result: str | None,
        history: dspy.History,
        step: int,
        trajectory: dict[str, Any],
    ) -> dspy.Prediction:
        """Handle errors during execution."""
        error_record: dict[str, Any] = {
            Fields.MACHINE_STATE: state,
            Fields.TOOL_NAME: SpecialTools.ERROR,
            Fields.TOOL_ARGS: {"error_message": str(error)},
            Fields.RESPONSE: f"I encountered an error: {error!s}",
        }

        # Add reasoning if using ChainOfThought (check if reasoning field is in unified signature)
        if self._has_reasoning:
            error_record[Fields.REASONING] = f"An error occurred during execution: {error!s}"

        # Add appropriate input fields for error record
        if state == MachineStates.USER_QUERY:
            error_record[self._first_input_field] = original_inputs[self._first_input_field]
        else:
            error_record[Fields.TOOL_RESULT] = tool_result

        updated_history = update_history(history, error_record)

        # Return with empty original outputs on error
        error_outputs = dict.fromkeys(self.signature.output_fields)
        return dspy.Prediction(**error_outputs, history=updated_history, steps=step + 1, trajectory=trajectory)

    def _handle_max_steps(self, history: dspy.History, trajectory: dict[str, Any]) -> dspy.Prediction:
        """Handle max steps reached scenario by triggering the INTERRUPTED state.

        This allows the LLM to synthesize a final answer based on the information
        gathered during the max_steps iterations, while acknowledging the interruption.
        """
        # Use max_steps - 1 as step number so final prediction shows steps = max_steps
        step = self.max_steps - 1

        # Call _process_interrupted to let LLM generate final answer acknowledging interruption
        return self._process_interrupted(INTERRUPTION_INSTRUCTIONS, history, step, trajectory)

    # Helper Methods

    def _get_input_fields_for_state(
        self,
        state: str,
        original_inputs: dict[str, Any],
        tool_result: str | None,
    ) -> dict[str, Any]:
        """Get the input fields that should be added to a record based on machine state.

        This is a helper method for tests - the actual logic is in create_interaction_record.

        Args:
            state: Current machine state
            original_inputs: Original user inputs
            tool_result: Tool execution result

        Returns:
            Dictionary containing the input field(s) to add to the record
        """
        if state == MachineStates.USER_QUERY:
            return {self._first_input_field: original_inputs[self._first_input_field]}
        else:
            return {Fields.TOOL_RESULT: tool_result}

    # Validation

    def _validate_react_prediction(self, prediction: dspy.Prediction, state: str) -> ValidationResult:
        """Validate that prediction has required ReAct output fields.

        Args:
            prediction: The prediction to validate
            state: Current machine state

        Returns:
            ValidationResult with is_valid flag and list of missing fields
        """
        # For FINISH and INTERRUPTED states, different validation rules apply
        # These states produce final outputs, not ReAct action fields
        if state in (MachineStates.FINISH, MachineStates.INTERRUPTED):
            return ValidationResult(is_valid=True, missing_fields=[])

        # Check for required ReAct output fields
        required_fields = [Fields.TOOL_NAME, Fields.TOOL_ARGS, Fields.RESPONSE]
        missing = [str(f) for f in required_fields if not hasattr(prediction, f) or getattr(prediction, f) is None]
        return ValidationResult(is_valid=len(missing) == 0, missing_fields=missing)
