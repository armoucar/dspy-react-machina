#!/usr/bin/env python3
"""
State Machine for ReActMachina

This module defines the state machine used by ReActMachina to manage the ReAct
(Reasoning and Acting) loop. The state machine ensures valid state transitions
and provides a clear interface for state management.
"""

from enum import StrEnum


class MachineStates(StrEnum):
    """Constants for machine state names used in ReActMachina.

    The ReActMachina module operates as a finite state machine with four distinct states:

    - USER_QUERY: Initial state when processing user input
    - TOOL_RESULT: State after a tool has been executed
    - INTERRUPTED: State for forced completion when max_steps is reached (interrupted execution)
    - FINISH: Terminal state for normal completion when agent calls finish tool (voluntary completion)

    Note: INTERRUPTED and FINISH both produce final outputs, but INTERRUPTED acknowledges the
    interruption whereas FINISH represents normal task completion.
    """

    USER_QUERY = "user_query"
    TOOL_RESULT = "tool_result"
    INTERRUPTED = "interrupted"
    FINISH = "finish"


# Valid state transitions define the allowed paths through the state machine
VALID_TRANSITIONS: dict[MachineStates, set[MachineStates]] = {
    MachineStates.USER_QUERY: {MachineStates.TOOL_RESULT},
    MachineStates.TOOL_RESULT: {MachineStates.TOOL_RESULT, MachineStates.INTERRUPTED, MachineStates.FINISH},
    MachineStates.INTERRUPTED: set(),  # Terminal state - no transitions allowed
    MachineStates.FINISH: set(),  # Terminal state - no transitions allowed
}


class StateMachine:
    """Simple finite state machine for managing ReAct state transitions.

    This class ensures that state transitions follow the defined rules,
    preventing invalid state changes and making the state flow explicit.

    Example:
        >>> sm = StateMachine(MachineStates.USER_QUERY)
        >>> sm.can_transition(MachineStates.TOOL_RESULT)
        True
        >>> sm.transition(MachineStates.TOOL_RESULT)
        >>> sm.current_state
        'tool_result'
    """

    def __init__(self, initial_state: MachineStates = MachineStates.USER_QUERY) -> None:
        """Initialize the state machine.

        Args:
            initial_state: The initial state (default: USER_QUERY)
        """
        self._current_state = initial_state

    @property
    def current_state(self) -> MachineStates:
        """Get the current state.

        Returns:
            The current machine state
        """
        return self._current_state

    def transition(self, new_state: MachineStates) -> None:
        """Transition to a new state.

        Args:
            new_state: The state to transition to

        Raises:
            ValueError: If the transition is invalid
        """
        if not self._can_transition(new_state):
            allowed = VALID_TRANSITIONS[self._current_state]
            raise ValueError(
                f"Invalid state transition from {self._current_state} to {new_state}. "
                f"Valid transitions: {allowed if allowed else 'none (terminal state)'}"
            )
        self._current_state = new_state

    def _can_transition(self, new_state: MachineStates) -> bool:
        """Check if transition to new_state is valid.

        Args:
            new_state: The state to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        return new_state in VALID_TRANSITIONS[self._current_state]
