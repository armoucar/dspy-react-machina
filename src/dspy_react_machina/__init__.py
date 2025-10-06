"""
dspy-react-machina: Alternative ReAct implementation for DSPy

This package provides ReActMachina, a ReAct module that uses a state machine
architecture to maintain full conversation history including all tool interactions.

Key differences from standard DSPy ReAct:
- State machine control flow with 4 distinct states
- State-specific signatures for precise field management
- Single predictor module (dspy.Predict or dspy.ChainOfThought) operating on different signatures
- Custom adapter for unified system prompts with dynamic field masking
- Complete trajectory and conversation history preservation
"""

from dspy_react_machina.exceptions import ToolExecutionError, ToolNotFoundError
from dspy_react_machina.react_machina import ReActMachina

__all__ = [
    # Main class
    "ReActMachina",
    # Exceptions
    "ToolNotFoundError",
    "ToolExecutionError",
]
