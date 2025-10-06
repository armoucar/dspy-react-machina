# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSPy-ReAct-Machina is an alternative ReAct implementation for DSPy that maintains full conversation history in a unified context. Unlike standard DSPy ReAct, this implementation uses a state machine approach to enable true multi-turn conversations with complete transparency.

## Development

For Commands, Setup, Code Quality, Testing, and Running single tests, see @README.md

## Architecture

### Core Components

**Main Module (`react_machina.py`)**
Orchestrates the state machine loop. Single predictor type operates on different signatures based on current state.

**Adapter (`adapter.py`)**
`ReActMachinaAdapter` maintains unified system prompts across all states while dynamically masking fields in user messages based on current state. Prevents LLM from generating irrelevant fields.

**State Machine (`state_machine.py`)**
Four states: `USER_QUERY`, `TOOL_RESULT`, `CONTINUE`, `FINISH`. Valid transitions enforce flow control. See `VALID_TRANSITIONS` for allowed paths.

**Signatures (`signatures.py`)**
Each state has its own DSPy signature with appropriate input/output fields. Built using `build_instructions()` and `create_state_field_specs()`.

**Conversation (`conversation.py`)**
Manages interaction records and history updates. All user queries, tool calls, and responses persist in a single `dspy.History` object.

### Key Design Principles

1. **Single unified context** - All interactions stored in one ever-growing conversation history
2. **State-based flow** - Simple finite state machine controls transition between query → tool → continue → finish
3. **Field masking** - System prompt stays constant; user messages only show state-relevant fields
4. **Trajectory preservation** - Complete history of reasoning, tool calls, and results maintained

## Testing Conventions

Follow principles in `tests/TESTING_CONVENTIONS.md`:

## Type Checking

Pyright is configured to check `src/` only. Tests and examples are ignored. Mode is "standard".
