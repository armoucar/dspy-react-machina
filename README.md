# DSPy-ReAct-Machina

Alternative ReAct implementation for DSPy with full conversation history in a unified context.

## Installation

```bash
pip install dspy-react-machina
```

## Quick Start

```python
import dspy
from dspy_react_machina import ReActMachina

# Configure your LM
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define tools
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 72°F, sunny"

# Create agent
agent = ReActMachina("history: dspy.History, question -> answer", tools=[get_weather])

# Chat with persistent history
history = dspy.History(messages=[])

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit"]:
        break

    response = agent(question=user_input, history=history)
    print(f"Agent: {response.answer}\n")

    # Update history for next turn
    history = response.history
```

## Documentation

- [Examples](examples/) - Working examples including async and instrumentation
- [Testing Conventions](tests/TESTING_CONVENTIONS.md) - Testing guidelines

## Development

### Setup

```bash
uv sync
uv run pre-commit install
```

### Code Quality

```bash
uv run quality-check  # Run all checks: ruff + pyright
```

### Testing

```bash
uv run tests                    # Run tests
uv run tests-coverage           # Run tests with coverage
uv run tests-coverage --web     # Run tests with coverage and open HTML report
```
