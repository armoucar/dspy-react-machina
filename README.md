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
agent = ReActMachina(tools=[get_weather])

# Chat with the agent
result = agent("What's the weather in Paris?")
print(result.answer)
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
