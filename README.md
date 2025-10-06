# DSPy-ReAct-Machina

Alternative ReAct implementation for DSPy with full conversation history.

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