# Testing Conventions

## Core Principles

1. **Test behavior, not implementation**
   - Assert on outcomes, not how code works internally
   - Ask: "What behavior does this test prove?"
   - Example: "it handles user queries by masking fields appropriately" ✓ vs "it formats a string correctly" ✗

2. **Favor real code over mocks**
   - Only mock dependencies that are hard to reproduce in tests (external APIs, DBs, LLMs, file systems)
   - Avoid asserting on mocks - no `assert_called_once()`, `assert_called_with()`, etc.

3. **Strong assertions over weak ones**
   - Weak assertions indicate noise:
     - `assert result is not None` → probably noise
     - `assert len(result) > 0` → probably noise
   - Strong assertions validate behavior:
     - `assert result == expected_complex_structure` → valuable
     - `assert "specific content" in result` → valuable

4. **Coverage is a side effect, not a goal**
   - Don't write tests just to hit branches
   - Focus on testing behaviors that matter to users
   - Branch coverage > statement coverage when both test real behavior

5. **Structure: Arrange-Act-Assert**
   - All tests should follow this pattern
   - Makes tests readable and maintainable

6. **Private method testing guidelines**
   - Simple utilities, getters, formatters → test through public API
   - Complex parsing, state management, error handling → can test directly
   - When in doubt, test through the public API
