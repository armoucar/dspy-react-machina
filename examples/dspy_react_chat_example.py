#!/usr/bin/env python3
"""
DSPy ReAct Weather Agent Example

A simple weather agent demonstrating standard DSPy ReAct with tool usage.
This example shows how dspy.ReAct works out-of-the-box, in contrast to ReActMachina.

Key differences from ReActMachina:
- Manual conversation history management (must manually append to history after each turn)
- No control over internal predictors (uses fixed dspy.Predict and dspy.ChainOfThought)
- Different trajectory structure (thought/tool_name/tool_args/observation vs state-based)
- Uses max_iters instead of max_steps
- History must be explicitly added to signature as input field

Usage:
    Interactive mode: uv run examples/dspy_react_chat_example.py
    One-turn mode: uv run examples/dspy_react_chat_example.py --query "What's the weather in Paris?"
    Async mode: uv run examples/dspy_react_chat_example.py --async
    Inspect history: uv run examples/dspy_react_chat_example.py --inspect-history
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from io import StringIO
from pathlib import Path

import dspy
from dspy import inspect_history

# Try to import instrumentation if available
try:
    from instrumentation import configure_instrumentation
except ImportError:
    # Optional instrumentation module not available
    def configure_instrumentation(project_name="dspy-react-machina", endpoint="http://localhost:6006/v1/traces"):  # noqa: ARG001
        pass


logger = logging.getLogger(__name__)


def save_inspect_history_to_file(prefix: str = "dspy_react") -> str:
    """
    Save the output of dspy.inspect_history() to a file.

    Args:
        prefix: Prefix for the filename (e.g., 'react_machina' or 'dspy_react')

    Returns:
        Path to the saved file
    """
    import re
    import sys

    # Capture inspect_history output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        inspect_history()
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Strip ANSI color codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    output = ansi_escape.sub("", output)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    examples_dir = Path(__file__).parent
    filename = examples_dir / f"{prefix}_history_{timestamp}.txt"

    # Save to file
    filename.write_text(output)

    return str(filename)


# Weather tool functions
def get_weather(city: str = "current location") -> str:
    """Get the current weather for a specified city."""
    # Mock weather data
    weather_data = {
        "paris": "Sunny, 18°C, light breeze",
        "london": "Cloudy, 12°C, moderate rain",
        "tokyo": "Partly cloudy, 22°C, calm",
        "new york": "Clear, 15°C, windy",
        "current location": "Sunny, 20°C, perfect weather",
    }

    city_lower = city.lower()
    weather = weather_data.get(city_lower, f"Weather data unavailable for {city}")
    return weather


def list_weather_cities() -> str:
    """List all cities available for weather checking."""
    cities = ["Paris", "London", "Tokyo", "New York", "current location"]
    return f"Available cities for weather: {', '.join(cities)}"


async def async_main(args, agent):
    """Async main function for async mode execution."""
    # Handle one-turn query
    if args.query:
        history = dspy.History(messages=[])
        try:
            response = await agent.acall(question=args.query, history=history)
            print(response.answer)

            # If inspect-history flag is also provided, show history after query
            if args.inspect_history:
                print("\n" + "=" * 50 + " INTERACTION HISTORY " + "=" * 50)
                inspect_history()
                saved_path = save_inspect_history_to_file("dspy_react")
                print(f"\n💾 History saved to: {saved_path}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Handle inspect history request (standalone)
    if args.inspect_history:
        inspect_history()
        saved_path = save_inspect_history_to_file("dspy_react")
        print(f"\n💾 History saved to: {saved_path}")
        return

    # Interactive mode
    history = dspy.History(messages=[])
    last_response = None

    print("🤖 DSPy ReAct Weather Agent (Async Mode)")
    print("=" * 60)
    print("I'm a weather agent that can check weather and list available cities.")
    print("Available tools: get_weather, list_weather_cities")
    print(
        "Commands: '/quit' to exit, '/tools' to list tools, '/inspect_history' to view full interaction history, '/trajectory' to view last trajectory"
    )
    print("Note: Conversation history is maintained across turns.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["/quit", "quit", "exit"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() in ["/tools", "tools"]:
                print("\n" + "=" * 60 + " AVAILABLE TOOLS " + "=" * 60)
                for tool in agent.tools.values():
                    print(f"\n• {tool.name}")
                    print(f"  Description: {tool.desc}")
                    if tool.args:
                        args_str = ", ".join(f"{k}: {v}" for k, v in tool.args.items())
                        print(f"  Arguments: {args_str}")
                    else:
                        print("  Arguments: None")
                print("\n" + "=" * 120 + "\n")
                continue

            if user_input.lower() in ["/inspect_history", "inspect_history"]:
                print("\n" + "=" * 60 + " INTERACTION HISTORY " + "=" * 60)
                inspect_history()
                saved_path = save_inspect_history_to_file("dspy_react")
                print(f"\n💾 History saved to: {saved_path}")
                print("=" * 120 + "\n")
                continue

            if user_input.lower() in ["/trajectory", "trajectory"]:
                print("\n" + "=" * 60 + " LAST TRAJECTORY " + "=" * 60)
                if last_response and hasattr(last_response, "trajectory"):
                    trajectory = last_response.trajectory
                    if trajectory:
                        # Determine number of steps from trajectory keys
                        steps = len([k for k in trajectory.keys() if k.startswith("thought_")])

                        for i in range(steps):
                            print(f"\n--- Iteration {i} ---")
                            if f"thought_{i}" in trajectory:
                                print(f"Thought: {trajectory[f'thought_{i}']}")
                            if f"tool_name_{i}" in trajectory:
                                print(f"Tool: {trajectory[f'tool_name_{i}']}")
                            if f"tool_args_{i}" in trajectory:
                                print(f"Args: {json.dumps(trajectory[f'tool_args_{i}'], indent=2)}")
                            if f"observation_{i}" in trajectory:
                                print(f"Observation: {trajectory[f'observation_{i}']}")
                    else:
                        print("Trajectory is empty.")
                else:
                    print("No trajectory available yet. Make a query first.")
                print("=" * 120 + "\n")
                continue

            if not user_input:
                continue

            # Process user input with conversation history (async)
            response = await agent.acall(question=user_input, history=history)

            print(f"\nAgent: {response.answer}")

            # Count steps from trajectory
            if hasattr(response, "trajectory") and response.trajectory:
                steps = len([k for k in response.trajectory.keys() if k.startswith("thought_")])
                print(f"(Completed in {steps} steps)\n")
            else:
                print()

            # Store last response for trajectory inspection
            last_response = response

            # Manually update conversation history
            history.messages.append({"question": user_input, **response})

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="DSPy ReAct Weather Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query for one-turn response")
    parser.add_argument("--inspect-history", "-i", action="store_true", help="Inspect the interaction history")
    parser.add_argument(
        "--async",
        "-a",
        dest="use_async",
        action="store_true",
        help="Use async mode for agent execution",
    )
    parser.add_argument(
        "--max-iters",
        "-m",
        type=int,
        default=10,
        help="Maximum number of agent iterations before forcing completion (default: 10)",
    )
    args = parser.parse_args()

    configure_instrumentation("dspy-react")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return

    lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0,
        cache=False,
    )

    dspy.settings.configure(lm=lm)

    # Initialize the standard DSPy ReAct agent with weather tools
    # Note: We add 'history' as an input field to maintain conversation context
    tools = [get_weather, list_weather_cities]
    signature = "question, history -> answer"

    agent = dspy.ReAct(
        signature,
        tools,
        max_iters=args.max_iters,
    )

    # Route to async or sync execution
    if args.use_async:
        asyncio.run(async_main(args, agent))
        return

    # Handle one-turn query (sync mode)
    if args.query:
        history = dspy.History(messages=[])
        try:
            response = agent(question=args.query, history=history)
            print(response.answer)

            # If inspect-history flag is also provided, show history after query
            if args.inspect_history:
                print("\n" + "=" * 50 + " INTERACTION HISTORY " + "=" * 50)
                inspect_history()
                saved_path = save_inspect_history_to_file("dspy_react")
                print(f"\n💾 History saved to: {saved_path}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Handle inspect history request (standalone)
    if args.inspect_history:
        inspect_history()
        saved_path = save_inspect_history_to_file("dspy_react")
        print(f"\n💾 History saved to: {saved_path}")
        return

    # Interactive mode
    history = dspy.History(messages=[])
    last_response = None

    print("🤖 DSPy ReAct Weather Agent")
    print("=" * 60)
    print("I'm a weather agent that can check weather and list available cities.")
    print("Available tools: get_weather, list_weather_cities")
    print(
        "Commands: '/quit' to exit, '/tools' to list tools, '/inspect_history' to view full interaction history, '/trajectory' to view last trajectory"
    )
    print("Note: Conversation history is maintained across turns.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["/quit", "quit", "exit"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() in ["/tools", "tools"]:
                print("\n" + "=" * 60 + " AVAILABLE TOOLS " + "=" * 60)
                for tool in agent.tools.values():
                    print(f"\n• {tool.name}")
                    print(f"  Description: {tool.desc}")
                    if tool.args:
                        args_str = ", ".join(f"{k}: {v}" for k, v in tool.args.items())
                        print(f"  Arguments: {args_str}")
                    else:
                        print("  Arguments: None")
                print("\n" + "=" * 120 + "\n")
                continue

            if user_input.lower() in ["/inspect_history", "inspect_history"]:
                print("\n" + "=" * 60 + " INTERACTION HISTORY " + "=" * 60)
                inspect_history()
                saved_path = save_inspect_history_to_file("dspy_react")
                print(f"\n💾 History saved to: {saved_path}")
                print("=" * 120 + "\n")
                continue

            if user_input.lower() in ["/trajectory", "trajectory"]:
                print("\n" + "=" * 60 + " LAST TRAJECTORY " + "=" * 60)
                if last_response and hasattr(last_response, "trajectory"):
                    trajectory = last_response.trajectory
                    if trajectory:
                        # Determine number of steps from trajectory keys
                        steps = len([k for k in trajectory.keys() if k.startswith("thought_")])

                        for i in range(steps):
                            print(f"\n--- Iteration {i} ---")
                            if f"thought_{i}" in trajectory:
                                print(f"Thought: {trajectory[f'thought_{i}']}")
                            if f"tool_name_{i}" in trajectory:
                                print(f"Tool: {trajectory[f'tool_name_{i}']}")
                            if f"tool_args_{i}" in trajectory:
                                print(f"Args: {json.dumps(trajectory[f'tool_args_{i}'], indent=2)}")
                            if f"observation_{i}" in trajectory:
                                print(f"Observation: {trajectory[f'observation_{i}']}")
                    else:
                        print("Trajectory is empty.")
                else:
                    print("No trajectory available yet. Make a query first.")
                print("=" * 120 + "\n")
                continue

            if not user_input:
                continue

            # Process user input with conversation history
            response = agent(question=user_input, history=history)

            print(f"\nAgent: {response.answer}")

            # Count steps from trajectory
            if hasattr(response, "trajectory") and response.trajectory:
                steps = len([k for k in response.trajectory.keys() if k.startswith("thought_")])
                print(f"(Completed in {steps} steps)\n")
            else:
                print()

            # Store last response for trajectory inspection
            last_response = response

            # Manually update conversation history
            history.messages.append({"question": user_input, **response})

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
