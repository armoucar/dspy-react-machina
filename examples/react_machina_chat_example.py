#!/usr/bin/env python3
"""
ReActMachina Weather Agent Example

A simple weather agent demonstrating ReActMachina's conversation history and tool usage.
The agent can check weather for different cities and list available cities.

Usage:
    Interactive mode: uv run examples/react_machina_chat_example.py
    One-turn mode: uv run examples/react_machina_chat_example.py --query "What's the weather in Paris?"
    With Predict: uv run examples/react_machina_chat_example.py --predictor predict
    Async mode: uv run examples/react_machina_chat_example.py --async
    Inspect history: uv run examples/react_machina_chat_example.py --inspect-history
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

from dspy_react_machina import ReActMachina

# Try to import instrumentation if available
try:
    from instrumentation import configure_instrumentation
except ImportError:
    # Optional instrumentation module not available
    def configure_instrumentation(project_name="dspy-react-machina", endpoint="http://localhost:6006/v1/traces"):  # noqa: ARG001
        pass


logger = logging.getLogger(__name__)


def save_inspect_history_to_file(prefix: str = "react_machina") -> str:
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
                saved_path = save_inspect_history_to_file("react_machina")
                print(f"\n💾 History saved to: {saved_path}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Handle inspect history request (standalone)
    if args.inspect_history:
        inspect_history()
        saved_path = save_inspect_history_to_file("react_machina")
        print(f"\n💾 History saved to: {saved_path}")
        return

    # Interactive mode
    history = dspy.History(messages=[])
    last_response = None

    predictor_name = "ChainOfThought" if args.predictor == "cot" else "Predict"
    print("🤖 DSPy ReActMachina Weather Agent (Async Mode)")
    print("=" * 60)
    print(f"Predictor: {predictor_name}")
    print("I'm a weather agent that can check weather and list available cities.")
    print("Available tools: get_weather, list_weather_cities")
    print(
        "Commands: '/quit' to exit, '/tools' to list tools, '/inspect_history' to view full interaction history, '/trajectory' to view last trajectory"
    )
    print("Note: All tool interactions are preserved in conversation history.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["/quit", "quit", "exit"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() in ["/tools", "tools"]:
                print("\n" + "=" * 60 + " AVAILABLE TOOLS " + "=" * 60)
                for tool in agent.tool_registry.values():
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
                saved_path = save_inspect_history_to_file("react_machina")
                print(f"\n💾 History saved to: {saved_path}")
                print("=" * 120 + "\n")
                continue

            if user_input.lower() in ["/trajectory", "trajectory"]:
                print("\n" + "=" * 60 + " LAST TRAJECTORY " + "=" * 60)
                if last_response and hasattr(last_response, "trajectory"):
                    trajectory = last_response.trajectory
                    if trajectory:
                        # Determine number of steps from trajectory keys
                        steps = len([k for k in trajectory.keys() if k.startswith("state_")])

                        for i in range(steps):
                            print(f"\n--- Iteration {i} ---")
                            if f"state_{i}" in trajectory:
                                print(f"State: {trajectory[f'state_{i}']}")
                            if f"reasoning_{i}" in trajectory:
                                print(f"Reasoning: {trajectory[f'reasoning_{i}']}")
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

            # Process user input with current history (async)
            response = await agent.acall(question=user_input, history=history)

            print(f"\nAgent: {response.answer}")
            print(f"(Completed in {response.steps} steps)\n")

            # Store last response for trajectory inspection
            last_response = response

            # Update history with the agent's full interaction history
            history = response.history

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="DSPy ReActMachina Weather Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query for one-turn response")
    parser.add_argument("--inspect-history", "-i", action="store_true", help="Inspect the interaction history")
    parser.add_argument(
        "--predictor",
        "-p",
        type=str,
        choices=["predict", "cot"],
        default="cot",
        help="Predictor type: 'predict' for dspy.Predict, 'cot' for dspy.ChainOfThought (default: cot)",
    )
    parser.add_argument(
        "--async",
        "-a",
        dest="use_async",
        action="store_true",
        help="Use async mode for agent execution",
    )
    parser.add_argument(
        "--max-steps",
        "-m",
        type=int,
        default=10,
        help="Maximum number of agent steps before forcing completion (default: 10)",
    )
    parser.add_argument(
        "--debug-fail-mode",
        type=str,
        choices=["malformed-markers", "missing-markers"],
        default=None,
        help="[DEBUG ONLY] Force LLM to produce malformed output for error handling tests. DO NOT USE IN PRODUCTION.",
    )
    parser.add_argument(
        "--debug-fail-step",
        type=int,
        default=None,
        help="[DEBUG ONLY] Step index (0-based) to inject failure. If not specified, injects on all steps.",
    )
    args = parser.parse_args()

    # Map predictor argument to class
    predictor_map = {"predict": dspy.Predict, "cot": dspy.ChainOfThought}
    predictor_class = predictor_map[args.predictor]

    configure_instrumentation("dspy-react-machina")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return

    lm = dspy.LM(
        "openai/gpt-4.1-mini",
        api_key=api_key,
        temperature=0,
        cache=False,
    )

    dspy.settings.configure(lm=lm)

    # Initialize the ReActMachina agent with weather tools
    tools = [get_weather, list_weather_cities]
    signature = "question -> answer"

    # Build debug config if debug flags are set
    debug_config = None
    if args.debug_fail_mode:
        debug_config = {
            "fail_mode": args.debug_fail_mode,
            "fail_step": args.debug_fail_step,
        }

    agent = ReActMachina(
        signature,
        tools,
        max_steps=args.max_steps,
        predictor_class=predictor_class,
        _debug_config=debug_config,
    )  # type: ignore[arg-type]

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
                saved_path = save_inspect_history_to_file("react_machina")
                print(f"\n💾 History saved to: {saved_path}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Handle inspect history request (standalone)
    if args.inspect_history:
        inspect_history()
        saved_path = save_inspect_history_to_file("react_machina")
        print(f"\n💾 History saved to: {saved_path}")
        return

    # Interactive mode
    history = dspy.History(messages=[])
    last_response = None

    predictor_name = "ChainOfThought" if predictor_class == dspy.ChainOfThought else "Predict"
    print("🤖 DSPy ReActMachina Weather Agent")
    print("=" * 60)
    print(f"Predictor: {predictor_name}")
    print("I'm a weather agent that can check weather and list available cities.")
    print("Available tools: get_weather, list_weather_cities")
    print(
        "Commands: '/quit' to exit, '/tools' to list tools, '/inspect_history' to view full interaction history, '/trajectory' to view last trajectory"
    )
    print("Note: All tool interactions are preserved in conversation history.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["/quit", "quit", "exit"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() in ["/tools", "tools"]:
                print("\n" + "=" * 60 + " AVAILABLE TOOLS " + "=" * 60)
                for tool in agent.tool_registry.values():
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
                saved_path = save_inspect_history_to_file("react_machina")
                print(f"\n💾 History saved to: {saved_path}")
                print("=" * 120 + "\n")
                continue

            if user_input.lower() in ["/trajectory", "trajectory"]:
                print("\n" + "=" * 60 + " LAST TRAJECTORY " + "=" * 60)
                if last_response and hasattr(last_response, "trajectory"):
                    trajectory = last_response.trajectory
                    if trajectory:
                        # Determine number of steps from trajectory keys
                        steps = len([k for k in trajectory.keys() if k.startswith("state_")])

                        for i in range(steps):
                            print(f"\n--- Iteration {i} ---")
                            if f"state_{i}" in trajectory:
                                print(f"State: {trajectory[f'state_{i}']}")
                            if f"reasoning_{i}" in trajectory:
                                print(f"Reasoning: {trajectory[f'reasoning_{i}']}")
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

            # Process user input with current history
            response = agent(question=user_input, history=history)

            print(f"\nAgent: {response.answer}")
            print(f"(Completed in {response.steps} steps)\n")

            # Store last response for trajectory inspection
            last_response = response

            # Update history with the agent's full interaction history
            history = response.history

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
