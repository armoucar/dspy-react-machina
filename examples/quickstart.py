#!/usr/bin/env python3
"""
QuickStart Example - Simple Weather Agent Loop

A minimal example demonstrating ReActMachina's conversation history capabilities.
This is the same code shown in the README Quick Start section.

Usage:
    Interactive: uv run examples/quickstart.py
    Single query: uv run examples/quickstart.py --query "What's the weather in Paris?"
"""

import argparse
import os

import dspy

from dspy_react_machina import ReActMachina


def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 72°F, sunny"


def main():
    parser = argparse.ArgumentParser(description="QuickStart Weather Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query for one-turn response")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_api_key_here")
        return

    # Configure your LM
    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    # Create agent
    agent = ReActMachina("hist: dspy.History, question -> answer", tools=[get_weather])

    # Chat with persistent history
    history = dspy.History(messages=[])

    # Handle single query mode
    if args.query:
        response = agent(question=args.query, hist=history)
        print(response.answer)
        return

    # Interactive mode
    print("🤖 Weather Agent (type 'quit' or 'exit' to stop)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        response = agent(question=user_input, hist=history)
        print(f"Agent: {response.answer}\n")

        # Update history for next turn
        history = response.history


if __name__ == "__main__":
    main()
