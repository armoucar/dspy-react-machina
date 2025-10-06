"""Test runner entry points for uv run commands."""

import http.server
import os
import socketserver
import sys
import webbrowser

import pytest


def run_tests() -> None:
    """Run the test suite without coverage."""
    sys.exit(pytest.main(["tests/dspy_react_machina/", "-v"]))


def run_tests_with_coverage() -> None:
    """Run the test suite with coverage reporting.

    Generates both terminal and HTML coverage reports.

    Usage:
        uv run tests-coverage          # Generate reports only
        uv run tests-coverage --web    # Generate reports and serve HTML in browser
    """
    # Check if --web flag is present
    serve_web = "--web" in sys.argv

    # Run tests with coverage (both terminal and HTML output)
    exit_code = pytest.main(
        [
            "tests/dspy_react_machina/",
            "--cov=src/dspy_react_machina",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
        ]
    )

    # If tests passed and --web flag present, start HTTP server
    if exit_code == 0 and serve_web:
        htmlcov_dir = "htmlcov"

        if not os.path.exists(htmlcov_dir):
            print(f"\n❌ Coverage HTML directory '{htmlcov_dir}' not found!")
            sys.exit(1)

        port = 8000

        # Change to htmlcov directory
        os.chdir(htmlcov_dir)

        print(f"\n🌐 Starting HTTP server on http://localhost:{port}")
        print("📊 Opening coverage report in browser...")
        print("Press Ctrl+C to stop the server\n")

        # Open browser
        webbrowser.open(f"http://localhost:{port}")

        # Start server
        handler = http.server.SimpleHTTPRequestHandler

        try:
            with socketserver.TCPServer(("", port), handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✅ Server stopped")
            sys.exit(0)

    sys.exit(exit_code)
