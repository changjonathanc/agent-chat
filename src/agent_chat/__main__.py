"""
Main entry point for the Agent Chat application.

This module provides the entry point for running the chat application.
Can be called with: python -m agent_chat
"""

import argparse
import logging

import uvicorn

from .app import app


def main():
    """Main entry point for the Agent Chat application."""
    parser = argparse.ArgumentParser(
        description="Agent Chat - Async messaging with agents"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    args = parser.parse_args()

    logging.getLogger(__name__).info("Starting chat server...")
    logging.getLogger(__name__).info(
        f"Open http://localhost:{args.port} in your browser to start chatting"
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
