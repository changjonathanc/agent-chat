"""
Main entry point for the Agent Chat application.

This module provides the entry point for running the chat application.
Can be called with: python -m agent_chat

Automatically opens the app in your browser once the server is reachable.
Disable with --no-open or AGENT_CHAT_NO_BROWSER=1.
"""

import argparse
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser

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
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the browser",
    )
    args = parser.parse_args()

    logging.getLogger(__name__).info("Starting chat server...")
    logging.getLogger(__name__).info(
        f"Open http://localhost:{args.port} in your browser to start chatting"
    )

    # Auto-open the browser once the server is reachable (best-effort)
    def _open_when_ready(url: str, timeout: float = 15.0, interval: float = 0.2):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    pass
                try:
                    webbrowser.open(url, new=1)
                except Exception:
                    pass  # Non-fatal if a browser cannot be opened
                return
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                time.sleep(interval)
            except Exception:
                time.sleep(interval)

    should_open = not args.no_open and os.environ.get("AGENT_CHAT_NO_BROWSER") != "1"
    url = f"http://localhost:{args.port}"
    if should_open:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
