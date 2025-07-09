"""Utility functions for the agent_chat package."""

import asyncio


def collect_all_pending(queue):
    """Collect all pending items from an async queue non-blocking.

    Args:
        queue: asyncio.Queue to collect from

    Returns:
        List of collected items
    """
    items = []
    try:
        while not queue.empty():
            items.append(queue.get_nowait())
    except asyncio.QueueEmpty:
        pass
    return items
