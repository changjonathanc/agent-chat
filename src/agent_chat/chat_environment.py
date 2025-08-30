from __future__ import annotations

import asyncio
from queue import Empty as QueueEmpty

from .agent import Environment
from .utils import collect_all_pending


class ChatEnvironment(Environment):
    """Environment subclass that handles user message polling."""

    async def poll_message(self):
        for plugin in self.plugins:
            queue = getattr(plugin, "message_queue", None) or getattr(plugin, "inbox", None)
            if not queue:
                continue
            try:
                if hasattr(plugin, "_read_all_messages"):
                    message = await plugin._read_all_messages()
                else:
                    first = queue.get_nowait()
                    extras = collect_all_pending(queue)
                    parts = [first, *extras]
                    message = "\n".join(parts) if isinstance(first, str) else first
            except (asyncio.QueueEmpty, QueueEmpty):
                continue
            self.log_item("user_input", {"content": message})
            return message
        return None
