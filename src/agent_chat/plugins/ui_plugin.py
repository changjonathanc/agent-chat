import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import WebSocket

from ..tool_registry import ToolResult
from ..utils import collect_all_pending


class UILogHandler(logging.Handler):
    """Logging handler that forwards log records to the UI."""

    def __init__(self, ui: "BaseUIPlugin"):
        super().__init__()
        self.ui = ui

    def emit(self, record: logging.LogRecord) -> None:
        # Filter out debug level logs from being sent to client
        if record.levelno > logging.DEBUG:
            # Check if record has structured data
            if hasattr(record, "structured"):
                self.ui.log_structured(record.structured, record.created)
            else:
                # Fallback to simple string logs
                log_entry = self.format(record)
                self.ui.log(log_entry)


class BaseUIPlugin:
    def __init__(self, interrupt_event=None):
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.status = "Connected"
        self.timezone_offset = 0  # minutes offset as received from client
        self.timezone_name = "UTC"
        self.interrupt_event = interrupt_event
        self.is_paused = False  # Flag for pause state
        self.resume_event = asyncio.Event()
        self.resume_event.set()  # Initially not paused
        self.pause_event = asyncio.Event()  # Event that gets set when paused

    async def add_message(self, message_data: dict):
        """Add a structured message to the queue.

        Expected format:
        {
            "type": "user_message",
            "content": str,
            "timestamp": str
        }
        """
        await self.message_queue.put(message_data)

        # Signal any waiting tools that user input arrived
        if self.interrupt_event:
            self.interrupt_event.set()

    def format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for UI display (kept for UI-specific uses)."""
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        tz = timezone(-timedelta(minutes=self.timezone_offset))
        dt = dt.astimezone(tz)
        return dt.strftime(f"%Y-%m-%d %H:%M:%S {self.timezone_name}")

    def log(self, content: str) -> None:
        """Send a log message to the UI asynchronously."""
        asyncio.create_task(self._send_internal_message(content, message_type="log"))

    def log_structured(self, structured_data: dict, timestamp: float) -> None:
        """Send structured log to UI (formatted as string for now)."""
        # Format structured data back to string format for log display
        formatted_content = self._format_structured_log(structured_data)
        asyncio.create_task(
            self._send_internal_message(
                formatted_content, message_type="structured_log"
            )
        )

    def _format_structured_log(self, data: dict) -> str:
        """Format structured log data back to the expected string format."""
        log_type = data.get("log_type", "")
        agent_id = data.get("agent_id", "main")

        # Add agent prefix for sub-agents
        prefix = f"[{agent_id}] " if agent_id != "main" else ""

        # Define formatters for each log type
        formatters = {
            "tool_call": lambda: f"{data.get('tool_name', 'unknown')}({data.get('arguments', '')})",
            "tool_result": lambda: f"{data.get('tool_name', 'unknown')} executed - returned: {data.get('result', '')}",
            "reasoning": lambda: data.get("content", ""),
            "output_text": lambda: data.get("content", ""),
            "user_input": lambda: data.get("content", ""),
            "message": lambda: f"{data.get('role', 'unknown')} - {data.get('content', '')}",
        }

        # Special handling for tool_signal (has conditional logic)
        if log_type == "tool_signal":
            content = data.get("content", "")
            if content:
                return f"{prefix}TOOL_SIGNAL: {content}"
            tool_name = data.get("tool_name", "")
            signal = data.get("signal", "unknown")
            if tool_name:
                return f"{prefix}TOOL_SIGNAL: {tool_name} requested {signal}"
            return f"{prefix}TOOL_SIGNAL: {signal}"

        # Use formatter if available
        if log_type in formatters:
            return f"{prefix}{log_type.upper()}: {formatters[log_type]()}"

        # Fallback to content field
        return data.get("content", str(data))

    async def _send_state_update(self):
        await self._send_to_ui({"type": "state", "status": self.status})

    # Abstract methods that differ between single and multiplayer
    async def _send_to_ui(self, message_data: dict):
        """Send message to UI (single websocket or broadcast)."""
        raise NotImplementedError

    async def _format_messages_for_model(self, messages: list) -> str:
        """Format messages for the Agent/Model."""
        raise NotImplementedError

    async def send_message(
        self,
        content: str,
        wait_for_response: bool = False,
        stop: bool = False,
    ):
        """Send a message to the user immediately and optionally wait for a reply.

        Parameters
        ----------
        content: str
            Message to send to the user.
        wait_for_response: bool, optional
            When ``True`` the method will block until the next message from the
            queue is received and return it. Defaults to ``False``.
        stop: bool, optional
            When ``True``, this will stop the tool call loop.
            Use this when you don't want to send any more messages to the user.
        """

        # Send the chat message
        message_data = {
            "type": "chat",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),  # Generate message ID for AI responses
        }
        await self._send_to_ui(message_data)

        if wait_for_response:
            # pause typing indicator while waiting for user reply
            prev_state = self.is_running
            prev_status = self.status
            self.is_running = False
            self.status = "Waiting for response"
            await self._send_state_update()
            user_response = await self._read_all_messages()
            self.is_running = prev_state
            self.status = prev_status
            await self._send_state_update()
            if stop:
                return ToolResult(user_response, stop_run=True)
            return user_response

        if stop:
            return ToolResult("message sent", stop_run=True)
        else:
            return "message sent"

    async def _send_internal_message(self, content: str, message_type: str = "chat"):
        """Send internal messages that bypass session tracking (for errors, state updates, etc.)."""
        message_data = {
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        await self._send_to_ui(message_data)

    async def _read_all_messages(self, first_message=None):
        """Read and combine all pending messages from queue."""
        if first_message is None:
            first_message_data = await self.message_queue.get()
        else:
            first_message_data = first_message

        additional_messages = collect_all_pending(self.message_queue)
        messages = [first_message_data] + additional_messages

        message_ids = [msg_data["message_id"] for msg_data in messages if "message_id" in msg_data]
        if message_ids:
            await self._send_to_ui({"type": "read_receipt", "message_ids": message_ids})

        return await self._format_messages_for_model(messages)

    def hook_provide_tools(self):
        """Return tools this plugin provides for auto-registration."""
        return [self.send_message]

    def hook_provide_system_prompt(self):
        """Return system prompt addition for UI functionality."""
        return """
## User Communication

- You should only communicate with user using the send_message tool.
- Only the payload of the send_message tool will be visible to the user.
- Any response you send outside of the send_message tool will NOT be visible.
  - The send_message tool also accepts an optional `wait_for_response` flag.
    When true it waits for the next user message and returns it.
    Use it for internal monologues or messages to the system admin.

When communicating with users:
- Use send_message() to send responses immediately to the user
  - You can split the response into multiple parts using the send_message tool.
    Each part will be sent as a separate message to the user.
- Each send_message call creates a separate message bubble in the UI
- Use send_message strategically to provide updates during long-running operations
- Use stop=True to finish responding to the user.

### note about the UI:
- the UI is a simple chat interface
- it doesn't support markdown, so don't use it
- it's a simple chat interface, so keep messages short and concise
- each message should be broken up into lines. soft limit: 10 lines
- use send_message wait_for_response=True to ask for clarifying feedback
""".strip()

    async def hook_modify_tool_result(self, tool_result):
        """Handle pause and message queue processing (timestamps handled by TimestampPlugin)."""
        # Check for pause flag - if paused, stop early and keep messages queued
        if self.is_paused:
            tool_result["stop_run"] = True
            return tool_result

        # Only check for new messages if not paused
        if not self.message_queue.empty():
            new_message = await self._read_all_messages()
            logging.getLogger(__name__).info(
                "SYSTEM: cleared message queue and put it in tool result"
            )
            tool_result["output"] += (
                f"\n\n<system message>New messages from user: {new_message}</system message>"
            )
            # If we have new messages, override any previous stop=True from model
            tool_result["stop_run"] = False

        return tool_result

    async def hook_modify_model_response(self, content: str):
        """Handle model responses that don't use send_message tool by auto-routing them.

        Returns the content (potentially enhanced) for presentation.
        The original content is preserved in conversation context.
        """
        try:
            # For UI, we don't enhance content but do auto-route to user
            # Other plugins might enhance content (link expansion, formatting, etc.)
            await self.send_message(content)
            return content  # Return content as-is for UI plugin
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in model response hook: {e}")
            return content  # Return original content on error

    def pause(self):
        """Pause agent processing.

        Sets pause flag and blocks processing until resumed.
        """
        self.is_paused = True
        self.resume_event.clear()  # Block processing
        self.pause_event.set()  # Signal pause happened
        self.status = "Paused"
        logging.getLogger(__name__).info("UI: Pause requested")

    def resume(self):
        """Resume agent processing.

        Clears pause flag and unblocks processing.
        """
        self.is_paused = False
        self.resume_event.set()  # Unblock processing
        self.pause_event.clear()  # Clear pause signal
        self.status = "Connected"
        logging.getLogger(__name__).info("UI: Resume requested")


# Single-user UI Plugin (original behavior)
class UIPlugin(BaseUIPlugin):
    def __init__(self, interrupt_event=None):
        super().__init__(interrupt_event)
        self.websocket = None

    async def set_websocket(self, websocket: WebSocket):
        self.websocket = websocket
        self.status = "Connected"
        await self._send_state_update()

    async def _send_to_ui(self, message_data: dict):
        """Send message to single websocket."""
        if self.websocket:
            await self.websocket.send_text(json.dumps(message_data, ensure_ascii=False))

    async def _format_messages_for_model(self, messages: list) -> str:
        """Format messages for the Agent/Model with timestamp."""
        formatted_messages = []
        for msg_data in messages:
            content = msg_data.get("content", "")
            timestamp = msg_data.get("timestamp", "")

            # Format timestamp for readability
            readable_ts = self.format_timestamp(timestamp) if timestamp else ""

            # user_message or default
            formatted_messages.append(
                f'<message timestamp="{readable_ts}">{content}</message>'
            )

        # Join all messages
        return "\n".join(formatted_messages)


# Multiplayer UI Plugin (new shared session behavior)
class MultiplayerUIPlugin(BaseUIPlugin):
    def __init__(self, session_data, interrupt_event=None):
        super().__init__(interrupt_event)
        self.session_data = session_data

    async def send_message(
        self, content: str, wait_for_response: bool = False, stop: bool = False
    ):
        """Override to add AI responses to session history."""
        # Call parent method to handle the actual sending
        result = await super().send_message(content, wait_for_response, stop)

        # Add AI response to session history
        ai_message = {
            "type": "chat",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),  # Generate message ID for AI messages
            "user_name": "Assistant",
            "user_id": -1,  # Special ID for AI
        }
        self.session_data.add_message_to_history(ai_message)

        return result

    async def _send_to_ui(self, message_data: dict):
        """Broadcast message to all connected websockets."""
        for websocket in self.session_data.users.keys():
            try:
                await websocket.send_text(json.dumps(message_data, ensure_ascii=False))
            except:
                # Handle disconnected websockets
                pass

    async def _format_messages_for_model(self, messages: list) -> str:
        """Format messages for the Agent/Model with user attribution."""
        formatted_messages = []
        for msg_data in messages:
            content = msg_data.get("content", "")
            timestamp = msg_data.get("timestamp", "")
            websocket = msg_data.get("websocket")  # Passed from message processing

            # Format timestamp for readability
            readable_ts = self.format_timestamp(timestamp) if timestamp else ""

            # Get user info for attribution
            user_info = self.session_data.users.get(websocket) if websocket else None
            user_name = user_info.name if user_info and user_info.name else ""
            user_id = user_info.user_id if user_info else ""

            formatted_messages.append(
                f'<message user_name="{user_name}" user_id="{user_id}" timestamp="{readable_ts}">{content}</message>'
            )

        return "\n".join(formatted_messages)
