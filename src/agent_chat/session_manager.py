import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import WebSocket

from .agent import Agent
from .plugins.agent_plugin import MainAgentPlugin
from .plugins.timestamp_plugin import TimestampPlugin
from .plugins.ui_plugin import MultiplayerUIPlugin
from .plugins.web_plugin import WebPlugin

logger = logging.getLogger(__name__)


class UserInfo:
    def __init__(self, user_id: int, name: Optional[str] = None):
        self.user_id = user_id
        self.name = name


class SessionData:
    def __init__(self, agent: Agent, ui_plugin: MultiplayerUIPlugin):
        self.agent = agent
        self.ui_plugin = ui_plugin
        self.users: Dict[WebSocket, UserInfo] = {}
        self.next_user_id = 0
        self.message_history = []  # Store messages for replay to new users

    def add_user(
        self, websocket: WebSocket, user_name: Optional[str] = None
    ) -> UserInfo:
        user_info = UserInfo(self.next_user_id, user_name)
        self.users[websocket] = user_info
        self.next_user_id += 1
        logger.info(
            f"User {user_info.user_id} ({user_info.name or 'anonymous'}) joined session"
        )
        return user_info

    def remove_user(self, websocket: WebSocket) -> Optional[UserInfo]:
        user_info = self.users.pop(websocket, None)
        if user_info:
            logger.info(
                f"User {user_info.user_id} ({user_info.name or 'anonymous'}) left session"
            )
        return user_info

    def get_user_count(self) -> int:
        return len(self.users)

    def add_message_to_history(self, message_data: dict):
        """Add message to history for replay to new users."""
        self.message_history.append(message_data)
        # Keep only last 100 messages to avoid memory issues
        if len(self.message_history) > 100:
            self.message_history.pop(0)


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.cleanup_task = None

    def create_session(
        self, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """Create a new shared session and return its ID."""
        session_id = str(uuid.uuid4())

        # Create shared interrupt event for this session
        interrupt_event = asyncio.Event()

        # Create plugins
        web_plugin = WebPlugin()
        main_agent_plugin = MainAgentPlugin(interrupt_event=interrupt_event)

        # Create session data placeholder (will be set after agent creation)
        session_data = SessionData(None, None)

        # Create multiplayer UI plugin with session data
        ui_plugin = MultiplayerUIPlugin(session_data, interrupt_event=interrupt_event)

        # Create timestamp plugin (shared sessions use UTC by default)
        timestamp_plugin = TimestampPlugin()

        plugins = [ui_plugin, web_plugin, main_agent_plugin, timestamp_plugin]

        # Create Agent with plugins and system prompt
        agent = Agent(plugins, system_prompt=system_prompt, agent_id="shared")

        # Update session data with actual agent and ui_plugin
        session_data.agent = agent
        session_data.ui_plugin = ui_plugin

        # Store session
        self.sessions[session_id] = session_data

        logger.info(f"Created new shared session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        return self.sessions.get(session_id)

    def add_user_to_session(
        self, session_id: str, websocket: WebSocket, user_name: Optional[str] = None
    ) -> Optional[UserInfo]:
        """Add a user to a session."""
        session_data = self.get_session(session_id)
        if not session_data:
            return None

        user_info = session_data.add_user(websocket, user_name)
        return user_info

    def remove_user_from_session(self, session_id: str, websocket: WebSocket) -> bool:
        """Remove a user from a session. Returns True if session should be cleaned up."""
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        session_data.remove_user(websocket)

        # If no users left, mark for cleanup
        if session_data.get_user_count() == 0:
            logger.info(f"Session {session_id} is empty, scheduling cleanup")
            return True

        return False

    def cleanup_session(self, session_id: str):
        """Remove an empty session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up empty session: {session_id}")

    async def notify_user_joined(self, session_id: str, user_info: UserInfo):
        """Notify existing users that a new user joined."""
        session_data = self.get_session(session_id)
        if not session_data:
            return

        # Create system message about user joining
        if user_info.name:
            join_message = f"System: {user_info.name} joined the chat"
        else:
            join_message = "System: Anonymous user joined the chat"

        # Send to all users via UI plugin
        await session_data.ui_plugin._send_to_ui(
            {
                "type": "system",
                "content": join_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def replay_history_to_user(self, session_id: str, websocket: WebSocket):
        """Replay message history to a newly joined user."""
        session_data = self.get_session(session_id)
        if not session_data or not session_data.message_history:
            return

        # Send history messages to the specific websocket
        for message_data in session_data.message_history:
            try:
                await websocket.send_text(json.dumps(message_data, ensure_ascii=False))
            except:
                # Handle disconnected websocket
                break

    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self.sessions)

    def get_total_user_count(self) -> int:
        """Get total number of users across all sessions."""
        return sum(session.get_user_count() for session in self.sessions.values())
