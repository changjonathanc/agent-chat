import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .agent import Agent
from .plugins.agent_plugin import MainAgentPlugin
from .plugins.timestamp_plugin import TimestampPlugin
from .plugins.ui_plugin import UILogHandler, UIPlugin
from .plugins.web_plugin import WebPlugin
from .session_manager import SessionManager

load_dotenv()

app = FastAPI()

# Get the templates directory (in the same package)
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Global session manager for shared sessions
session_manager = SessionManager()


def get_default_system_prompt() -> str:
    """Get the default system prompt."""
    return """You are a helpful AI assistant in a chat interface.
Be concise, friendly, and direct in your responses. When users send multiple messages in quick succession, address all their points comprehensively in a single response.

- acknowledge user's message immediately 
- break down your response into smaller chunks

<tool usage guide>
- you can use multiple tools in parallel in a single turn
- use as many tools in a single turn as possible to minimaize latency
example:
- send_message to notify user what you are going to search, and at the same time, call the actual web search
</tool usage guide>"""


def create_default_agent(ui_plugin, interrupt_event, log_handler):
    """Create default general-purpose agent."""
    # Create timestamp plugin with UI plugin's timezone settings
    timestamp_plugin = TimestampPlugin(
        timezone_offset=ui_plugin.timezone_offset, timezone_name=ui_plugin.timezone_name
    )

    plugins = [
        WebPlugin(),
        ui_plugin,
        MainAgentPlugin(interrupt_event),
        timestamp_plugin,
    ]

    # Use default system prompt
    system_prompt = get_default_system_prompt()

    return Agent(
        plugins, system_prompt=system_prompt, log_handler=log_handler, agent_id="main"
    )


def create_research_agent(ui_plugin, interrupt_event, log_handler):
    """Create research-focused agent with specialized sub-agents."""

    # Research-specific MainAgentPlugin with research sub-agent prompts
    research_agent_plugin = MainAgentPlugin(
        interrupt_event=interrupt_event,
        subagent_prompt="""You are a research sub-agent specialized in comprehensive information gathering.

Your mission:
- Conduct focused, thorough research on specific aspects of topics
- Use web search extensively to gather current, accurate information
- Verify information across multiple authoritative sources
- Provide regular updates to the main research coordinator
- Synthesize findings into clear, actionable insights
- Always cite sources and verify claims

Research methodology:
1. Start with broad searches to understand the topic landscape
2. Dive deep into specific aspects as directed
3. Cross-reference information across multiple sources
4. Report findings progressively with source attribution
5. Flag any contradictions or uncertainties found

Focus on factual accuracy, source credibility, and comprehensive coverage of your assigned research area.""",
        subagent_model="o4-mini",
    )

    # Create timestamp plugin with UI plugin's timezone settings
    timestamp_plugin = TimestampPlugin(
        timezone_offset=ui_plugin.timezone_offset, timezone_name=ui_plugin.timezone_name
    )

    plugins = [
        ui_plugin,
        research_agent_plugin,
        timestamp_plugin,
    ]  # No WebPlugin - delegates to sub-agents

    system_prompt = """You are a research coordinator specialized in systematic information gathering and analysis.

Your role as a research coordinator:
- **Research Planning**: Break complex topics into focused research components
- **Team Coordination**: Spawn and manage specialized research sub-agents
- **Information Synthesis**: Combine findings from multiple sub-agents into comprehensive insights
- **Quality Control**: Ensure source verification and fact-checking across all research
- **Progressive Research**: Build knowledge incrementally with proper attribution

Your methodology:
1. Analyze research questions and break them into specific aspects
2. Spawn multiple research sub-agents with focused assignments
3. Coordinate parallel research streams across different topic areas
4. Synthesize sub-agent findings into coherent, well-sourced conclusions
5. Ensure all claims are verified across multiple authoritative sources

<tool usage guide>
- You do NOT have direct web access or realtime information- delegate all to sub-agents to perform web search
- Spawn sub-agents for each distinct research component or question
- Give sub-agents specific, focused research assignments
- Synthesize and cross-reference information from multiple sub-agents
- Always request source attribution from sub-agents
</tool usage guide>"""

    return Agent(
        plugins,
        system_prompt=system_prompt,
        log_handler=log_handler,
        agent_id="research",
    )


async def handle_common_websocket_message(message_data, ui_plugin):
    """Handle common WebSocket message types (timezone, pause, resume)"""
    if message_data.get("type") == "timezone":
        ui_plugin.timezone_offset = message_data.get("offset", 0)
        ui_plugin.timezone_name = message_data.get("timezone", "UTC")
        return True
    elif message_data.get("type") == "pause":
        ui_plugin.pause()
        await ui_plugin._send_state_update()
        return True
    elif message_data.get("type") == "resume":
        ui_plugin.resume()
        await ui_plugin._send_state_update()
        return True
    return False


async def handle_initial_websocket_message(
    websocket, ui_plugin, handle_user_message_callback=None
):
    """Handle first WebSocket message (timezone, pause/resume, or user message)"""
    try:
        init_data = await websocket.receive_text()
        init_msg = json.loads(init_data)

        # Handle common message types
        if await handle_common_websocket_message(init_msg, ui_plugin):
            return None  # Common message handled
        else:
            # Handle user message (callback handles session-specific logic)
            if handle_user_message_callback:
                await handle_user_message_callback(init_msg)
            return init_msg
    except Exception as e:
        logging.getLogger(__name__).warning(f"SYSTEM: Failed initial handshake: {e}")
        return None


async def process_websocket_message(
    message_data, ui_plugin, handle_user_message_callback=None
):
    """Process a WebSocket message with common routing logic"""
    # Handle common message types
    if await handle_common_websocket_message(message_data, ui_plugin):
        return  # Common message handled
    elif message_data.get("type") == "switch_agent":
        # This is only for regular sessions - return for special handling
        return message_data
    else:
        # Handle user message (callback handles session-specific logic)
        if handle_user_message_callback:
            await handle_user_message_callback(message_data)


async def message_loop(get_current_agent, ui_plugin: UIPlugin):
    """Orchestrates message processing between UI and Agent (with dynamic agent reference)."""
    while True:
        try:
            # Wait for resume if paused
            await ui_plugin.resume_event.wait()

            # Race between getting first message and pause detection
            message_task = asyncio.create_task(ui_plugin.message_queue.get())
            pause_task = asyncio.create_task(ui_plugin.pause_event.wait())

            done, pending = await asyncio.wait(
                [message_task, pause_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # If paused, continue to wait for resume
            if pause_task in done:
                continue

            # If got message, collect all remaining messages
            if message_task in done:
                # Get the first message that was already retrieved
                first_message_data = await message_task

                # Use read_message with the first message to handle collection and formatting
                message = await ui_plugin.read_message(first_message=first_message_data)

                # Set running state
                ui_plugin.is_running = True
                ui_plugin.status = "Running"
                await ui_plugin._send_state_update()

                # Process messages with current agent (might have switched)
                current_agent = get_current_agent()
                await current_agent.run(message)

                # Set not running state
                ui_plugin.is_running = False
                ui_plugin.status = "Connected" if not ui_plugin.is_paused else "Paused"
                await ui_plugin._send_state_update()

        except Exception as e:
            logging.getLogger(__name__).error(f"ERROR: Error processing message: {e}")
            ui_plugin.is_running = False
            ui_plugin.status = "Connected" if not ui_plugin.is_paused else "Paused"
            await ui_plugin._send_state_update()

            await ui_plugin._send_internal_message(
                f"Sorry, I encountered an error: {str(e)}"
            )


async def handle_websocket_session(websocket: WebSocket):
    """Helper function to handle a websocket session."""
    # Create shared interrupt event for this session
    interrupt_event = asyncio.Event()

    # Create UI plugin and log handler
    ui_plugin = UIPlugin(interrupt_event=interrupt_event)
    log_handler = UILogHandler(ui_plugin)
    await ui_plugin.set_websocket(websocket)

    # Track current agent and type
    current_agent = create_default_agent(ui_plugin, interrupt_event, log_handler)
    current_agent_type = "default"

    async def switch_agent_type(new_type: str):
        nonlocal current_agent, current_agent_type

        if new_type == current_agent_type:
            return f"Already in {new_type} mode"

        # Get context from current agent
        context = current_agent.get_conversation_context()

        # Create new agent
        if new_type == "research":
            new_agent = create_research_agent(ui_plugin, interrupt_event, log_handler)
        else:
            new_agent = create_default_agent(ui_plugin, interrupt_event, log_handler)

        # Transfer context
        new_agent.set_conversation_context(context)

        # Update references
        current_agent = new_agent
        current_agent_type = new_type

        # Send confirmation to UI
        await ui_plugin._send_to_ui(
            {
                "type": "agent_switched",
                "agent_type": new_type,
                "message": f"Switched to {new_type.title()} mode",
            }
        )

        return f"Switched to {new_type} mode"

    # Receive initial timezone info (first message from client)
    await handle_initial_websocket_message(
        websocket,
        ui_plugin,
        handle_user_message_callback=lambda msg: ui_plugin.add_message(
            {
                "type": msg.get("type", "user_message"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "message_id": str(uuid.uuid4()),  # Generate message ID for single-user mode
            }
        ),
    )

    # Start message processing loop with dynamic agent reference
    processing_task = asyncio.create_task(
        message_loop(lambda: current_agent, ui_plugin)
    )

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle agent switching (regular sessions only)
            if message_data.get("type") == "switch_agent":
                agent_type = message_data.get("agent_type", "default")
                await switch_agent_type(agent_type)
            else:
                # Use common message processing
                await process_websocket_message(
                    message_data,
                    ui_plugin,
                    handle_user_message_callback=lambda msg: ui_plugin.add_message(
                        {
                            "type": msg.get("type", "user_message"),
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("timestamp", ""),
                            "message_id": str(uuid.uuid4()),  # Generate message ID for single-user mode
                        }
                    ),
                )
    except WebSocketDisconnect:
        logging.getLogger(__name__).info("SYSTEM: Client disconnected")
    except Exception as e:
        logging.getLogger(__name__).error(f"ERROR: WebSocket error: {e}")
    finally:
        processing_task.cancel()
        ui_plugin.websocket = None


@app.get("/")
async def get():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await handle_websocket_session(websocket)


# Shared session routes
@app.post("/api/create-session")
async def create_shared_session():
    """Create a new shared session and return session ID."""
    system_prompt = get_default_system_prompt()
    session_id = session_manager.create_session(system_prompt)
    return {"session_id": session_id}


@app.get("/session/{session_id}")
async def get_shared_session(session_id: str):
    """Serve the shared session UI."""
    # For now, return the same HTML file (we'll need to modify the frontend later)
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.websocket("/ws/session/{session_id}")
async def shared_websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle shared session websocket connections."""
    await websocket.accept()
    await handle_shared_websocket_session(websocket, session_id)


async def handle_shared_websocket_session(websocket: WebSocket, session_id: str):
    """Handle a shared session websocket connection."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Handle initial user name message
    user_name = None
    try:
        init_data = await websocket.receive_text()
        init_msg = json.loads(init_data)

        if init_msg.get("type") == "user_join":
            user_name = init_msg.get("user_name", "").strip() or None
        elif init_msg.get("type") == "timezone":
            session_data.ui_plugin.timezone_offset = init_msg.get("offset", 0)
            session_data.ui_plugin.timezone_name = init_msg.get("timezone", "UTC")
        elif init_msg.get("type") == "pause":
            session_data.ui_plugin.pause()
            await session_data.ui_plugin._send_state_update()
        elif init_msg.get("type") == "resume":
            session_data.ui_plugin.resume()
            await session_data.ui_plugin._send_state_update()
        else:
            # Handle user message with websocket attribution
            message_data = {
                "type": init_msg.get("type", "user_message"),
                "content": init_msg.get("content", ""),
                "timestamp": init_msg.get("timestamp", ""),
                "websocket": websocket,
            }
            await session_data.ui_plugin.add_message(message_data)
    except Exception as e:
        logging.getLogger(__name__).warning(f"SYSTEM: Failed initial handshake: {e}")

    # Add user to session
    user_info = session_manager.add_user_to_session(session_id, websocket, user_name)
    if not user_info:
        await websocket.close(code=1008, reason="Failed to join session")
        return

    # Send user their assigned user_id
    await websocket.send_text(
        json.dumps(
            {
                "type": "user_assigned",
                "user_id": user_info.user_id,
                "user_name": user_info.name,
            },
            ensure_ascii=False,
        )
    )

    # Replay message history to new user
    await session_manager.replay_history_to_user(session_id, websocket)

    # Notify other users about new user
    await session_manager.notify_user_joined(session_id, user_info)

    # Start message processing loop if not already running
    if (
        not hasattr(session_data, "processing_task")
        or session_data.processing_task.done()
    ):
        session_data.processing_task = asyncio.create_task(
            message_loop(lambda: session_data.agent, session_data.ui_plugin)
        )

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle timezone/pause/resume with common function
            if message_data.get("type") in ["timezone", "pause", "resume"]:
                await process_websocket_message(message_data, session_data.ui_plugin)
            else:
                # Handle user messages with shared session-specific logic
                # Generate unique message ID for read receipts
                message_id = str(uuid.uuid4())
                timestamp = message_data.get("timestamp", "")
                if not timestamp:
                    timestamp = datetime.now().isoformat()
                
                # Broadcast user message to all other users in the session
                user_message_broadcast = {
                    "type": "chat",
                    "content": message_data.get("content", ""),
                    "timestamp": timestamp,
                    "message_id": message_id,
                }
                # Add user attribution for frontend
                if user_info.name:
                    user_message_broadcast["user_name"] = user_info.name
                if user_info.user_id is not None:
                    user_message_broadcast["user_id"] = user_info.user_id

                # Broadcast to all users (including sender for consistency)
                await session_data.ui_plugin._send_to_ui(user_message_broadcast)

                # Add websocket info to message for user attribution in Agent
                structured_message = {
                    "type": message_data.get("type", "user_message"),
                    "content": message_data.get("content", ""),
                    "timestamp": timestamp,  # Use the same timestamp as broadcast
                    "message_id": message_id,  # Include message ID for read receipts
                    "websocket": websocket,
                }
                await session_data.ui_plugin.add_message(structured_message)

                # Add to session history for replay
                session_data.add_message_to_history(user_message_broadcast)

    except WebSocketDisconnect:
        logging.getLogger(__name__).info(
            f"SYSTEM: User {user_info.user_id} disconnected"
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"ERROR: WebSocket error: {e}")
    finally:
        # Remove user from session
        should_cleanup = session_manager.remove_user_from_session(session_id, websocket)
        if should_cleanup:
            session_manager.cleanup_session(session_id)
            if hasattr(session_data, "processing_task"):
                session_data.processing_task.cancel()


# Main entry point moved to __main__.py for proper package execution
