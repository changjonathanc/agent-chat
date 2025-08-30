"""Multi-agent plugin for spawning and managing sub-agents."""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from ..agent import Agent
from ..tool_registry import ToolResult
from ..utils import collect_all_pending
from .timestamp_plugin import TimestampPlugin
from .web_plugin import WebPlugin

logger = logging.getLogger(__name__)


class SubAgentPlugin:
    """Plugin for sub-agents that replaces UIPlugin."""

    def __init__(
        self,
        agent_name: str,
        parent_response_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ):
        self.agent_name = agent_name
        self.parent_response_queue = parent_response_queue
        self.inbox = asyncio.Queue()
        self.is_running = False
        self.should_stop = False
        self.stop_event = stop_event

    async def send_update(self, content: str, stop: bool = False):
        """Send an update back to the parent agent.

        Parameters
        ----------
        content : str
            The update message to send to the parent agent
        stop : bool, optional
            If True, signals that this agent is done and should stop
        """
        update = {
            "agent": self.agent_name,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "is_final": stop,
        }

        # Send to parent's response queue
        await self.parent_response_queue.put(update)
        logger.info(f"SUB_AGENT[{self.agent_name}]: Sent update to parent")

        if stop:
            self.should_stop = True
            return ToolResult("Update sent, stopping agent", stop_run=True)

        return "Update sent"

    async def _read_all_messages(self):
        """Read and combine pending messages from inbox."""
        first = await self.inbox.get()
        others = collect_all_pending(self.inbox)
        messages = [first, *others]
        return "\n".join(messages)

    def hook_provide_tools(self):
        """Provide tools for the sub-agent."""
        return [self.send_update]

    def hook_provide_system_prompt(self):
        """Provide system prompt addition for sub-agent functionality."""
        return """
## Sub-Agent Communication

You are a sub-agent working as part of a multi-agent system:

- You should only communicate with the main agent using the send_update tool
- Use send_update() to send progress reports, findings, or completed work back to the main agent
- You can send multiple updates during your work to show progress
- Use stop=True in send_update() when you have completed your assigned task
- The main agent may send you additional instructions at any time

When working on tasks:
- Use send_update() regularly to report progress and findings
- Break complex work into steps and update after each major step
- Always send a final update with stop=True when your task is complete
- If you encounter issues or need clarification, use send_update() to ask questions

Example workflow:
1. Receive task from main agent
2. Send initial update confirming task understanding
3. Work on task, sending progress updates
4. Send final results with stop=True

Your updates will be forwarded to the main agent and may be shown to users.
""".strip()

    async def hook_modify_tool_result(self, tool_result):
        """Check for new prompts from parent after each tool execution."""
        # Non-blocking check for new prompts
        pending_prompts = collect_all_pending(self.inbox)

        if pending_prompts:
            prompts_text = "\n".join(pending_prompts)
            tool_result["output"] += (
                f"\n\n<system message>New instructions from main agent:\n{prompts_text}</system message>"
            )

        return tool_result

    async def hook_modify_model_response(self, content: str):
        """Handle model responses that don't use send_update tool by auto-routing them.

        Returns the content (potentially enhanced) for presentation.
        The original content is preserved in conversation context.
        """
        try:
            # For sub-agents, we don't enhance content but do auto-route to main agent
            # Other plugins might enhance content (link expansion, formatting, etc.)
            await self.send_update(content)
            return content  # Return content as-is for sub-agent plugin
        except Exception as e:
            logger.error(f"Error in sub-agent model response hook: {e}")
            return content  # Return original content on error


class SubAgentRunner:
    """Manages the lifecycle of a sub-agent."""

    def __init__(
        self, name: str, model: str, response_queue: asyncio.Queue, system_prompt: str
    ):
        self.name = name
        self.model = model
        self.response_queue = response_queue
        self.system_prompt = system_prompt
        self.stop_event = asyncio.Event()
        self.sub_agent_plugin = SubAgentPlugin(name, response_queue, self.stop_event)
        self.task = None

    async def send_prompt(self, prompt: str):
        """Send a prompt to this sub-agent."""
        await self.sub_agent_plugin.inbox.put(prompt)
        logger.info(f"SUB_AGENT[{self.name}]: Received prompt")

    async def stop(self, timeout: int):
        """Signal the agent to stop."""
        logger.info(f"SUB_AGENT[{self.name}]: Stopping with timeout={timeout}s")
        self.stop_event.set()

        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"SUB_AGENT[{self.name}]: Stop timeout, cancelling task")
                self.task.cancel()

    async def run(self):
        """Run the sub-agent."""
        logger.info(f"SUB_AGENT[{self.name}]: Starting with model={self.model}")

        # Create plugins for sub-agent
        web_plugin = WebPlugin()
        timestamp_plugin = TimestampPlugin()  # Sub-agents use UTC by default
        plugins = [self.sub_agent_plugin, web_plugin, timestamp_plugin]

        # Create agent with base system prompt
        bot = Agent(
            plugins,
            model_name=self.model,
            system_prompt=self.system_prompt,
            agent_id=f"sub-{self.name}",
        )

        # Process messages until stopped
        while not self.stop_event.is_set():
            try:
                # Wait for message
                message = await self.sub_agent_plugin._read_all_messages()

                # Process message
                logger.info(f"SUB_AGENT[{self.name}]: Processing message")
                self.sub_agent_plugin.is_running = True
                await bot.run(message)
                self.sub_agent_plugin.is_running = False

                # Check if agent requested stop
                if self.sub_agent_plugin.should_stop:
                    logger.info(f"SUB_AGENT[{self.name}]: Agent requested stop")
                    break

            except Exception as e:
                logger.error(f"SUB_AGENT[{self.name}]: Error: {e}")
                error_update = {
                    "agent": self.name,
                    "content": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "is_final": True,
                }
                await self.response_queue.put(error_update)
                break

        logger.info(f"SUB_AGENT[{self.name}]: Stopped")


class MainAgentPlugin:
    """Plugin for main agent to spawn and manage sub-agents."""

    def __init__(
        self,
        interrupt_event=None,
        subagent_prompt: str = None,
        subagent_model: str = "o4-mini",
    ):
        self.agents: Dict[str, SubAgentRunner] = {}
        self.response_queues: Dict[str, asyncio.Queue] = {}
        self.interrupt_event = interrupt_event
        self.subagent_prompt = subagent_prompt
        self.subagent_model = subagent_model

    async def send_message_to_agent(self, agent_name: str, prompt: str):
        """Send a message to a sub-agent immediately without waiting.

        Parameters
        ----------
        agent_name : str
            Name of the agent to send message to (will be created if doesn't exist)
        prompt : str
            The prompt/instruction to send to the agent

        Returns
        -------
        str
            Confirmation that message was sent
        """
        # Spawn agent if it doesn't exist
        if agent_name not in self.agents:
            await self._spawn_agent(agent_name)

        # Send prompt to agent
        await self.agents[agent_name].send_prompt(prompt)
        logger.info(f"MAIN_AGENT: Sent prompt to {agent_name}")

        return f"Message sent to agent {agent_name}"

    async def wait_for_agent(self, agent_name: str = None, timeout: float = 60.0):
        """Wait for a response from a specific agent or any agent.

        Parameters
        ----------
        agent_name : str, optional
            Name of the specific agent to wait for. If None, waits for any agent.
        timeout : float, optional
            Maximum time to wait for response in seconds (default: 60.0)

        Returns
        -------
        str
            Response from the agent or status message about timeout/interruption
        """
        if agent_name and agent_name not in self.agents:
            return f"Agent {agent_name} does not exist. Use spawn_agent first."

        if self.interrupt_event:
            # ATOMIC: Clear event before waiting to only catch NEW interrupts
            self.interrupt_event.clear()

            if agent_name:
                # Wait for specific agent
                response_queue = self.response_queues[agent_name]
                response_task = asyncio.create_task(response_queue.get())
                interrupt_task = asyncio.create_task(self.interrupt_event.wait())

                tasks = [response_task, interrupt_task]
                wait_description = f"{agent_name}"
            else:
                # Wait for any agent to respond
                response_tasks = []
                for name, queue in self.response_queues.items():
                    if name in self.agents:  # Only active agents
                        task = asyncio.create_task(queue.get())
                        task.agent_name = name  # Tag task with agent name
                        response_tasks.append(task)

                if not response_tasks:
                    return "No active agents to wait for"

                interrupt_task = asyncio.create_task(self.interrupt_event.wait())
                tasks = response_tasks + [interrupt_task]
                wait_description = "any agent"

            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                if not done:
                    # Timeout occurred
                    return f"Timeout after {timeout} seconds waiting for {wait_description}"

                completed_task = done.pop()

                # Check what completed first
                if completed_task == interrupt_task:
                    return f"Interrupted by user input while waiting for {wait_description}"
                else:
                    # Agent responded
                    response = completed_task.result()
                    responding_agent = (
                        agent_name
                        if agent_name
                        else getattr(completed_task, "agent_name", "unknown")
                    )
                    return f"Response from {responding_agent}: {response['content']}"

            except Exception as e:
                return f"Error waiting for {wait_description}: {str(e)}"
        else:
            # Fallback to simple timeout if no interrupt event
            if agent_name:
                response_queue = self.response_queues[agent_name]
                try:
                    response = await asyncio.wait_for(
                        response_queue.get(), timeout=timeout
                    )
                    return f"Response from {agent_name}: {response['content']}"
                except asyncio.TimeoutError:
                    return f"Agent {agent_name} timeout after {timeout} seconds"
            else:
                return "Cannot wait for any agent without interrupt_event support"

    async def stop_agent(self, agent_name: str, timeout: int = 60):
        """Force stop a sub-agent immediately.

        Use this for immediate termination. For graceful stop, send a message asking the agent to wrap up.

        Parameters
        ----------
        agent_name : str
            Name of the agent to stop
        timeout : int, optional
            Seconds to wait for graceful shutdown before forcing (default: 60)
        """
        if agent_name in self.agents:
            logger.info(f"MAIN_AGENT: Force stopping agent {agent_name}")
            agent = self.agents[agent_name]
            await agent.stop(timeout)
            del self.agents[agent_name]
            del self.response_queues[agent_name]
            return f"Agent {agent_name} force stopped"
        return f"Agent {agent_name} not found"

    async def _spawn_agent(self, name: str):
        """Spawn a new sub-agent."""
        logger.info(
            f"MAIN_AGENT: Spawning agent {name} with model {self.subagent_model}"
        )

        response_queue = asyncio.Queue()
        self.response_queues[name] = response_queue

        # Create sub-agent runner with system prompt
        if not self.subagent_prompt:
            raise ValueError(f"Cannot spawn sub-agent '{name}' without a system prompt")

        runner = SubAgentRunner(
            name,
            self.subagent_model,
            response_queue,
            system_prompt=self.subagent_prompt,
        )
        self.agents[name] = runner

        # Start agent in background task
        runner.task = asyncio.create_task(runner.run())

    def hook_provide_tools(self):
        """Provide tools for the main agent."""
        return [self.send_message_to_agent, self.wait_for_agent, self.stop_agent]

    async def hook_modify_tool_result(self, tool_result):
        """Check for updates from all sub-agents and inject into tool result."""
        # Collect all pending updates from sub-agents non-blocking
        all_updates = []

        for agent_name, response_queue in self.response_queues.items():
            # Non-blocking collection of all pending updates from this agent
            agent_updates = collect_all_pending(response_queue)

            if agent_updates:
                # Format updates from this agent
                for update in agent_updates:
                    formatted_update = f"[{agent_name}] {update['content']}"
                    if update.get("is_final"):
                        formatted_update += " (final update - agent stopped)"
                    all_updates.append(formatted_update)

                    logger.info(f"MAIN_AGENT: Received update from {agent_name}")

        # Inject all updates into tool result if any
        if all_updates:
            updates_text = "\n".join(all_updates)
            tool_result["output"] += (
                f"\n\n<system message>Updates from sub-agents:\n{updates_text}</system message>"
            )
            logger.info(
                f"MAIN_AGENT: Injected {len(all_updates)} sub-agent updates into tool result"
            )

        return tool_result

    def hook_provide_system_prompt(self):
        """Provide system prompt addition for multi-agent functionality."""
        return """
## Multi-Agent Capabilities

You can spawn and manage sub-agents to handle tasks in parallel:

- Use send_message_to_agent() to create/message sub-agents (returns immediately)
- Use wait_for_agent() to wait for responses from specific agents or any agent
- Sub-agents run concurrently and can research, analyze, or perform tasks
- Use stop_agent() to terminate agents when done
- Each sub-agent has access to web search and browsing tools

**Tool Separation:**
- send_message_to_agent(agent_name, prompt) - Send message and return immediately
- wait_for_agent(agent_name=None, timeout=60.0) - Wait for agent response (interruptible)
  - If agent_name is provided, waits for that specific agent
  - If agent_name is None, waits for ANY agent to respond
  - Can be interrupted by user input

  **Important**: When waiting for sub-agent responses, if the user sends new input,
  you will be interrupted. However, the sub-agents continue working in the background.
  You can:
1. Respond to the user's new request immediately
2. Use wait_for_agent() again later to check for updates
3. Send a message asking agents to wrap up gracefully
4. Use stop_agent() to forcefully terminate if needed

Example workflow:
1. Send messages to multiple agents for different aspects of a task
2. Use wait_for_agent() to coordinate responses
3. If interrupted, handle user's new request but remember agents are still working
4. Use wait_for_agent() when convenient to collect findings
5. Use stop_agent() once an agent's task is done.

  **Note**: Sub-agent updates are automatically injected after each tool execution,
  so you'll see their progress naturally as you work.
"""
