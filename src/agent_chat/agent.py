import logging
import os

from openai import AsyncOpenAI

from .tool_registry import ToolCall, ToolRegistry

# Set up logging for message history
logger = logging.getLogger(__name__)


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically injects agent_id into structured logs."""

    def __init__(self, logger, agent_id):
        self.agent_id = agent_id
        super().__init__(logger, {})

    def process(self, msg, kwargs):
        # Inject agent_id into structured logs
        if "extra" in kwargs and "structured" in kwargs["extra"]:
            kwargs["extra"]["structured"]["agent_id"] = self.agent_id
        return msg, kwargs


class Environment:
    """Environment owns tools, hooks, and stream event handling."""

    def __init__(self, base_system_prompt: str, plugins: list, logger: logging.Logger):
        self.base_system_prompt = base_system_prompt
        self.plugins = plugins
        self.logger = logger

        # Initialize tool registry and register plugin tools
        self.tool_registry = ToolRegistry()
        for plugin in plugins:
            if hasattr(plugin, "hook_provide_tools"):
                for method in plugin.hook_provide_tools():
                    self.tool_registry.register_callable(method)

        # Build complete system prompt
        self._instructions = self._assemble_system_prompt()

    def _assemble_system_prompt(self) -> str:
        instructions = self.base_system_prompt
        additions = []
        for plugin in self.plugins:
            if hasattr(plugin, "hook_provide_system_prompt"):
                try:
                    addition = plugin.hook_provide_system_prompt()
                    if addition and addition.strip():
                        additions.append(addition.strip())
                except Exception as e:
                    self.logger.error(
                        f"Error collecting system prompt from {plugin.__class__.__name__}: {e}"
                    )
        if additions:
            instructions = f"{instructions}\n\n" + "\n\n".join(additions)
        return instructions

    def instructions(self) -> str:
        """Return the assembled system prompt."""
        return self._instructions

    def tool_schemas(self, provider: str) -> list:
        """Return provider-shaped tool schemas."""
        # Provider is unused in this phase but kept for compatibility
        return self.tool_registry.get_schemas()

    async def _apply_hook(self, hook_name: str, value):
        for plugin in self.plugins:
            if hasattr(plugin, hook_name):
                try:
                    result = await getattr(plugin, hook_name)(value)
                    if result is not None:
                        value = result
                except Exception as e:
                    self.logger.error(
                        f"Error in {hook_name} from {plugin.__class__.__name__}: {e}"
                    )
        return value

    async def _apply_tool_call_hooks(self, tool_name: str, arguments: dict):
        tool_call = ToolCall(name=tool_name, arguments=arguments.copy())
        modified = await self._apply_hook("hook_modify_tool_call", tool_call)
        return modified.arguments

    async def _apply_tool_result_hooks(self, tool_result):
        return await self._apply_hook("hook_modify_tool_result", tool_result)

    async def _apply_model_response_hooks(self, content: str):
        return await self._apply_hook("hook_modify_model_response", content)

    def log_item(self, item_type: str, extra: dict):
        structured = {"log_type": item_type, **extra}
        self.logger.info(
            f"{item_type.replace('_', ' ').title()} received", extra={"structured": structured}
        )

    async def step(self, chunk=None):
        if chunk is None:
            for plugin in self.plugins:
                if hasattr(plugin, "has_messages") and plugin.has_messages():
                    message = await plugin.read_message()
                    self.log_item("user_input", {"content": message})
                    return message
            return None

        if chunk.type == "response.output_item.done":
            item = chunk.item
            if item.type == "function_call":
                self.log_item(
                    "tool_call",
                    {
                        "tool_name": item.name,
                        "arguments": item.arguments,
                        "call_id": item.call_id,
                    },
                )
                tool_result = await self.tool_registry.execute_tool_openai_response_api(
                    item,
                    tool_call_hook=self._apply_tool_call_hooks,
                    tool_result_hook=self._apply_tool_result_hooks,
                )
                self.log_item(
                    "tool_result", {"tool_name": item.name, "result": tool_result["output"]}
                )
                return tool_result

            if item.type == "reasoning":
                reasoning_content = "\n".join([x.text for x in item.summary])
                if reasoning_content.strip():
                    self.log_item("reasoning", {"content": reasoning_content})
                return None

            if item.type in ["output_text", "message"]:
                if item.type == "output_text":
                    content_text = item.text
                else:
                    if isinstance(item.content, list):
                        content_text = "".join(
                            c.text if hasattr(c, "text") else str(c) for c in item.content
                        )
                    else:
                        content_text = str(item.content)

                payload = {"content": content_text}
                if item.type == "message":
                    role = getattr(item, "role", None)
                    if role is not None:
                        payload["role"] = role
                self.log_item(item.type, payload)

                if content_text.strip() and (
                    item.type == "output_text" or getattr(item, "role", None) == "assistant"
                ):
                    await self._apply_model_response_hooks(content_text)
                return None

            self.logger.info(f"STREAM: UNKNOWN ITEM: {item}")
            return None

        if chunk.type == "response.error":
            self.logger.error(f"Stream error: {chunk}")
        return None


class Agent:
    def __init__(
        self,
        plugins: list,
        model_name: str = "gpt-5",
        system_prompt: str = "You are a helpful assistant.",
        log_handler=None,
        max_iterations=100,
        agent_id: str = None,
    ):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_context = []  # Local context for zero retention (cookbook pattern)
        self.plugins = plugins

        # Setup logging if handler provided
        if log_handler:
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            root_logger.addHandler(log_handler)
            root_logger.setLevel(logging.INFO)

        self.max_iterations = max_iterations  # Prevent infinite tool call loops
        self.model_name = model_name
        self.reasoning_effort = "minimal"  # default; can be changed via config

        # Create agent-specific logger with automatic agent_id injection
        self.logger = AgentLoggerAdapter(logger, agent_id or "main")

        # Initialize environment which manages tools and system prompt
        self.env = Environment(system_prompt, self.plugins, self.logger)

    async def _apply_hook(self, hook_name: str, value):
        """Generic hook application helper that handles None returns gracefully.

        Parameters
        ----------
        hook_name : str
            Name of the hook method to call on plugins
        value : Any
            The value to be modified by the hook chain

        Returns
        -------
        Any
            The final modified value after all plugin hooks have been applied
        """
        for plugin in self.plugins:
            if hasattr(plugin, hook_name):
                try:
                    result = await getattr(plugin, hook_name)(value)
                    if result is not None:
                        value = result
                except Exception as e:
                    self.logger.error(
                        f"Error in {hook_name} from {plugin.__class__.__name__}: {e}"
                    )

        return value

    async def _apply_user_message_hooks(self, message):
        """Apply user message modification hooks from all plugins."""
        return await self._apply_hook("hook_modify_user_message", message)

    async def run(self, message: str | None = None):
        """Process a single message with tool call loop."""
        if message:
            # Apply user message modification hooks
            message = await self._apply_user_message_hooks(message)
            # logger trigger just right before user message goes into context
            self.logger.info(
                "User message received",
                extra={"structured": {"log_type": "user_input", "content": message}},
            )
            self.conversation_context.append({"role": "user", "content": message})

        # Tool call loop - continue until no more tool calls
        iterations = 0

        while iterations < self.max_iterations:
            # Process using current context
            tool_results, should_stop = await self._handle_streaming_response()
            if should_stop:
                self.logger.info(
                    "Tool loop ending",
                    extra={
                        "structured": {
                            "log_type": "tool_signal",
                            "signal": "ending_loop",
                            "content": "Tool requested stop - ending tool call loop",
                        }
                    },
                )
                break

            if tool_results:
                iterations += 1
                continue

            next_msg = await self.env.step()
            if next_msg:
                next_msg = await self._apply_user_message_hooks(next_msg)
                self.conversation_context.append({"role": "user", "content": next_msg})
                iterations += 1
                continue

            break

    async def _handle_streaming_response(self):
        """Handle streaming response with immediate tool execution and context building."""
        # Prepare arguments for Responses API with zero retention
        create_args = {
            "model": self.model_name,
            "input": self.conversation_context,  # Use local context (cookbook pattern)
            "instructions": self.env.instructions(),
            "tools": self.env.tool_schemas("openai"),
            "tool_choice": "auto",
            "store": False,  # Zero data retention
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            "reasoning": {"summary": "auto", "effort": self.reasoning_effort},
        }

        # Start streaming response using new Responses API
        try:
            stream = await self.client.responses.create(**create_args)
        except Exception as e:
            logger.error(f"Error creating response: {e}")
            logger.error(f"{self.conversation_context=}")
            return [], False

        tool_results = []
        should_stop = False

        # Process streaming events
        async for chunk in stream:
            result = await self.env.step(chunk)
            if result:
                if result.get("stop_run"):
                    should_stop = True
                # Remove stop_run before storing
                clean_result = {k: v for k, v in result.items() if k != "stop_run"}
                tool_results.append(clean_result)
            if chunk.type == "response.completed":
                self.conversation_context.extend(chunk.response.output)

        if tool_results:
            self.conversation_context.extend(tool_results)

        return tool_results, should_stop

    def get_conversation_context(self):
        """Export current conversation context for transfer to another agent."""
        return self.conversation_context.copy()

    def set_conversation_context(self, context: list):
        """Import conversation context from another agent.

        Parameters
        ----------
        context : list
            The conversation context to import
        """
        self.conversation_context = context.copy()
        self.logger.info(f"Imported conversation context with {len(context)} items")

    # Configuration helpers
    def set_model_and_effort(self, model: str | None = None, effort: str | None = None) -> dict:
        """Set model and reasoning effort with validation and clamping.

        Returns the effective values that were applied.
        """
        allowed_models = {"gpt-5", "o3"}
        allowed_effort = {
            "gpt-5": ["minimal", "low", "medium", "high"],
            "o3": ["low", "medium", "high"],
        }

        effective_model = self.model_name
        effective_effort = self.reasoning_effort

        if model:
            if model not in allowed_models:
                model = self.model_name  # ignore invalid
            effective_model = model

        if effort:
            # Clamp effort to model's allowed set; map minimal->low for o3
            if effective_model == "o3" and effort == "minimal":
                effort = "low"
            if effort not in allowed_effort.get(effective_model, []):
                # fallback to current or default
                effort = (
                    self.reasoning_effort
                    if self.reasoning_effort in allowed_effort.get(effective_model, [])
                    else allowed_effort.get(effective_model, ["low"])[0]
                )
            effective_effort = effort

        # Apply
        self.model_name = effective_model
        self.reasoning_effort = effective_effort
        return {"model": effective_model, "effort": effective_effort}
