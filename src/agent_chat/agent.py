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


class Agent:
    def __init__(
        self,
        plugins: list,
        model_name: str = "o3",
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

        # Create agent-specific logger with automatic agent_id injection
        self.logger = AgentLoggerAdapter(logger, agent_id or "main")

        # Set up tool registry and register tools from all plugins
        self.tool_registry = ToolRegistry()
        for plugin in self.plugins:
            if hasattr(plugin, "hook_provide_tools"):
                for method in plugin.hook_provide_tools():
                    self.tool_registry.register_callable(method)

        # Get auto-generated tool schemas
        self.tools = self.tool_registry.get_schemas()

        # Set base system prompt (passed from app)
        self.instructions = system_prompt

        # Collect and append plugin system prompt additions
        self._prepare_system_prompt()

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

    def _prepare_system_prompt(self):
        """Collect system prompt additions and update instructions."""
        self.plugin_system_prompts = []
        for plugin in self.plugins:
            if hasattr(plugin, "hook_provide_system_prompt"):
                try:
                    addition = plugin.hook_provide_system_prompt()
                    if addition and addition.strip():
                        self.plugin_system_prompts.append(addition.strip())
                except Exception as e:
                    logger.error(
                        f"Error collecting system prompt from {plugin.__class__.__name__}: {e}"
                    )

        if self.plugin_system_prompts:
            combined_additions = "\n\n".join(self.plugin_system_prompts)
            self.instructions = f"{self.instructions}\n\n{combined_additions}"

    async def _apply_tool_result_hooks(self, tool_result):
        """Apply tool result modification hooks from all plugins."""
        return await self._apply_hook("hook_modify_tool_result", tool_result)

    async def _apply_user_message_hooks(self, message):
        """Apply user message modification hooks from all plugins."""
        return await self._apply_hook("hook_modify_user_message", message)

    async def _apply_model_response_hooks(self, content: str):
        """Apply model response hooks from all plugins. Returns modified content for presentation."""
        return await self._apply_hook("hook_modify_model_response", content)

    async def _apply_tool_call_hooks(self, tool_name: str, arguments: dict):
        """Apply tool call modification hooks from all plugins. Returns modified arguments."""
        # Create ToolCall object for cleaner hook interface
        tool_call = ToolCall(name=tool_name, arguments=arguments.copy())

        # Apply hooks - they may return a modified ToolCall or None
        modified_tool_call = await self._apply_hook("hook_modify_tool_call", tool_call)

        # Extract arguments from the potentially modified ToolCall
        return modified_tool_call.arguments

    def log_item(self, item_type: str, extra: dict):
        """Log a structured item event."""
        structured_data = {"log_type": item_type, **extra}
        self.logger.info(
            f"{item_type.replace('_', ' ').title()} received",
            extra={"structured": structured_data},
        )

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

            # If no tool calls, conversation is complete
            if not tool_results:
                break

            # If any tool requested stop, break the loop
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

            iterations += 1

    async def _handle_streaming_response(self):
        """Handle streaming response with immediate tool execution and context building."""
        # Prepare arguments for Responses API with zero retention
        create_args = {
            "model": self.model_name,
            "input": self.conversation_context,  # Use local context (cookbook pattern)
            "instructions": self.instructions,
            "tools": self.tools,
            "tool_choice": "auto",
            "store": False,  # Zero data retention
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            "reasoning": {"summary": "auto"},
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
            if chunk.type == "response.output_item.done":
                # Log each output item as it completes
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

                    # Execute tool with both hooks
                    tool_result = (
                        await self.tool_registry.execute_tool_openai_response_api(
                            item,
                            tool_call_hook=self._apply_tool_call_hooks,
                            tool_result_hook=self._apply_tool_result_hooks,
                        )
                    )

                    # Check if this tool result indicates we should stop
                    if tool_result.get("stop_run", False):
                        should_stop = True

                    # Remove stop_run flag before adding to context (OpenAI API doesn't accept it)
                    clean_tool_result = {
                        k: v for k, v in tool_result.items() if k != "stop_run"
                    }

                    tool_results.append(clean_tool_result)
                    self.log_item(
                        "tool_result",
                        {"tool_name": item.name, "result": clean_tool_result["output"]},
                    )

                elif item.type == "reasoning":
                    reasoning_content = "\n".join([x.text for x in item.summary])
                    if reasoning_content.strip():  # Only log if there's actual content
                        self.log_item("reasoning", {"content": reasoning_content})
                elif item.type in ["output_text", "message"]:
                    # Extract text
                    if item.type == "output_text":
                        content_text = item.text
                    else:  # message
                        if isinstance(item.content, list):
                            content_text = "".join(
                                c.text if hasattr(c, "text") else str(c)
                                for c in item.content
                            )
                        else:
                            content_text = str(item.content)

                    # Log it
                    self.log_item(item.type, {"content": content_text})

                    # Apply hooks if needed
                    if content_text.strip() and (
                        item.type == "output_text"
                        or getattr(item, "role", None) == "assistant"
                    ):
                        await self._apply_model_response_hooks(
                            content_text
                        )
                else:
                    self.logger.info(f"STREAM: UNKNOWN ITEM: {item}")

            elif chunk.type == "response.completed":
                # Response completed - store the complete response output
                self.conversation_context.extend(chunk.response.output)

        # Add tool results to context if any
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
