"""
Simple tool registry for automatic schema generation and tool execution.

Minimal implementation for mapping method callables to OpenAI tool schemas.
"""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, get_type_hints

logger = logging.getLogger(__name__)


class ToolCall(NamedTuple):
    """Container for tool call information passed to hooks."""

    name: str
    arguments: Dict[str, Any]


class ToolResult:
    """Enhanced tool result that can include control flags."""

    def __init__(self, output: str, stop_run: bool = False):
        """
        Create a tool result with optional control flags.

        Args:
            output: The tool output string
            stop_run: If True, stop the tool call loop after this result
        """
        self.output = output
        self.stop_run = stop_run

    def __str__(self):
        return self.output


def callable_to_tool_schema(
    callable_func: Callable, name: str, description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert a Python callable (function or method) to OpenAI tool schema format.

    Args:
        callable_func: The callable to convert
        name: Tool name
        description: Optional description

    Returns:
        OpenAI tool schema dictionary
    """
    sig = inspect.signature(callable_func)
    type_hints = get_type_hints(callable_func)

    # Get description from docstring if not provided
    if description is None:
        doc = inspect.getdoc(callable_func)
        description = doc.strip() if doc else f"Execute {name}"

    # Schema format for Responses API
    schema = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    # Process parameters
    for param_name, param in sig.parameters.items():
        # Skip self parameter
        if param_name == "self":
            continue

        # Get parameter type
        param_type = type_hints.get(param_name, str)

        # Convert to JSON schema type
        if param_type == str or param_type is str:
            json_type = "string"
        elif param_type == int or param_type is int:
            json_type = "integer"
        elif param_type == float or param_type is float:
            json_type = "number"
        elif param_type == bool or param_type is bool:
            json_type = "boolean"
        else:
            json_type = "string"  # Default fallback

        # Add parameter to schema
        param_schema = {"type": json_type, "description": f"The {param_name} parameter"}

        schema["parameters"]["properties"][param_name] = param_schema

        # Add to required if no default value
        if param.default is inspect.Parameter.empty:
            schema["parameters"]["required"].append(param_name)

    return schema


class ToolRegistry:
    """Registry for managing tools and their schemas."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}  # name -> callable
        self.schemas: List[Dict[str, Any]] = []  # OpenAI schemas

    def register_callable(
        self,
        callable_func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a callable (function or method) and auto-generate its OpenAI tool schema.

        Args:
            callable_func: The callable to register
            name: Optional name override (defaults to callable name)
            description: Optional description
        """
        tool_name = name or callable_func.__name__
        schema = callable_to_tool_schema(callable_func, tool_name, description)

        self.tools[tool_name] = callable_func
        self.schemas.append(schema)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for OpenAI API."""
        return self.schemas

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a registered tool by name.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        callable_func = self.tools[name]

        # Execute the callable (handle both sync and async)
        if inspect.iscoroutinefunction(callable_func):
            return await callable_func(**args)
        else:
            return callable_func(**args)


    async def execute_tool_openai_response_api(
        self, item: Any, tool_call_hook=None, tool_result_hook=None
    ) -> Dict[str, Any]:
        """
        Execute a tool from an OpenAI Responses API chunk item and return a tool result dictionary.

        Args:
            item: The chunk.item from response.output_item.done event (must have name, arguments, call_id).
            tool_call_hook: Optional async function to modify tool arguments before execution.
            tool_result_hook: Optional async function to modify tool result after execution.

        Returns:
            A tool result dictionary for the OpenAI Responses API format.
        """
        name = item.name

        try:
            # Parse arguments from JSON string
            args = json.loads(item.arguments)

            # Apply tool call hook to modify arguments if provided
            if tool_call_hook:
                args = await tool_call_hook(name, args)

            # Execute the tool
            result = await self.execute_tool(name, args)

            # Handle both string and ToolResult returns
            if isinstance(result, ToolResult):
                output = result.output
                stop_run = result.stop_run
            else:
                output = (
                    str(result) if result is not None else "Tool executed successfully"
                )
                stop_run = False
        except json.JSONDecodeError as e:
            logger.info(f"TOOL JSON ERROR: {name} - {str(e)}")
            output = f"Error parsing arguments: {str(e)}"
            stop_run = False
        except Exception as e:
            logger.info(f"TOOL ERROR: {name} - {str(e)}")
            output = f"Error: {str(e)}"
            stop_run = False

        # Return in Responses API format with stop_run flag
        tool_result = {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": output,
            "stop_run": stop_run,
        }

        # Apply tool result hook if provided
        if tool_result_hook:
            tool_result = await tool_result_hook(tool_result)

        return tool_result

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
        self.schemas.clear()

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self.tools)
