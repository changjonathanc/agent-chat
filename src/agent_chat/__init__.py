"""
Chat LLM - A real-time chat application with OpenAI integration.

This package provides a streaming chat interface with tool-based communication,
supporting progressive responses through send_message tool calls.
"""

__version__ = "0.1.0"

from .agent import Agent
from .plugins.ui_plugin import UIPlugin
from .tool_registry import ToolRegistry, callable_to_tool_schema

__all__ = ["Agent", "UIPlugin", "ToolRegistry", "callable_to_tool_schema"]
