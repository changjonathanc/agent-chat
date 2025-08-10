# Agent Chat

A live chat demo showing how to build async messaging with agents using the Agent-Environment Middleware pattern.

Unlike traditional chat apps that block on API calls, this demonstrates true async messaging with an agent through composable plugins.

## Features

- **Message queue**: Send messages without waiting for responses
- **Shared chat**: Multiple users can talk to the same agent
- **Pause/resume**: Stop the agent while humans discuss
- **Background subagents**: Agent spawns helpers that work async
- **Agent handoff**: Switch between different agent modes

## How it works

Built with simple, composable plugins:

- **Agent loop**: Runs LLM calls and executes tools
- **Web Plugin**: Google search and URL fetching
- **UI Plugin**: Real-time message handling
- **Timestamp Plugin**: Adds timestamps to interactions
- **Multi Player Plugin**: Manages multiple users
- **Subagent Plugins**: Agent-to-agent communication

Each plugin provides tools, hooks into the agent-environment interaction, and composes cleanly with others.

## Learn more

This demo implements the patterns described in: [Agent-Environment Middleware](https://jonathanc.net/blog/agent-environment-middleware)

## Running

```bash
# Setup
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "GOOGLE_SEARCH_API_KEY=your-search-api-key" >> .env
echo "GOOGLE_SEARCH_ENGINE_ID=6398277409ff848d1" >> .env

# clone and run
git clone http://github.com/changjonathanc/agent-chat.git
cd agent-chat
uv run -m agent_chat

# run with uvx
uvx git+http://github.com/changjonathanc/agent-chat.git
# Access at http://localhost:8000
```

## Project Structure

```
src/agent_chat/
├── __init__.py
├── __main__.py           # Entry point
├── agent.py              # Core Agent class
├── app.py                # FastAPI server
├── session_manager.py    # Session handling
├── tool_registry.py      # Tool handling
├── utils.py              # Utilities
├── templates/            # HTML templates
└── plugins/
    ├── ui_plugin.py      # WebSocket & UI management
    ├── web_plugin.py     # Web search & content
    ├── agent_plugin.py   # Multi-agent support
    └── timestamp_plugin.py
```

## License

MIT
