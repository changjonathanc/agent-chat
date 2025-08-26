# Environment Abstraction Implementation Plan

This plan outlines phased steps to introduce the Environment abstraction proposed in the design document. The goal is to move responsibility for tools, hooks, prompt assembly, and policies out of the Agent while keeping the Agent responsible for conversation context.

## Phase 1: Introduce Environment Skeleton
- Create an `Environment` class inside `agent.py` to keep the initial change small.
- Implement minimal interfaces:
  - `instructions()` returns the existing system prompt.
  - `tool_schemas(provider)` returns current tool schema list.
  - `step(chunk=None)` acts as a pass-through and returns `None`.
- Instantiate `Environment` in the Agent and call `env.instructions()` and `env.tool_schemas()` when invoking the provider.
- Ensure existing behavior is preserved and tests still pass.

## Phase 2: Move Prompt and Tool Ownership
- Move system prompt assembly logic from `agent.py` into `Environment.instructions()`.
- Relocate `ToolRegistry` creation and registration into `Environment`.
- Update `env.tool_schemas(provider)` to build provider-shaped schemas using the registry.
- Remove equivalent code from `agent.py`.

## Phase 3: Handle Provider Events
- Expand `env.step(chunk)` to process streaming events from the provider.
- For `function_call` events:
  - Execute tools through `ToolRegistry` and run any tool hooks.
  - Return a `ToolResultDict` (`{"type": "function_call_output", "call_id": str, "output": str, "stop_run": bool}`).
- For other events (assistant text, reasoning, errors):
  - Route text to UI/logging plugins.
  - Perform side effects as needed and return `None`.
- Remove tool execution and hook logic from `agent.py`.

## Phase 4: Poll for Follow-up Input
- Implement `env.step()` with no arguments to poll for queued user/system messages.
- Environment formats and returns the next user message string or `None` if there is no message or policies prevent continuation.
- Replace any existing queue or injection logic in the Agent with calls to `env.step()`.

## Phase 5: Consolidate Policies and Observability
- Track pause/interrupt states, budgets, and other policies inside `Environment`.
- Allow tools to set `stop_run` to control the loop.
- Emit structured logs for tool calls, results, assistant output, reasoning summaries, and policy decisions via UI/logging plugins.
- Agent loop only inspects `stop_run` flags and decides whether to continue.

## Phase 6: Cleanup and Extensions
- Extract `Environment` into its own module if desired (`environment.py`) once the interface stabilizes.
- Provide optional `env.step_openai_event(chunk)` alias specialized for Responses API event types.
- Update documentation and examples to showcase the new Agent/Environment split.
- Remove dead code and ensure full test coverage of the new design.

Each phase should be implemented in a separate pull request with accompanying tests and documentation updates to keep changes reviewable and maintainable.
