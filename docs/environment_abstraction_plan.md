# Environment Abstraction Implementation Plan

This plan outlines phased steps to introduce the Environment abstraction proposed in the design document. The goal is to move responsibility for tools, hooks, prompt assembly, and policies out of the Agent while keeping the Agent responsible for conversation context.

## Status Overview
- ✅ **Phase 1** – Environment skeleton created inside `agent.py` with `instructions()`, `tool_schemas()`, and a pass-through `step()`.
- ✅ **Phase 2** – System prompt assembly and `ToolRegistry` ownership moved into `Environment`.
- ✅ **Phase 3** – Provider event handling delegated to `Environment.step()` and tool logic removed from `Agent`.
- ⏳ **Phases 4–6** – Follow-up input polling, policies/observability, and cleanup remain.

## Phase 1: Introduce Environment Skeleton *(completed)*
- [x] Create an `Environment` class inside `agent.py` to keep the initial change small.
- [x] Implement minimal interfaces:
  - `instructions()` returns the existing system prompt.
  - `tool_schemas(provider)` returns current tool schema list.
  - `step(chunk=None)` acts as a pass-through and returns `None`.
- [x] Instantiate `Environment` in the Agent and call `env.instructions()` and `env.tool_schemas()` when invoking the provider.
- [x] Ensure existing behavior is preserved and tests still pass.

## Phase 2: Move Prompt and Tool Ownership *(completed)*
- [x] Move system prompt assembly logic from `agent.py` into `Environment.instructions()`.
- [x] Relocate `ToolRegistry` creation and registration into `Environment`.
- [x] Update `env.tool_schemas(provider)` to build provider-shaped schemas using the registry.
- [x] Remove equivalent code from `agent.py`.

## Phase 3: Handle Provider Events *(completed)*
- [x] Expand `env.step(chunk)` to process streaming events from the provider.
- For `function_call` events:
  - [x] Execute tools through `ToolRegistry` and run any tool hooks.
  - [x] Return a `ToolResultDict` (`{"type": "function_call_output", "call_id": str, "output": str, "stop_run": bool}`).
- For other events (assistant text, reasoning, errors):
  - [x] Route text to UI/logging plugins.
  - [x] Perform side effects as needed and return `None`.
- [x] Remove tool execution and hook logic from `agent.py`.

## Phase 4: Poll for Follow-up Input *(next)*
- [ ] Implement `env.step()` with no arguments to poll for queued user/system messages.
- [ ] Environment formats and returns the next user message string or `None` if there is no message or policies prevent continuation.
- [ ] Replace any existing queue or injection logic in the Agent with calls to `env.step()`.

## Phase 5: Consolidate Policies and Observability *(next)*
- [ ] Track pause/interrupt states, budgets, and other policies inside `Environment`.
- [ ] Allow tools to set `stop_run` to control the loop.
- [ ] Emit structured logs for tool calls, results, assistant output, reasoning summaries, and policy decisions via UI/logging plugins.
- [ ] Agent loop only inspects `stop_run` flags and decides whether to continue.

## Phase 6: Cleanup and Extensions *(next)*
- [ ] Extract `Environment` into its own module if desired (`environment.py`) once the interface stabilizes.
- [ ] Provide optional `env.step_openai_event(chunk)` alias specialized for Responses API event types.
- [ ] Update documentation and examples to showcase the new Agent/Environment split.
- [ ] Remove dead code and ensure full test coverage of the new design.

Each phase should be implemented in a separate pull request with accompanying tests and documentation updates to keep changes reviewable and maintainable.
