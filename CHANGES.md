# Recent Changes to PDP-Agent (main.py)

## Summary
Fixed critical bugs in input handling, LLM integration, and added a dedicated `list_directory` tool for better folder listing support.

---

## Bugs Fixed

### 1. **Fixed `refactor_tool` message mutation (lines 50–54)**
   - **Issue**: Printed full `AIMessage` Pydantic repr; mutated `HumanMessage` object in place
   - **Fix**: 
     - Changed `print(response)` to `print("Refactored:", response.content)` — cleaner output
     - Replaced in-place mutation with immutable message reconstruction: `state["messages"][:-1] + [HumanMessage(...)]`
   - **Benefit**: No lost context; cleaner stdout; respects message immutability

### 2. **Fixed `main()` interactive loop (lines 213–238)**
   - **Issue**: `running = True` set but never used in a loop; agent ran once and exited despite README claiming "interactive loop"
   - **Fix**: Wrapped input/invoke block in proper `while running:` loop with clean exit on "no"
   - **Benefit**: True multi-turn conversation; persists conversation history via checkpointer thread

### 3. **Fixed response output (lines 232–238)**
   - **Issue**: Printed all messages (refactored input, intermediate tool calls, raw shell output, etc.)
   - **Fix**: Filter to only print the final `AIMessage` with content and no pending `tool_calls`
   - **Benefit**: Clean, relevant output; no confusion from intermediate steps

---

## Features Added

### 4. **Added `list_directory` tool (lines 84–103)**
   - Dedicated tool for listing directories (no more relying on LLM guessing shell commands)
   - Supports custom paths; defaults to `/home/pdp28`
   - Uses `ls -lA --group-directories-first` for clean, organized output
   - Handles errors gracefully (timeouts, exceptions)

### 5. **Registered `list_directory` tool (lines 107, 187)**
   - Bound to `llm_with_tools` for LLM access
   - Added to `tool_by_name` mapping in `tool_node` for execution

---

## Improvements

- **Better prompt response**: "list the folders in my laptop" now clearly calls `list_directory` instead of relying on LLM interpretation
- **Cleaner output**: Only final answer printed; no metadata dumps
- **True interactive experience**: Multi-turn conversation works as documented
- **Immutable state**: Message history respects LangChain conventions

---

## Testing

Run: `uv run main.py`

Test cases:
1. **List folders**: `"list the folders in my laptop"` → calls `list_directory` → clean output
2. **Multi-turn**: Enter multiple requests without restarting → conversation continues
3. **Exit**: Type `"no"` → clean exit
4. **Greeting**: Type `"hello"` → agent responds warmly (via refactor_tool)

---

## Files Modified

- `main.py`: All changes above

## Files NOT Modified

- `README.md` - Still accurate (now that loop actually works!)
- `.env.example` - No changes needed
- `requirements.txt` - Still uses `pyproject.toml` (not requirements.txt)
