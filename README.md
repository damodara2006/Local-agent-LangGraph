# PDP-Agent: Autonomous Linux Automation Agent

A sophisticated AI-powered Linux automation agent that intelligently executes shell commands with self-correction capabilities and built-in verification mechanisms. Powered by LangChain, LangGraph, and advanced LLM reasoning.

## Overview

PDP-Agent is an autonomous agent designed to fulfill complex Linux automation tasks through natural language instructions. The agent employs intelligent decision-making, automatic error detection, and self-correction loops to ensure reliable task execution without constant user intervention.

## Key Features

🤖 **Intelligent Task Execution**
- Processes natural language requests and translates them into shell commands
- Automatically discovers and validates file paths using intelligent search mechanisms
- Executes commands with full context awareness

🔄 **Self-Correction Loop**
- Analyzes command output for errors and unexpected results
- Diagnoses failure root causes (permissions, missing files, syntax errors, environment issues)
- Automatically retries with corrected commands without requiring user approval
- Maximum 3 retry attempts per task with detailed error reporting

✅ **Comprehensive Verification**
- Validates every critical action (file creation, directory operations, config changes)
- Uses verification commands to confirm successful execution
- Reports verification results for complete transparency

🛡️ **Safety & Reliability**
- Uses absolute paths exclusively to prevent ambiguity
- Handles timeout errors, permission issues, and missing dependencies gracefully
- Provides clean, structured output summaries without verbose logs

## Technology Stack

- **LangChain**: LLM orchestration and prompt management
- **LangGraph**: Agent workflow and state management
- **ChatOpenRouter**: LLM API integration (step-3.5-flash:free)
- **Python 3.8+**: Core runtime

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Valid OpenRouter API key
- Linux/Unix-based system (or WSL on Windows)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd PDP-Agent
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### LLM Configuration

Modify the LLM settings in `main.py`:

```python
llm = ChatOpenRouter(
    model="stepfun/step-3.5-flash:free",  # Model selection
    temperature=0,                         # Response determinism
    max_retries=3                          # Retry attempts
)
```

## Usage

### Basic Execution

```bash
python main.py
```

### Interactive Mode

The agent operates in an interactive loop:

1. **Input Prompt**: `Enter you input | Enter no to exit:`
2. **Agent Processing**: Displays `⏳ Agent is processing your request...`
3. **Execution**: Automatically executes necessary shell commands
4. **Output**: Returns verified results with status information
5. **Loop**: Ready for next command or enter "no" to exit

### Example Commands

```
# File operations
"Create a file named test.txt in /home/pdp28/PDP-Agent"

# Directory management
"List all Python files in the current directory"

# System information
"Show disk usage for the home directory"

# Package management
"Install requirements from requirements.txt"
```

## Project Structure

```
PDP-Agent/
├── main.py                 # Main agent implementation
├── .env.example           # Environment template
├── .venv/                 # Virtual environment (auto-created)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── 1.Graph.ipynb         # Jupyter notebook for visualization
```

## Core Components

### `Tool_state` (TypedDict)
Manages the agent's state with message history for context maintenance.

### `shell_tool` (Function)
Executes shell commands with:
- Error handling and timeout management (30-second default)
- Output capture (stdout + stderr)
- Automatic input handling (auto-confirms with "y\n")
- Working directory: `/home/pdp28/`

### `llm_node` (Function)
- Processes message history (keeps first message + last 10 for token optimization)
- Invokes LLM with system prompt and user messages
- Binds tools for command execution

### `tool_node` (Function)
- Routes tool calls to appropriate handlers
- Appends tool responses to message history

### `graph` (StateGraph)
Orchestrates the workflow:
```
START → llm_node → if_tool_call → (tool_node → llm_node)*  → END
```

## Execution Flow

```
User Request
    ↓
Path Validation & Discovery
    ↓
Command Execution
    ↓
Output Analysis
    ↓
Verification
    ↓
Error Detection? → Self-Correct Loop → Re-execute
    ↓ (No errors)
Final Clean Summary
```

## Error Handling

The agent intelligently handles:

| Error Type | Strategy |
|-----------|----------|
| Timeout Errors | Increase timeout or reduce command scope |
| Permission Errors | Check file ownership, group membership, or suggest sudo |
| Missing Dependencies | Attempt installation or suggest alternatives |
| Environment Errors | Verify PATH, variables, and virtual environments |
| Path Not Found | Auto-discover using intelligent search |

## Message History Management

- Keeps the initial user message for context
- Retains the last 10 messages to avoid token limit overflow
- Automatically trims conversation when exceeding 11 messages
- Maintains message ordering for proper LLM reasoning

## Performance Considerations

- **Timeout**: 30 seconds per command (configurable in `shell_tool`)
- **Working Directory**: `/home/pdp28/` (configurable in `shell_tool`)
- **Max Retries**: 3 attempts per task (configurable in LLM initialization)
- **Token Optimization**: Automatic message history pruning

## Troubleshooting

### Agent Not Responding
- Check OpenRouter API key validity in `.env`
- Verify internet connectivity
- Check API rate limits

### Command Execution Failures
- Review error messages for root cause
- Ensure required tools/packages are installed
- Check file permissions and path accessibility

### Timeout Issues
- Increase timeout value in `shell_tool` function
- Break complex commands into smaller tasks
- Check system resource availability

## Contributing

Contributions are welcome! Please consider:

1. Maintaining code quality and readability
2. Adding comprehensive error handling
3. Updating documentation for new features
4. Testing with various command scenarios
5. Following the existing code style

## Support

For issues, questions, or suggestions:
- Review the troubleshooting section
- Check existing error messages and logs
- Verify environment configuration

## Future Enhancements

- [ ] Multi-step workflow orchestration
- [ ] File system monitoring and automation
- [ ] Task scheduling and cron integration
- [ ] Advanced logging and audit trails
- [ ] Web UI for command visualization
- [ ] Task history and rollback capabilities
- [ ] Integration with CI/CD pipelines

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Status**: Active Development
