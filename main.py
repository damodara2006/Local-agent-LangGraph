from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from typing import List, TypedDict, Optional, Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import ShellTool
from IPython.display import Image
import subprocess
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
load_dotenv()

class Tool_state(TypedDict):
    messages: List
    pending_commands: Optional[List[str]]
    approval_status: Optional[Literal["pending", "approved", "rejected"]]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3
)


# @tool
def refactor_tool(state : Tool_state) :
    """
    This is a helpful tool for the LLM, you need to refactor the input that the user asked for, in linux
    Example :
    user : list the folders
    your response : the user is asking to list the folders in the current directory including the hidden ones
    """
    print(state["messages"][0].content)
    state_msg = state["messages"][0].content

    print("⏳Refactoring the input")

    promt = ChatPromptTemplate.from_messages([
        ("system", """You are a message refactoring agent for linux. Your task is to:
         Example :
    user : list the folders
    your response : the user is asking to list the folders in the current directory including the hidden ones
1. Detect greetings (hello, hi, hey, good morning, etc.) and pass them through as-is with a greeting flag
2. For non-greeting messages, clarify and enhance the user's request for downstream LLM processing
3. Identify the core intent and requirements
4. Be concise and structured in your response
5. Format: If greeting, output "GREETING: [greeting]". Otherwise output the clarified request."""),
        ("human", "{input}")
    ])

    chain = promt | llm

    response = chain.invoke({
        "input" : state_msg
    })
    print("Refactored:", response.content)
    # Replace the last message with a new HumanMessage containing the refactored text
    refactored_messages = state["messages"][:-1] + [HumanMessage(content=response.content)]

    return {"messages": refactored_messages}

@tool
def shell_tool(commands: List[str]) -> str:
    """Use this tool to execute shell commands on the local machine.
Only run safe and necessary commands.
Return the command output exactly as produced by the shell.
Example : {"commands": ["echo 'Hello World!'"]}
"""
    results = []
    for cmd in commands:
        try:
            output = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd="/home/pdp28/",
                input="y\n",
                timeout=30
            )
            print(f"Executing : {cmd}")
            results.append(output.stdout + output.stderr)

        except subprocess.TimeoutExpired:
            results.append("Error: Command timed out after 30 seconds")
        except Exception as e:
            results.append(f"Error: {str(e)}")
    # print(results)
    return "\n".join(results)

@tool
def list_directory(path: str = "/home/pdp28") -> str:
    """List folders and files in a given directory path.
Use this when the user asks to list folders, files, or directory contents.
Defaults to the home directory if no path is specified.
Example: {"path": "/home/pdp28"}
"""
    try:
        output = subprocess.run(
            f"ls -lA --group-directories-first '{path}'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return output.stdout + output.stderr
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"

checkpointer = InMemorySaver()

llm_with_tools = llm.bind_tools([shell_tool, list_directory])

def llm_node(state : Tool_state) -> Tool_state:
    messages = state["messages"]

    # Keep only the last N messages to avoid token limits
    # Keep the initial user message + last 10 messages to maintain context
    if len(messages) > 11:
        messages = [messages[0]] + messages[-10:]

    print("\n⏳ Agent is processing your request...")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """### SYSTEM ROLE
You are an expert Linux Automation Agent with intelligent self-correction capabilities. Your 
mission is to efficiently fulfill user requests by executing appropriate shell commands. Your primary working directory is `/home/pdp28`.

### GREETING HANDLING
- If the input starts with "GREETING:", respond warmly to the greeting and ask how you can help
- Keep the response brief and friendly
- Then wait for the user's actual task

### CORE PRINCIPLES
1. **Proactive Path Resolution**:
   - When users mention paths or files that may not exist, automatically search for them using intelligent discovery.
   - Use `find . -iname "*keyword*" -type f 2>/dev/null | head -20` to locate resources.
   - Never assume paths; verify and correct them before proceeding.

2. **Intelligent Self-Correction (MANDATORY)**:
   - Always analyze command output for errors, warnings, or unexpected results.
   - Diagnose failure root causes (permissions, missing files, syntax errors, environment issues).
   - Immediately execute corrected commands without requesting user approval.
   - Maximum 3 retry attempts per task; then report the issue clearly.

3. **Comprehensive Verification**:
   - Validate every critical action (file creation, directory operations, config changes).
   - Use verification commands: `ls -la`, `cat`, `stat`, `file` to confirm success.
   - Report verification results to ensure transparency.

4. **Communication & Safety**:
   - Use absolute paths exclusively to prevent ambiguity.
   - Clearly explain any `sudo` requirements and security implications.
   - Provide concise, structured output summaries with no unnecessary verbose logs.
   - Do not include `tool_calls` in the final response when the task is complete.

5. **Error Handling**:
   - Timeout errors: Increase timeout or reduce command scope.
   - Permission errors: Check file ownership, group membership, or sudo requirements.
   - Missing dependencies: Install them or suggest alternatives.
   - Environment errors: Verify PATH, environment variables, and active virtual environments.

### EXECUTION STRATEGY
User Request → Path Validation → Command Execution → Output Analysis → Verification → (Errors? Self-Correct Loop) → Final Clean Summary"""
            ),
            ("placeholder", "{input}")
        ]
    )

    chain = prompt | llm_with_tools
    response = chain.invoke({"input":messages})
    # print(response)
    state["messages"].append(AIMessage(content=response.content, tool_calls=response.tool_calls, invalid_tool_calls=response.invalid_tool_calls ))
    # print(state)
    return {"messages" : state["messages"]}



def if_tool_call(state : Tool_state) -> str:
    messages = state["messages"][-1]

    if messages.tool_calls:
        return "approval"
    else:
        return "end"

def tool_node(state : Tool_state) -> Tool_state:
    """Extract tool calls and prepare them for approval"""
    messages = state["messages"][-1]

    tool_by_name = {t.name : t for t in [shell_tool, list_directory]}
    if messages.tool_calls:
        # Store all pending tool calls for approval
        pending_cmds = []
        for tool_call in messages.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            if tool_name == "shell_tool":
                commands = args.get("commands", [])
                pending_cmds.extend(commands)
            elif tool_name == "list_directory":
                path = args.get("path", "/home/pdp28")
                pending_cmds.append(f"list_directory(path='{path}')")

        if pending_cmds:
            state["pending_commands"] = pending_cmds
            state["approval_status"] = "pending"
            return {
                "messages": state["messages"],
                "pending_commands": pending_cmds,
                "approval_status": "pending"
            }

    return {"messages": state["messages"]}

def approval_node(state: Tool_state):
    """Ask user to approve pending commands before execution"""
    if not state.get("pending_commands"):
        return {"approval_status": "no_pending"}

    commands_str = "\n".join(f"  • {cmd}" for cmd in state["pending_commands"])

    # Use interrupt to pause and wait for user input
    decision = interrupt({
        "question": "Do you approve executing these commands?",
        "details": f"Pending commands:\n{commands_str}",
    })

    # Return the decision (will be resumed with the user's answer)
    if decision:
        return {"approval_status": "approved"}
    else:
        return {"approval_status": "rejected"}

def execute_commands(state: Tool_state) -> Tool_state:
    """Execute the approved tool calls"""
    messages = state["messages"][-1]
    tool_by_name = {t.name : t for t in [shell_tool, list_directory]}

    if messages.tool_calls:
        for tool_call in messages.tool_calls:
            tool_name = tool_call["name"]
            response = tool_by_name.get(tool_name).invoke(tool_call["args"])
            state["messages"].append(ToolMessage(content=str(response), tool_call_id=tool_call["id"]))

    return {
        "messages": state["messages"],
        "approval_status": "approved",
        "pending_commands": None
    }

def should_execute(state: Tool_state) -> str:
    """Decide whether to execute or cancel based on approval"""
    if state.get("approval_status") == "approved":
        return "execute_commands"
    elif state.get("approval_status") == "rejected":
        return "cancel_execution"
    else:
        return "llm_node"

def cancel_execution(state: Tool_state) -> Tool_state:
    """Cancel command execution"""
    print("❌ Command execution cancelled by user")
    return {
        "approval_status": "rejected",
        "pending_commands": None,
        "messages": state["messages"]
    }


state_graph = StateGraph(Tool_state)

state_graph.add_node("llm_node", llm_node)
state_graph.add_node("tool_node", tool_node)
state_graph.add_node("refactor_tool", refactor_tool)
state_graph.add_node("approval", approval_node)
state_graph.add_node("execute_commands", execute_commands)
state_graph.add_node("cancel_execution", cancel_execution)

state_graph.add_edge(START, "refactor_tool")
state_graph.add_edge("refactor_tool", "llm_node")
state_graph.add_conditional_edges("llm_node", if_tool_call, {"approval": "tool_node", "end": END})
state_graph.add_edge("tool_node", "approval")
state_graph.add_conditional_edges("approval", should_execute, {
    "execute_commands": "execute_commands",
    "cancel_execution": "cancel_execution",
    "llm_node": "llm_node"
})
state_graph.add_edge("execute_commands", "llm_node")
state_graph.add_edge("cancel_execution", END)

graph = state_graph.compile(
    checkpointer=checkpointer,
    interrupt_after=["approval"]  # Interrupt after approval node
)



def main():
    running = True
    config = {"configurable": {"thread_id": "1"}}

    while running:
        print("\n" + "="*60)
        user_input = input("Enter your input | Enter 'no' to exit: ").strip()
        if user_input.lower() == "no":
            print("Bye!")
            break

        if not user_input:
            continue

        # Initialize state with all required fields
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "pending_commands": None,
            "approval_status": None
        }

        try:
            # Invoke the graph with initial state
            graph.invoke(initial_state, config)

            # Check for interrupts in a loop
            while True:
                state = graph.get_state(config)

                # If there's no next node or we've finished, break
                if not state.next:
                    break

                # Check if we're at the approval node with pending commands
                if state.values.get("pending_commands"):
                    print("\n⚠️  APPROVAL REQUIRED:")
                    print("-"*60)
                    for cmd in state.values["pending_commands"]:
                        print(f"  • {cmd}")

                    approval = input("\nApprove execution? (yes/no): ").strip().lower()
                    decision = approval == "yes"

                    # Resume the graph with updated approval status
                    state.values["approval_status"] = "approved" if decision else "rejected"
                    graph.invoke(state.values, config)
                else:
                    # No pending commands, continue execution
                    graph.invoke(None, config)

            # Get final state and display response
            final_state = graph.get_state(config)
            print("\n✅ Agent Response:")
            print("-"*60)
            for msg in reversed(final_state.values["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    print(msg.content)
                    break

        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
