from langchain_openrouter import ChatOpenRouter
from typing import List, TypedDict, Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain_community.tools import ShellTool
from IPython.display import Image
import subprocess
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

class Tool_state(TypedDict):
    messages : List
    pending_command: Optional[str]
    approval_status: Optional[Literal["pending", "approved", "rejected"]]


llm = ChatOpenRouter(
    model="stepfun/step-3.5-flash:free",
    temperature=0,
    max_retries=3
)

# @tool
def refactor_tool(state : Tool_state) :
    """
    This is a helpful tool for the LLM it will refactor the input that human
    """
    print(state["messages"][0].content)
    state_msg = state["messages"][0].content
    
    print(" ⏳ Refactoring the input")
    
    promt = ChatPromptTemplate.from_messages([
        ("system", """You are a message refactoring agent. Your task is to:
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

def approval_node(state: Tool_state) -> Tool_state:
    """Node that asks for command approval using interrupt"""
    pending_cmd = state.get("pending_command", "")

    print("\n🔍 Command to execute:")
    print(f"   {pending_cmd}")

    # Use interrupt to pause execution and ask for approval
    decision = interrupt({
        "question": "Approve this command?",
        "command": pending_cmd,
    })

    # Set approval status and return state for conditional routing
    state["approval_status"] = "approved" if decision else "rejected"
    return state


def execute_command_node(state: Tool_state) -> Tool_state:
    """Execute the approved command"""
    cmd = state.get("pending_command", "")

    try:
        print("⏳ Executing command...")
        output = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd="/home/pdp28/",
            input="y\n",
            timeout=30
        )
        result = output.stdout + output.stderr
        state["approval_status"] = "approved"
    except subprocess.TimeoutExpired:
        result = "Error: Command timed out after 30 seconds"
        state["approval_status"] = "approved"
    except Exception as e:
        result = f"Error: {str(e)}"
        state["approval_status"] = "approved"

    return {"approval_status": state["approval_status"], "messages": state["messages"]}


def skip_command_node(state: Tool_state) -> Tool_state:
    """Skip execution if command denied"""
    state["approval_status"] = "rejected"
    print("❌ Command execution denied by user")
    return {"approval_status": state["approval_status"], "messages": state["messages"]}


@tool
def shell_tool(commands: List[str]) -> str:
    """Use this tool to execute shell commands on the local machine.
Only run safe and necessary commands.
Return the command output exactly as produced by the shell.
Example : {"commands": ["echo 'Hello World!'"]}
"""
    # This is kept as a placeholder for now
    # Commands will be handled by the approval workflow
    return "Command execution handled by approval workflow"


    
checkpointer = InMemorySaver()

llm_with_tools = llm.bind_tools([shell_tool])

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
You are an expert Linux Automation Agent with intelligent self-correction capabilities. Your mission is to efficiently fulfill user requests by executing appropriate shell commands. Your primary working directory is `/home/pdp28`.

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
    print(response)
    state["messages"].append(AIMessage(content=response.content, tool_calls=response.tool_calls, invalid_tool_calls=response.invalid_tool_calls ))
    # print(state)
    return {"messages" : state["messages"]}



def if_tool_call(state : Tool_state) -> str:
    messages = state["messages"][-1]

    if messages.tool_calls:
        return "tool_node"
    else:
        return "end"

def tool_node(state : Tool_state) -> Tool_state:
    messages = state["messages"][-1]

    tool_by_name = {t.name : t for t in [shell_tool]}
    if messages.tool_calls:
        for tool_call in messages.tool_calls:
            # print(tool_by_name.get(tool_call["name"]))

            tool_name = tool_call["name"]
            response = tool_by_name.get(tool_name).invoke(tool_call["args"])
            state["messages"].append(ToolMessage(content=str(response), tool_call_id=tool_call["id"]))
        return {"messages" : state["messages"]}
    
state_graph = StateGraph(Tool_state)

state_graph.add_node("llm_node", llm_node)
state_graph.add_node("tool_node", tool_node)
state_graph.add_node("refactor_tool", refactor_tool)
state_graph.add_node("approval_node", approval_node)
state_graph.add_node("execute_command_node", execute_command_node)
state_graph.add_node("skip_command_node", skip_command_node)

state_graph.add_edge(START, "refactor_tool")
state_graph.add_edge("refactor_tool", "llm_node")
state_graph.add_conditional_edges("llm_node", if_tool_call, {"tool_node": "tool_node", "end": END})
state_graph.add_edge("tool_node", "approval_node")
state_graph.add_conditional_edges(
    "approval_node",
    lambda x: x.get("approval_status", "rejected"),
    {"approved": "execute_command_node", "rejected": "skip_command_node"}
)
state_graph.add_edge("execute_command_node", "llm_node")
state_graph.add_edge("skip_command_node", "llm_node")

graph = state_graph.compile(checkpointer=checkpointer)



def main():
    running = True
    config = {"configurable": {"thread_id": "1"}}

    print("📝 Agent started. Type 'no' to exit\n")

    while running:
        print("\n" + "="*60)
        user_input = input("Enter your input | Enter 'no' to exit: ").strip()
        if user_input.lower() == "no":
            print("Bye!")
            break

        if not user_input:
            continue

        # Invoke the graph
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "pending_command": None,
            "approval_status": None
        }

        # Handle graph execution with potential interrupts
        while True:
            res = graph.invoke(input_state, config)

            # Check if there's an interrupt (approval needed)
            if "__interrupt__" in res and res["__interrupt__"]:
                interrupt_info = res["__interrupt__"][0].value
                print(f"\n❓ {interrupt_info['question']}")
                print(f"   Command: {interrupt_info['command']}")

                approval = input("👤 Approve? (yes/no): ").strip().lower()
                decision = approval in ["yes", "y"]

                # Resume with decision
                res = graph.invoke(Command(resume=decision), config)

            # No more interrupts
            if "__interrupt__" not in res or not res["__interrupt__"]:
                break

        print("\n✅ Agent Response:")
        print("-"*60)
        # Only print the final AIMessage (last message with content and no pending tool calls)
        for msg in reversed(res["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                print(msg.content)
                break
    
    
if __name__ == "__main__":
    main()