from langchain_groq.chat_models import ChatGroq
from typing import List, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import ShellTool
from IPython.display import Image
import subprocess
from typing import List
from langgraph.checkpoint.memory import InMemorySaver  
load_dotenv()

class Tool_state(TypedDict):
    messages : List


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3
)

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
            results.append(output.stdout + output.stderr)

        except subprocess.TimeoutExpired:
            results.append("Error: Command timed out after 30 seconds")
        except Exception as e:
            results.append(f"Error: {str(e)}")
    print(results)
    return "\n".join(results)
    
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
You are a Linux Automation Agent with a self-correction loop. Your goal is to fulfill user requests by executing real shell commands. Your working directory is `/home/pdp28`, dont wait for any instructions from user other than sudo password.

### OPERATIONAL RULES
1. **Case-Insensitive Discovery**:
   - If a user mentions a path that doesn't exist, DO NOT fail.
   - Run `ls -F | grep -i "name"` or `find . -maxdepth 2 -iname "*name*"` to locate the correct directory.
   - Automatically `cd` into the matched path and continue the task.

2. **Self-Correction Loop (CRITICAL)**:
   - If a command returns an error (stderr) or an unexpected empty result, analyze the output.
   - Formulate a hypothesis on why it failed (e.g., "Permission denied", "Wrong path", "Syntax error").
   - Execute a corrected command immediately. Do not ask for permission to try a fix unless you have failed 3 times in a row.

3. **Validation**:
   - After creating a file or moving a directory, always run `ls` or `cat` to verify the action was successful before reporting to the user.

4. **Safety & Formatting**:
   - Use absolute paths when possible.
   - If a command requires `sudo`, explain why briefly in the final response.
   - Provide the final output in a clean summary; do not dump raw logs unless requested.
   - when you think the process is completed dont give the tool_calls leave it as tool_calls as empty

### EXECUTION FLOW
- Receive Goal -> Search/Navigate -> Execute -> Verify -> (If Error: Fix & Repeat) -> Final Answer."""
            ),
            ("placeholder", "{input}")
        ]
    )

    chain = prompt | llm_with_tools
    response = chain.invoke({"input":messages})

    state["messages"].append(response)
    return state

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
    return state
    
state_graph = StateGraph(Tool_state)

state_graph.add_node("llm_node", llm_node)
state_graph.add_node("tool_node", tool_node)
state_graph.add_edge(START, "llm_node")
state_graph.add_conditional_edges("llm_node", if_tool_call,{"tool_node":"tool_node", "end":END})
state_graph.add_edge("tool_node", "llm_node")

graph = state_graph.compile(checkpointer=checkpointer)



def main():

    running = True
    config = {"configurable": {"thread_id": "1"}}
    messages = []

    print("\n" + "="*60)
    print("Enter you input | Enter no to exit : ", end="")
    user_input = str(input())
    if user_input.lower() == "no":
        running = False
        print("Bye!")
        return

    messages.append(HumanMessage(content=user_input))

    res = graph.invoke({
        "messages": messages
    }, config)

    messages = res["messages"]

    print("\n✅ Agent Response:")
    print("-"*60)
    for msg in res["messages"]:
        if hasattr(msg, 'content') and msg.content:
            print(msg.content)
    
    
if __name__ == "__main__":
    main()