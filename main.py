from langchain_openrouter import ChatOpenRouter
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


llm = ChatOpenRouter(
    model="stepfun/step-3.5-flash:free",
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
    # print(results)
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
                "system",n
                """### SYSTEM ROLE
You are an expert Linux Automation Agent with intelligent self-correction capabilities. Your mission is to efficiently fulfill user requests by executing appropriate shell commands. Your primary working directory is `/home/pdp28`.

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

    # res = graph.invoke({
    #     "messages": messages
    # }, config)
    for chunk in graph.stream(
    {
        "messages": messages,
    },
    stream_mode=["updates"],
    config=config
):
        print(chunk)

    # messages = res["messages"]

    # print("\n✅ Agent Response:")
    # print("-"*60)
    # for msg in res["messages"]:
    #     if hasattr(msg, 'content') and msg.content:
    #         print(msg.content)
    
    
if __name__ == "__main__":
    main()