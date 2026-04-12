from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from typing import List, TypedDict, Literal
from langgraph.types import interrupt, Command
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
from rich.console import Console
from langchain_community.tools import DuckDuckGoSearchRun
load_dotenv()

class Tool_state(TypedDict):
    messages : List
    allow_all : bool

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3
)

duckduckgo_search = DuckDuckGoSearchRun()


@tool
def human_in_loop(question : str):
    """This helps when llm to ask some recommandations to the human and then execute or when you need to wait for users next step
    Example : 
    "question":"What i need to do next"
    """
    print("Question ", question)
    reply = interrupt({
        "question" : question
    })
    
    return reply

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
    return "\n".join(results)
    
checkpointer = InMemorySaver()
    
llm_with_tools = llm.bind_tools([shell_tool, duckduckgo_search, human_in_loop])

def llm_node(state : Tool_state) -> Tool_state:
    messages = state["messages"]
    allow = state.get("allow", False)

    if len(messages) > 11:
        messages = [messages[0]] + messages[-10:]

    print("\n⏳ Agent is processing your request...")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """### SYSTEM ROLE
You are an expert Linux Automation Agent with intelligent self-correction capabilities. Your mission is to efficiently fulfill user requests by executing appropriate shell commands or what the user asked for. Your primary working directory is `/home/pdp28`.
You can also use human_in_loop tool to interrupt and ask questions to the user when needed.

CRITICAL: When you need to wait for the user's next step or need user input, you MUST call the human_in_loop tool. This is how you pause and wait for the user.

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
   - When user told you to wait or you have some doubt about next step, CALL the human_in_loop tool with your question. The user will answer and you can proceed with further steps.

5. **Error Handling**:
   - Timeout errors: Increase timeout or reduce command scope.
   - Permission errors: Check file ownership, group membership, or sudo requirements.
   - Missing dependencies: Install them or suggest alternatives.
   - Environment errors: Verify PATH, environment variables, and active virtual environments.

User Request → Path Validation → Command Execution → Output Analysis → Verification → (Errors? Self-Correct Loop) → Final Clean Summary"""
            ),
            ("placeholder", "{input}")
        ]
    )

    chain = prompt | llm_with_tools
    response = chain.invoke({"input":messages})
    print(response)
    state["messages"].append(AIMessage(content=response.content, tool_calls=response.tool_calls, invalid_tool_calls=response.invalid_tool_calls ))
    return {"messages" : state["messages"], "allow": allow}

def if_tool_call(state : Tool_state) -> str:
    messages = state["messages"][-1]

    if messages.tool_calls:
        return "tool_node"
    else:
        return "end"

def refactor_node(state : Tool_state):
    human_msg = state["messages"][-1]
    
    print(human_msg)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Rewrite the user's input into one short execution prompt for the next model.

If conversational, return a brief friendly reply instruction.
If technical, return a direct action prompt that tells the next model exactly what to do.
Use concrete shell/GitHub steps when relevant.
Return only the rewritten instruction text."""),
            ("human", "{input}")

        ]
    )
    chain = prompt | llm
    res = HumanMessage(content=chain.invoke({"input":human_msg.content}).content)
    return { "messages" : [res]}

def tool_node(state : Tool_state) -> Tool_state:
    messages = state["messages"][-1]
    allow_all = state.get("allow_all", False)

    tool_by_name = {t.name : t for t in [shell_tool, duckduckgo_search, human_in_loop]}
    if messages.tool_calls:
        for tool_call in messages.tool_calls:

            tool_name = tool_call["name"]
            if tool_name == "shell_tool" and not allow_all:
                cmd_list = tool_call["args"].get("commands", [])
                cmd_preview = cmd_list[0] if cmd_list else "the requested command(s)"
                approval = interrupt({
                    "question": "Are you okay to run, type 'allow all' to allow all the commands or type 'yes' to allow single command" ,
                    "query": cmd_preview,
                })
                if str(approval).strip().lower() not in {"y", "yes", "allow all", "allow", "allow al"}:
                    state["messages"].append(
                        ToolMessage(content=f"Skipped by user: {cmd_preview}", tool_call_id=tool_call["id"])
                    )
                    continue

                
                if str(approval).strip().lower() in {"allow all", "allow", "allow al"}:
                    allow_all = True
                print(f"\n⏳Executing {cmd_preview}")
            response = tool_by_name.get(tool_name).invoke(tool_call["args"])
            state["messages"].append(ToolMessage(content=str(response), tool_call_id=tool_call["id"]))
        return {"messages" : state["messages"], "allow_all" : allow_all}
    
state_graph = StateGraph(Tool_state)

state_graph.add_node("llm_node", llm_node)
state_graph.add_node("tool_node", tool_node)
state_graph.add_node("refactor_node", refactor_node)

state_graph.add_edge(START, "refactor_node")
state_graph.add_edge("refactor_node", "llm_node")
state_graph.add_conditional_edges("llm_node", if_tool_call,{"tool_node":"tool_node", "end":END})
state_graph.add_edge("tool_node", "llm_node")

graph = state_graph.compile(checkpointer=checkpointer)
# print(graph.get_graph().draw_mermaid())


console = Console()

def main():

    running = True
    config = {"configurable": {"thread_id": "1"}}

    while running:
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
            "messages": messages,
            "allow":False
        }, config)
        
        while True:
            snapshot = graph.get_state(config)
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                interrupt_value = snapshot.tasks[0].interrupts[0].value
                print(console.print(interrupt_value.get("question",""), style="bold green") ,  console.print(interrupt_value.get("query", ""), style="bold red")  ,"(y/n) : ", end="")
                input_human = input()

                res = graph.invoke(Command(resume=input_human), config=config)
                if input_human.lower() == "n":
                    print("Rejected by user")
                    break
                for msg in res["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        print(msg.content)
            else :
                print("="*100)
                print("\n✅ Agent Response:")
                print(res["messages"][-1].content)
                print("Work done !")
                break   
    
if __name__ == "__main__":
    main()