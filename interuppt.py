from langchain_groq import ChatGroq
from langgraph.types import interrupt, Command
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from simple_chalk import chalk
load_dotenv()
# Using a valid Groq model name
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3
)

class State(TypedDict):
    input: str
    response: str

def abort(state: State):
    print("="*100)
    print()
    print("The final response is : ")
    print(state["response"])
    return {"response": "Aborted"}

def human_in_loop(state: State) -> Command[Literal["llm_node", "abort"]]:
    answer = interrupt(
        {
            "question": "Is it okay to continue ?",
            "res_preview": state["response"]
        }
    )
    
    if answer.lower() == "n":
        return Command(goto="abort")
    else:
        old_input = state["input"]
        new_input = f" old input : {old_input} ,new suggestions : {answer} " 
        return Command(update={"input" : new_input}, goto="llm_node") 

def llm_node(state: State):
    input_human = state["input"]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you a assistant to give content for the linkedin post, in the Gen Z tone, if the new suggestions are empty just update the content"),
            ("human", "{input}")
        ]
    )
    
    chain = prompt | llm
    response = chain.invoke({"input": input_human})
    return {"response": response.content}

graph = StateGraph(State)
graph.add_node("llm_node", llm_node)
graph.add_node("abort", abort)
graph.add_node("human", human_in_loop)

graph.add_edge(START, "llm_node")
graph.add_edge("llm_node", "human")


memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}
graph_com = graph.compile(checkpointer=memory)

def main():
    print("Enter your topic : ", end=" ")
    user_inp = str(input())
    graph_com.invoke({"input": user_inp}, config)
    
    while True:
         snapshot = graph_com.get_state(config)
         if snapshot.tasks and snapshot.tasks[0].interrupts:
            print("="*100)
            print(f"AI message: ")
            print()
            print(chalk.cyan(snapshot.tasks[0].interrupts[0].value["res_preview"]))
            print(f"{snapshot.tasks[0].interrupts[0].value["question"]} (y/n) ")
            print("If yes, give suggestions : ", end= "")
            
            user_input = str(input())
            
            graph_com.invoke(Command(resume=user_input ),config)
            
            if user_input.lower() == "n":
                return
            
    
if __name__ == "__main__":
    main()