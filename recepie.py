from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, List, Dict
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
# Define a simple state structure
class RecipeState(TypedDict):
    ingredients: List[str]
    recipe_name: str
    recipe_steps: List[str]
    approved: bool
    notes: List[str]
    
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3
)

def generate_recipe(state: RecipeState) -> RecipeState:
    print("\n🍳 Generating recipe...")
    ingredients_text = ", ".join(state["ingredients"])
    prompt = f"""
    Create a recipe using these ingredients: {ingredients_text}.
    
    Please provide:
    1. A creative recipe name
    2. A list of 3-5 cooking steps

    Format your response as:
    Recipe Name: [name]
    Steps:
    - Step 1
    - Step 2
    - Step 3
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    try:
        name_section = content.split("Recipe Name:")[1].split("Steps:")[0].strip()
        steps_section = content.split("Steps:")[1].strip()
        recipe_name = name_section
        recipe_steps = [step.strip().lstrip("- ") for step in steps_section.split("\n") if step.strip()]
    except:
        recipe_name = "Mixed Ingredient Recipe"
        recipe_steps = ["Combine all ingredients", "Cook until done", "Serve and enjoy"]

    print(f"✅ Generated: {recipe_name}")
    return {**state, "recipe_name": recipe_name, "recipe_steps": recipe_steps}

def review_recipe(state: RecipeState) -> RecipeState:
    print("\n📋 RECIPE REVIEW")
    print(f"Recipe: {state['recipe_name']}")
    
    print("\nIngredients:")
    for ingredient in state["ingredients"]:
        print(f"- {ingredient}")

    print("\nSteps:")
    for i, step in enumerate(state["recipe_steps"]):
        print(f"{i+1}. {step}")

    approval = input("\nDo you approve this recipe? (yes/no): ").strip().lower() == "yes"
    notes = input("Any notes? ").strip()

    return {
        **state,
        "approved": approval,
        "notes": state["notes"] + ([notes] if notes else [])
    }
    
def refine_recipe(state: RecipeState) -> Dict:
    if state["approved"]:
        return state

    print("\n🔄 Refining recipe based on feedback...")
    ingredients_text = ", ".join(state["ingredients"])
    notes_text = "\n".join(state["notes"])

    prompt = f"""
    Please improve this recipe based on the feedback:

    Original Recipe: {state["recipe_name"]}
    Ingredients: {ingredients_text}
    Original Steps:
    {chr(10).join(f"- {step}" for step in state["recipe_steps"])}

    Feedback:
    {notes_text}

    Create an improved version with:
    1. A better recipe name
    2. Improved cooking steps

    Format your response as:
    Recipe Name: [name]
    Steps:
    - Step 1
    - Step 2
    - Step 3
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    try:
        name_section = content.split("Recipe Name:")[1].split("Steps:")[0].strip()
        steps_section = content.split("Steps:")[1].strip()
        recipe_name = name_section
        recipe_steps = [step.strip().lstrip("- ") for step in steps_section.split("\n") if step.strip()]
    except:
        recipe_name = state["recipe_name"] + " (Improved)"
        recipe_steps = state["recipe_steps"]

    print(f"✅ Refined: {recipe_name}")
    return {
        **state,
        "recipe_name": recipe_name,
        "recipe_steps": recipe_steps,
        "approved": False
    }
    
def save_recipe(state: RecipeState) -> RecipeState:
    if state["approved"]:
        print("\n✅ Recipe saved successfully!")
        print(f"\nFinal Recipe: {state['recipe_name']}")
        print("\nIngredients:")
        for ingredient in state["ingredients"]:
            print(f"- {ingredient}")
        print("\nSteps:")
        for i, step in enumerate(state["recipe_steps"]):
            print(f"{i+1}. {step}")
    else:
        print("\n❌ Recipe not approved - sending back for refinement.")

    return state

builder = StateGraph(RecipeState)

builder.add_node("generate_recipe", generate_recipe)
builder.add_node("review_recipe", review_recipe)
builder.add_node("refine_recipe", refine_recipe)
builder.add_node("save_recipe", save_recipe)

builder.set_entry_point("generate_recipe")
builder.add_edge("generate_recipe", "review_recipe")
builder.add_conditional_edges(
    "review_recipe",
    lambda state: "save_recipe" if state["approved"] else "refine_recipe"
)
builder.add_edge("refine_recipe", "review_recipe")
builder.add_edge("save_recipe", END)

recipe_graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["review_recipe"],
    interrupt_after=["generate_recipe", "refine_recipe"]
)

print(recipe_graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("\n===== LLM RECIPE GENERATOR WITH LANGGRAPH =====")
    print("This example demonstrates LangGraph with LLM-powered recipe generation")

    print("\nEnter ingredients (comma separated):")
    ingredients_input = input("> ")
    ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]

    if not ingredients:
        ingredients = ["chicken", "broccoli", "soy sauce"]
        print(f"Using default ingredients: {ingredients}")

    initial_state = {
        "ingredients": ingredients,
        "recipe_name": "",
        "recipe_steps": [],
        "approved": False,
        "notes": []
    }

    thread_id = "recipe_thread_1"

    print("\n🚀 Starting recipe workflow with ingredients:", initial_state["ingredients"])

    # Start and interrupt after generation
    result = recipe_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )

    print("\n⏸️ INTERRUPT AFTER generate_recipe")
    print(f"Generated recipe: {result['recipe_name']}")
    input("\nPress Enter to continue to review...")

    # Continue to review
    result = recipe_graph.invoke(None, config={"configurable": {"thread_id": thread_id}})

    # Loop until approved
    while True:
        result = recipe_graph.invoke(None, config={"configurable": {"thread_id": thread_id}})
        if result.get("approved", False):
            break

        if "recipe_steps" in result:
            print("\n⏸️ INTERRUPT AFTER refine_recipe")
            print(f"Refined recipe: {result['recipe_name']}")
            input("\nPress Enter to continue to review...")

            result = recipe_graph.invoke(None, config={"configurable": {"thread_id": thread_id}})

    print("\n===== WORKFLOW COMPLETED =====")
    print("Thank you for using the LLM Recipe Generator!")