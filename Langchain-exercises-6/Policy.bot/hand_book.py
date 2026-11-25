
from langchain_core.tools import tool
from langchain.agents import create_agent
from datetime import date, datetime

from rag import search_policies


# Tool 1: Calculate days until a given date
@tool
def days_until(date_string: str):
    """Calculate the number of days between today and a given date."""

    target_date = datetime.strptime(date_string, "%Y-%m-%d").date()
    today = date.today()

    return (target_date - today).days


# Register both tools
tools = [search_policies, days_until]


# Get today's date
today = date.today()

# Create the agent
agent = create_agent(
    model="ollama:qwen3.5:4b",  
    tools=tools,
    system_prompt=f"""
    You are a helpful company handbook assistant.

    Today's date is {today}.
    
    When the user mentions a relative date, calculate the
    exact date using today's date.

    Use days_until to calculate the number of days.

    Use search_policies to find relevant company policies.

    Use the results from both tools to provide one clear,
    accurate answer.

    Do not invent company policies.
    If the handbook does not contain the required information,
    say that you don't know.
    """
)


# Ask the agent a question
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I want to go on vacation on the 1st of next month. How many days is that from now, and how early do I have to request it?"
        }
    ]
})


# Print the final answer
final_message = result["messages"][-1]

print("\n--- Agent Messages ---")

for message in result["messages"]:
    print("\n", type(message).__name__)
    print("Content:", message.content)

    if hasattr(message, "tool_calls"):
        print("Tool calls:", message.tool_calls)