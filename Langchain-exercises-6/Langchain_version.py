# 1. Imports
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# 2. Initialize the checkpointer
# This saves conversation states in memory while the program runs.
checkpointer = InMemorySaver()


# 3. Our todo list (application data)
tasks = []


# 4. Define the tools

@tool
def add_task(title: str):
    """Add a new task to the todo list."""
    task = {"title": title, "completed": False}
    tasks.append(task)
    return f"Task '{title}' has been added."


@tool
def list_tasks():
    """List all available tasks and their completion status."""
    return tasks


@tool
def complete_task(title: str):
    """Mark an existing task as completed."""
    for task in tasks:
        if task["title"] == title:
            task["completed"] = True
            return f"{title} is completed."

    return "There is no such a task."


# 5. Initialize the model
llm = ChatOllama(
    model="qwen3.5:4b",
    reasoning=False,
    temperature=0
)


# 6. Collect the available tools
tools = [add_task, list_tasks, complete_task]


# 7. Create the agent
# The checkpointer allows the agent to remember previous messages.
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful todo assistant. Use the available tools to manage the user's todo list.",
    checkpointer=checkpointer
)


# 8. Start the conversation loop
while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Invoke the agent with the user's message.
    # The thread_id identifies this particular conversation.
    # Using the same ID allows the agent to retrieve previous messages.
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        },
        config={
            "configurable": {
                "thread_id": "todo-chat"
            }
        }
    )

    # Display the agent's latest response.
    print("Assistant:", result["messages"][-1].content)
