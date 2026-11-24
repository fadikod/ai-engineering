from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool, tool
from langchain.agents import create_agent


tasks = []

@tool
def add_task(title:str):
    """this function adds a task"""
    task = {"title": title, "completed": False}
    tasks.append(task)
    
@tool
def list_tasks():
    """this funciton list all of the tasks available and their completion status"""
    result = []
    for task in tasks:
        result.append(( f"The task is {task['title']}and its status is {task['completed']}"))
    return result
     
@tool
def complete_task(title: str):
    """this function makes the status of a task completed"""
    for task in tasks:
        if task['title']  == title:
            task['completed'] = True
            return(f"{title} is completed")
            
    else:
        return("there is no such a task")
        
        
        
llm = ChatOllama(
    model="qwen3.5:4b",
    reasoning = False,
    Temperature = 0
)

tools = [add_task, list_tasks, complete_task]


Tools = {
    "add_task": add_task,
    "list_tasks": list_tasks,
    "complete_tasks": complete_task
}

llm_with_tools = llm.bind_tools(tools)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful todo assistant. Use the available tools to manage the user's todo list."
)

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })

    print("Assistant:", result["messages"][-1].content)
    
        
        
        
