
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# 1. Initialize the model
llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0
)

# 2. Define the tools

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract the second number from the first."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: int, b: int) -> float | str:
    """Divide the first number by the second."""
    if b == 0:
        return "Error: division by zero"
    return a / b


# 3. Create a list of available tools
tools = [add, subtract, multiply, divide]

# 4. Create a dictionary to access tools by name
TOOLS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide
}

# 5. Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

# 6. Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a calculator assistant.
    Use the available tools to perform calculations.
    Break complex calculations into smaller steps.
    Return the final answer when finished."""),
    ("human", "{question}")
])

# 7. Prepare the conversation
messages = prompt.invoke({
    "question": "What is (12 * 7) - (100 / 4)?"
}).messages

# 8. Start the agent loop
while True:

    # Ask the model what to do next
    response = llm_with_tools.invoke(messages)

    # Add the model's response to the conversation
    messages.append(response)

    # Check whether the model requested any tools
    if not response.tool_calls:
        print("Final answer:", response.content)
        break

    # Execute each requested tool
    for tool_call in response.tool_calls:

        name = tool_call["name"]
        args = tool_call["args"]

        print(f"Tool call: {name}({args})")

        # Retrieve and execute the correct tool
        selected_tool = TOOLS[name]
        result = selected_tool.invoke(args)

        print("Tool result:", result)

        # Send the tool result back to the model
        messages.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call["id"]
        })
