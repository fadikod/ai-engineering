from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOllama(
    model="qwen2.5-coder:7b",
    reasoning=False,
    temperature=0
)

prompt = ChatPromptTemplate([
    ("system", "you are a student assistant, you will help the student to learn the topic"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "can you explain this topic:{input}")
])

chain = prompt | llm

histories = {}

while True:

    name = input("Enter your name: ")

    if name.lower() == "exit":
        break

    if name not in histories:
        histories[name] = []

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "switch":
            break

        response = chain.invoke({
            "history": histories[name],
            "input": user_input
        })

        print("Assistant:", response.content)

        histories[name].append(HumanMessage(content=user_input))
        histories[name].append(AIMessage(content=response.content))