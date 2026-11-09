from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen2.5-coder:7b",
    reasoning = False,
    Temperature = 0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate into the language requested by the user."
     "return onlt the translation, do not include any other text"),
    ("human", "Translate the following text into {language}: {text}")
])

chain = prompt | llm

inputs = [
    {"text": "Hello, how are you", "language":"Dutch"},
    {"text": "Hello,how are you", "language": "German"},
    {"text": "Hello, how are you", "language": "Hungarian"},
    {"text": "Hello, how are you", "language": "Italian"},
    {"text": "Hello, how are you", "language": "Japanese"},
]

results = chain.batch(inputs)
for result in results:
    print(result.content)