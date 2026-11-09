
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    reasoning=False,
    temperature=0
)

# Prompt templates

prompt_title = ChatPromptTemplate([
    ("system", "You are a title generator. Return only the title."),
    ("human", "Generate a title for this topic: {topic}")
])

prompt_outline = ChatPromptTemplate([
    ("system", "You are an outline generator. Return only the outline."),
    ("human", "Generate an outline for this title: {title}")
])

prompt_introduction = ChatPromptTemplate([
    ("system", "You are an introduction generator. Return only the introduction."),
    ("human", "Generate an introduction for this outline: {outline}")
])

# Individual chains

chain_title = prompt_title | llm | StrOutputParser()
chain_outline = prompt_outline | llm | StrOutputParser()
chain_introduction = prompt_introduction | llm | StrOutputParser()


# 3. Separate invocation 

# title_result = chain_title.invoke({"topic": "The impact of artificial intelligence on modern healthcare"})
# print(title_result.content)

# outline_result = chain_outline.invoke({"title": title_result.content})
# print(outline_result.content)

# introduction_result = chain_introduction.invoke({"outline": outline_result.content})
# print(introduction_result.content)

# Connected pipeline

blog_pipeline = (
    {
        "title": chain_title,
        "topic": lambda x: x["topic"]
    }
    | {
        "title": lambda x: x["title"],
        "outline": chain_outline
    }
    | {
        "title": lambda x: x["title"],
        "outline": lambda x: x["outline"],
        "introduction": chain_introduction
    }
)

# Invoke the pipeline

result = blog_pipeline.invoke({
    "topic": "The impact of artificial intelligence on modern healthcare"
})

# Print every step

print("TITLE:")
print(result["title"])

print("\nOUTLINE:")
print(result["outline"])

print("\nINTRODUCTION:")
print(result["introduction"])
