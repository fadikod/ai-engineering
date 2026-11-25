
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from datetime import date, datetime

# Connect to the same embedding model used during ingestion
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load the existing Chroma database
vector_store = Chroma(
    collection_name="company_policies",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Create a retriever to find the 4 most relevant chunks
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# Configure the local language model
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)

# Define the prompt used to answer questions
prompt = ChatPromptTemplate.from_template("""
You are a company policy assistant.

Answer the question using ONLY the provided context.
If the answer is not available in the context, say that you don't know.

Include the sources at the end of your answer.
For each source, provide the filename and section name.

Context:
{context}

Question:
{question}
""")

# Format retrieved documents and their metadata into context
def format_docs(docs):
    formatted_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        section = doc.metadata.get("section", "Unknown")

        formatted_docs.append(
            f"Source: {source} | Section: {section}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


# Connect retrieval, prompt, LLM and output parser
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


@tool
def search_policies(question: str):
    """Search the company handbook for policies relevant to the question."""
    return rag_chain.invoke(question)


