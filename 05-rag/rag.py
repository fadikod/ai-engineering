
import chromadb
import requests


# Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")

# Get the existing collection
collection = client.get_collection(
    name="company_policies"
)


# Retrieve relevant policy chunks
def retrieve(question):

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # Get the distance of the closest result
    distance = results["distances"][0][0]

    # Check if the result is relevant enough
    if distance > 1.1:
        return None

    return results


# Generate an answer using Qwen through Ollama
def generate_answer(question, results):

    # Combine retrieved chunks into one context
    context = "\n\n".join(results["documents"][0])

    # Prepare the prompt
    prompt = f"""
    You are an assistant for Nexora Technologies.

    Answer the user's question using only the policy information
    provided below.

    If the information does not contain the answer, say that
    you don't have enough information to answer.

    Policy information:
    {context}

    Question:
    {question}

    Answer:
    """

    # Send the prompt to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5-coder:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result["response"]


# Main RAG function
def answer(question):

    # Retrieve relevant chunks
    results = retrieve(question)

    # Refuse if no relevant information was found
    if results is None:
        return (
            "I couldn't find this in the company documents.",
            []
        )

    # Generate an answer using the retrieved information
    response = generate_answer(question, results)

    # Extract source documents and sections
    sources = [
        f'{item["source"]} ({item["section"]})'
        for item in results["metadatas"][0]
    ]

    return response, sources


# Test the RAG pipeline
if __name__ == "__main__":

    response, sources = answer(
        "How many vacation days do employees get?"
    )

    print("Answer:", response)
    print("Sources:", sources)
