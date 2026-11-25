
from rag import rag_chain

# Start the interactive question-answering loop
print("Company Policy Bot 2.0")
print("Ask questions about company policies.")
print("Type 'exit' to quit.\n")

while True:
    # Get a question from the user
    question = input("You: ")

    # Stop the application when the user types exit
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Skip empty questions
    if not question.strip():
        continue

    # Send the question through the RAG chain
    answer = rag_chain.invoke(question)

    # Display the generated answer
    print(f"\nAssistant: {answer}\n")
