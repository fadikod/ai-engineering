
from rag import answer


print("Nexora Policy Assistant")
print("Ask me anything about company policies.")
print("Type 'exit' to quit.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Get the answer and sources
    response, sources = answer(question)

    # Display the answer
    print(f"\nAssistant: {response}")

    # Display the sources if available
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source}")
