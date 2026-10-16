import ollama

MODEL_NAME = "qwen2.5-coder:7b"


def main():
  print(f"Starting Terminal Chatbot with Memory (Model: {MODEL_NAME})")
  print("Commands: Type your message, '/reset' to clear memory, or 'quit' to exit.\n")

  # Initialize conversation history with a custom system prompt
  messages = [{
      "role": "system",
      (
          " content"
      ): "You are an expert, concise, and friendly AI engineering mentor.",
  }]

  while True:
    try:
      user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
      print("\nExiting chat. Goodbye!")
      break

    if not user_input:
      continue

    if user_input.lower() == "quit":
      print("Exiting chat. Goodbye!")
      break

    if user_input.lower() == "/reset":
      # Keep the system prompt (index 0) and clear the rest of the history
      messages = [messages[0]]
      print("🔄 Conversation history has been reset.")
      continue

    # Append user message to history
    messages.append({"role": "user", "content": user_input})

    # Keep only the system prompt + last 10 messages of conversation history
    # If history exceeds system prompt + 10 messages, slice it
    system_prompt = messages[0]
    conversation_history = messages[1:]
    if len(conversation_history) > 10:
      conversation_history = conversation_history[-10:]
    messages = [system_prompt] + conversation_history

    print("AI: ", end="", flush=True)

    try:
      # Stream the response from the model
      stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)

      full_response = ""
      for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        full_response += content
      print()

      # Append assistant response to history
      messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
      print(f"\n[Error communicating with Ollama: {e}]")


if __name__ == "__main__":
  main()