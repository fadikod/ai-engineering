import ollama

MODEL_NAME = "qwen2.5-coder:7b"


def summarize_oldest_messages(messages_to_summarize: list) -> str:
  """Sends the oldest 6 messages to the model to get a concise 3-sentence summary."""
  prompt = (
      "Summarize the following conversation history in exactly 3 sentences.\n"
      "Preserve key personal information like user details, names, or specific facts mentioned:\n\n"
  )
  for msg in messages_to_summarize:
    prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

  response = ollama.generate(model=MODEL_NAME, prompt=prompt)
  return response.get("response", "").strip()


def main():
  print(f"Starting Chatbot with Summarized Memory (Model: {MODEL_NAME})")
  print("Commands: Type message, '/reset' to clear memory, or 'quit' to exit.\n")

  # System prompt setup
  system_prompt = {
      "role": "system",
      "content": "You are a helpful, friendly AI assistant with long-term context tracking.",
  }
  history = []

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
      history = []
      print("🔄 Conversation history has been reset.")
      continue

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Check if history exceeds 10 messages
    if len(history) > 10:
      print("\n⚡ [Memory threshold reached (>10 messages). Summarizing oldest 6 messages...]")
      oldest_6 = history[:6]
      remaining = history[6:]

      summary_text = summarize_oldest_messages(oldest_6)
      print(f"📌 [Summary generated]: {summary_text}\n")

      # Replace the 6 messages with one summary message
      summary_message = {
          "role": "system",
          "content": f"Prior conversation summary: {summary_text}",
      }
      history = [summary_message] + remaining

    # Prepare complete payload including main system prompt + active history
    messages_payload = [system_prompt] + history

    print("AI: ", end="", flush=True)

    try:
      stream = ollama.chat(
          model=MODEL_NAME, messages=messages_payload, stream=True
      )
      full_response = ""
      for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        full_response += content
      print()

      # Add AI response to history
      history.append({"role": "assistant", "content": full_response})

    except Exception as e:
      print(f"\n[Error communicating with Ollama: {e}]")


if __name__ == "__main__":
  main()