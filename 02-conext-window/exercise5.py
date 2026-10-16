import json
import os
import ollama

MODEL_NAME = "qwen2.5-coder:7b"
CHAT_FILE = "chat_history.json"


def load_chat_history() -> list:
  """Loads chat history from JSON file if it exists, otherwise returns empty list."""
  if os.path.exists(CHAT_FILE):
    try:
      with open(CHAT_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
        print(f"📂 [Loaded previous chat history from {CHAT_FILE}]")
        return history
    except Exception as e:
      print(f"⚠️ [Could not load chat file: {e}]")
  return []


def save_chat_history(history: list):
  """Saves current conversation history to a JSON file."""
  try:
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
      json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"💾 [Chat history saved to {CHAT_FILE}]")
  except Exception as e:
    print(f"⚠️ [Failed to save chat history: {e}]")


def main():
  print(f"Starting Persistent Chatbot (Model: {MODEL_NAME})")
  print("Commands: Type message, '/reset' to clear memory, or 'quit' to save & exit.\n")

  system_prompt = {
      "role": "system",
      "content": "You are a helpful and concise AI assistant.",
  }

  # Load history from file or initialize empty list
  history = load_chat_history()

  while True:
    try:
      user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
      print("\nSaving and exiting...")
      save_chat_history(history)
      break

    if not user_input:
      continue

    if user_input.lower() == "quit":
      save_chat_history(history)
      print("Exiting chat. Goodbye!")
      break

    if user_input.lower() == "/reset":
      history = []
      if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)
      print("🔄 Conversation history and saved file have been reset.")
      continue

    # Append user message
    history.append({"role": "user", "content": user_input})

    # Keep active payload clean
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

      # Append assistant response
      history.append({"role": "assistant", "content": full_response})

    except Exception as e:
      print(f"\n[Error communicating with Ollama: {e}]")


if __name__ == "__main__":
  main()