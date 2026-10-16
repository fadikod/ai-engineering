import ollama

MODEL_NAME = "qwen2.5-coder:7b"


def run_fact_test(num_facts: int):
  """Constructs a prompt with N facts where Fact 3 holds the secret code."""
  prompt_lines = []
  for i in range(1, num_facts + 1):
    if i == 3:
      prompt_lines.append(
          f"Fact {i}: The secret security access code is ALPHA-999-OMEGA."
      )
    else:
      prompt_lines.append(
          f"Fact {i}: Department asset tracking ID number is {7000 + i}."
      )

  prompt_lines.append(
      "\nQuestion: What is Fact 3 / the secret security access code?"
  )
  full_prompt = "\n".join(prompt_lines)

  # Send prompt with num_ctx set to 4096
  response = ollama.generate(
      model=MODEL_NAME, prompt=full_prompt, options={"num_ctx": 4096}
  )

  tokens_read = response.get("prompt_eval_count", 0)
  answer = response.get("response", "").strip()
  return tokens_read, answer


def main():
  print(
      f"Running 'Find the limit' experiment using model: {MODEL_NAME}\n"
  )
  print(f"{'Facts':<8} | {'Tokens Read':<12} | {'Model Answer Preview'}")
  print("-" * 75)

  # Test incremental scales of facts to observe where recall holds or degrades
  test_levels = [10, 50, 100, 250, 500]

  for count in test_levels:
    tokens, answer = run_fact_test(count)
    # Clean up answer for single line printing
    clean_ans = answer.replace("\n", " ")
    print(f"{count:<8} | {tokens:<12} | {clean_ans}")

  print("-" * 75)


if __name__ == "__main__":
  main()