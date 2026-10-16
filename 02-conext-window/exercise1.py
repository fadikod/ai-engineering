import ollama

MODEL_NAME = "qwen2.5-coder:7b"


def count_tokens(text: str) -> int:
  """Sends text to the model and returns prompt_eval_count."""
  response = ollama.generate(model=MODEL_NAME, prompt=text)
  # Ollama returns prompt_eval_count directly in the response dictionary
  return response.get("prompt_eval_count", 0)


def main():
  # 1. Paragraph in English
  english_text = (
      "Artificial intelligence engineering involves building practical"
      " applications using large language models, managing context windows,"
      " and optimizing token efficiency for local and cloud deployments."
  )

  # 2. The same paragraph in Arabic (native language)
  arabic_text = (
      "تنطوي هندسة الذكاء الاصطناعي على بناء تطبيقات عمليّة باستخدام نماذج"
      " اللغة الكبيرة، وإدارة نوافذ السياق، وتحسين كفاءة الرموز للنشر المحلي"
      " والسحابي."
  )

  # 3. Exactly 20 lines of Python code
  python_code = """import os
import sys
from datetime import datetime

def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

class TokenAnalyzer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.history = []

    def track(self, text):
        length = len(text)
        self.history.append((text, length))
        return length

if __name__ == "__main__":
    analyzer = TokenAnalyzer("qwen2.5-coder:7b")
    print("Starting token analysis...")
"""

  samples = {
      "English Paragraph": english_text,
      "Arabic Paragraph": arabic_text,
      "20 Lines Python Code": python_code,
  }

  print(f"Running Token Counter Analysis using model: {MODEL_NAME}\n")
  print(f"{'Sample Type':<25} | {'Chars':<6} | {'Tokens':<8} | {'Tokens/Char'}")
  print("-" * 60)

  results = {}
  for name, text in samples.items():
    char_count = len(text)
    token_count = count_tokens(text)
    ratio = token_count / char_count if char_count > 0 else 0
    results[name] = ratio

    print(f"{name:<25} | {char_count:<6} | {token_count:<8} | {ratio:.4f}")

  print("-" * 60)
  most_expensive = max(results, key=results.get)
  print(f"\n👉 Highest tokens per character: **{most_expensive}**")


if __name__ == "__main__":
  main()