import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5-coder:7b",
        "prompt": "Explain RAG in one sentence.",
        "stream": False
    }
)

result = response.json()

print(result["response"])