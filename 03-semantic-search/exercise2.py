import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> np.ndarray:
    """Generates a vector embedding for a given text string using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes the cosine similarity between two numpy vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def main():
    # 6 sentences: 2 sports, 2 food, 2 programming
    sentences = [
        "The football team scored a dramatic winning goal in the final minute.",  # Sports 1
        "Fans cheered wildly as the basketball player dunked the ball.",  # Sports 2
        "Freshly baked sourdough bread pairs wonderfully with olive oil.",  # Food 1
        "The spicy Thai curry was loaded with fresh vegetables and herbs.",  # Food 2
        "Python functions help keep code modular, clean, and reusable.",  # Programming 1
        "Debugging asynchronous JavaScript code requires careful handling of promises.",  # Programming 2
    ]

    print(
        f"=== Generating Embeddings for Similarity Matrix using [{EMBED_MODEL}] ==="
    )
    embeddings = [get_embedding(s) for s in sentences]

    print("\n--- Similarity Matrix Table ---")
    # Print header row
    header = f"{'':<6}" + "".join([f"S{i+1:<8}" for i in range(len(sentences))])
    print(header)
    print("-" * (6 + 8 * len(sentences)))

    # Compute pairwise similarities and print matrix
    for i, sent_i in enumerate(sentences):
        row_str = f"S{i+1:<5} "
        for j, sent_j in enumerate(sentences):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row_str += f"{sim:.4f}  "
        print(row_str)

    print("\n--- Sentence Reference Guide ---")
    for idx, s in enumerate(sentences, start=1):
        print(f"S{idx}: {s}")


if __name__ == "__main__":
    main()