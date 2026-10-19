import numpy as np
import ollama

# Model used for generating embeddings
EMBED_MODEL = "nomic-embed-text"



def get_embedding(text: str) -> np.ndarray:
    """Generates a vector embedding for a given text string using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    # response['embedding'] returns a direct list of numbers for embeddings()
    return np.array(response["embedding"])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes the cosine similarity between two numpy vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def odd_one_out(words: list[str]) -> str:
    """Finds and returns the word that is least similar to the others on average."""
    # Step A: Get embeddings for all words in the list
    embeddings = {word: get_embedding(word) for word in words}

    avg_similarities = {}

    # Step B: Calculate average similarity for each word against all other words
    for target_word in words:
        similarities = []
        for other_word in words:
            if target_word != other_word:
                sim = cosine_similarity(
                    embeddings[target_word], embeddings[other_word]
                )
                similarities.append(sim)

        # Average similarity to the rest of the list
        avg_similarities[target_word] = np.mean(similarities)

    # Print debugging scores to see average similarities
    print("\nAverage Similarity Scores:")
    for word, score in avg_similarities.items():
        print(f"  - {word:<12}: {score:.4f}")

    # Step C: The odd one out is the word with the MINIMUM average similarity
    odd_word = min(avg_similarities, key=avg_similarities.get)
    return odd_word


def main():
    # 1. Mandatory test list from the prompt
    test1 = ["apple", "banana", "cherry", "laptop"]

    # 2. Custom Test List 1: Vehicles vs. Animal
    test2 = ["car", "truck", "motorcycle", "dog"]

    # 3. Custom Test List 2: Programming Languages vs. Food
    test3 = ["python", "javascript", "c++", "pizza"]

    # 4. Custom Test List 3: Countries vs. Space Object
    test4 = ["france", "germany", "japan", "moon"]

    test_cases = [test1, test2, test3, test4]

    print(f"=== Running 'The Odd One Out' Analysis using [{EMBED_MODEL}] ===")

    for idx, words in enumerate(test_cases, start=1):
        print(f"\nTest {idx}: Words = {words}")
        result = odd_one_out(words)
        print(f"👉 Odd one out: **{result}**")
        print("-" * 50)


if __name__ == "__main__":
    main()