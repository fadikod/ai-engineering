import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> np.ndarray:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def search_movies_with_threshold(query: str, movies: list[dict], movie_embeddings: list[np.ndarray], min_score: float = 0.6) -> tuple[dict | None, float]:
    """Searches movies and returns the best match only if it meets or exceeds min_score."""
    query_emb = get_embedding(query)
    
    best_score = -1.0
    best_movie = None

    for idx, movie in enumerate(movies):
        sim = cosine_similarity(query_emb, movie_embeddings[idx])
        if sim > best_score:
            best_score = sim
            best_movie = movie

    # Check against threshold
    if best_score < min_score:
        return None, best_score

    return best_movie, best_score


def main():
    movies = [
        {
            "title": "The Matrix",
            "desc": "A hacker discovers a reality controlled by machines and joins a rebellion.",
        },
        {
            "title": "WALL-E",
            "desc": "A lonely waste-collecting robot on a desolate Earth embarks on a space journey.",
        },
        {
            "title": "Interstellar",
            "desc": "A team of explorers travel through a wormhole in space to ensure survival.",
        },
        {
            "title": "Jurassic Park",
            "desc": "Scientists clone dinosaurs for a theme park, but security systems fail.",
        },
        {
            "title": "Titanic",
            "desc": "An aristocrat falls in love with an artist aboard the luxury ocean liner.",
        },
    ]

    print(f"=== Initializing Threshold Movie Finder using [{EMBED_MODEL}] ===")
    movie_embeddings = [get_embedding(m["desc"]) for m in movies]

    # Test queries: some movie-related, some completely unrelated (out-of-domain)
    test_queries = [
        "a movie about dinosaurs breaking out",  # Should match Jurassic Park
        "space travel and wormholes",            # Should match Interstellar
        "how do I bake a chocolate cake?",       # Out-of-domain (Should trigger threshold)
        "quantum physics equations in python",   # Out-of-domain (Should trigger threshold)
    ]

    # Set threshold (determined by testing what scores out-of-domain text gets)
    MIN_SCORE_THRESHOLD = 0.55

    print(f"\nUsing Minimum Score Threshold: {MIN_SCORE_THRESHOLD}\n")

    for i, query in enumerate(test_queries, start=1):
        print(f"Test {i}: Query = '{query}'")
        match, score = search_movies_with_threshold(query, movies, movie_embeddings, min_score=MIN_SCORE_THRESHOLD)

        if match is None:
            print(f"  ❌ No match found (Best score was {score:.4f} which is below threshold)")
        else:
            print(f"  ✅ Found Match: {match['title']} (Score: {score:.4f})")
            print(f"     Summary: {match['desc']}")
        print("-" * 60)


if __name__ == "__main__":
    main()