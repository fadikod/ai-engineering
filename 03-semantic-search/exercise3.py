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
    # 1. List of 15 movies with one-sentence descriptions
    movies = [
        {
            "title": "The Matrix",
            "desc": (
                "A hacker discovers a reality controlled by machines and joins"
                " a rebellion."
            ),
        },
        {
            "title": "WALL-E",
            "desc": (
                "A lonely waste-collecting robot on a desolate Earth embarks"
                " on a space journey that alters the fate of mankind."
            ),
        },
        {
            "title": "Interstellar",
            "desc": (
                "A team of explorers travel through a wormhole in space in an"
                " attempt to ensure humanity's survival."
            ),
        },
        {
            "title": "Jurassic Park",
            "desc": (
                "Scientists clone dinosaurs for a theme park, but security"
                " systems fail and chaos ensues."
            ),
        },
        {
            "title": "Titanic",
            "desc": (
                "A seventeen-year-old aristocrat falls in love with a kind"
                " artist aboard the luxurious, ill-fated R.M.S. Titanic."
            ),
        },
        {
            "title": "The Godfather",
            "desc": (
                "The aging patriarch of an organized crime dynasty transfers"
                " control of his clandestine empire to his reluctant son."
            ),
        },
        {
            "title": "Gladiator",
            "desc": (
                "A former Roman general sets out to exact vengeance against"
                " the corrupt emperor who murdered his family."
            ),
        },
        {
            "title": "Inception",
            "desc": (
                "A thief who steals corporate secrets through the use of"
                " dream-sharing technology is given the inverse task of"
                " planting an idea."
            ),
        },
        {
            "title": "Toy Story",
            "desc": (
                "A cowboy doll is profoundly threatened and jealous when a new"
                " spaceman action figure supplants him as top toy in a boy's"
                " room."
            ),
        },
        {
            "title": "Finding Nemo",
            "desc": (
                "After his son is captured in the Great Barrier Reef, a timid"
                " clownfish sets out on a journey to bring him home."
            ),
        },
        {
            "title": "The Lord of the Rings: The Fellowship of the Ring",
            "desc": (
                "A meek Hobbit of the Shire and eight companions set out on a"
                " journey to destroy the powerful One Ring and save"
                " Middle-earth."
            ),
        },
        {
            "title": "Blade Runner 2049",
            "desc": (
                "A young blade runner's discovery of a long-buried secret leads"
                " him to track down former blade runner Rick Deckard."
            ),
        },
        {
            "title": "La La Land",
            "desc": (
                "While navigating their careers in Los Angeles, a pianist and an"
                " actress fall in love while attempting to reconcile their"
                " aspirations."
            ),
        },
        {
            "title": "Whiplash",
            "desc": (
                "A promising young drummer enrolls at a cut-throat music"
                " conservatory where his dreams of greatness are mentored by"
                " an instructor who will stop at nothing."
            ),
        },
        {
            "title": "Jurassic World",
            "desc": (
                "A new theme park, built on the original site of Jurassic"
                " Park, creates a genetically modified hybrid dinosaur,"
                " which escapes and goes on a rampage."
            ),
        },
    ]

    print(f"=== Initializing Movie Finder using [{EMBED_MODEL}] ===")
    print(f"Encoding {len(movies)} movie descriptions...")

    # Pre-compute embeddings for all movies
    movie_embeddings = [get_embedding(m["desc"]) for m in movies]

    # Test query from the assignment prompt
    query = "a film about a robot that falls in love"
    print(f"\n🔍 Search Query: '{query}'")

    # Generate embedding for the search query
    query_embedding = get_embedding(query)

    # Compute scores against every movie
    scored_movies = []
    for idx, movie in enumerate(movies):
        sim = cosine_similarity(query_embedding, movie_embeddings[idx])
        scored_movies.append((sim, movie))

    # Sort results by similarity score in descending order
    scored_movies.sort(key=lambda x: x[0], reverse=True)

    print("\n--- Top 3 Best Matches ---")
    for i in range(3):
        score, movie = scored_movies[i]
        print(f"{i+1}. {movie['title']} (Score: {score:.4f})")
        print(f"   Summary: {movie['desc']}")


if __name__ == "__main__":
    main()