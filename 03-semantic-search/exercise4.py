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


def keyword_search(query: str, faq_database: list[dict]) -> dict | None:
    """Simple keyword search: returns the first FAQ where at least one word matches."""
    query_words = set(query.lower().split())
    for faq in faq_database:
        faq_words = set(faq["answer"].lower().split()) | set(
            faq["question"].lower().split()
        )
        # Check for intersection (shares at least one word)
        if query_words & faq_words:
            return faq
    return None


def semantic_search(query_embedding: np.ndarray, faq_embeddings: list[np.ndarray], faq_database: list[dict]) -> tuple[dict, float]:
    """Semantic search: returns the FAQ with the highest cosine similarity."""
    best_score = -1.0
    best_faq = None
    for idx, emb in enumerate(faq_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        if sim > best_score:
            best_score = sim
            best_faq = faq_database[idx]
    return best_faq, best_score


def main():
    # FAQ Database
    faq_database = [
        {
            "id": 1,
            "question": "What is the return policy?",
            "answer": "You can return any item within 30 days of purchase for a full refund.",
        },
        {
            "id": 2,
            "question": "How do I track my shipment?",
            "answer": "A tracking link is emailed to you as soon as your order leaves our warehouse.",
        },
        {
            "id": 3,
            "question": "Do you offer international shipping?",
            "answer": "Yes, we ship worldwide with varying delivery times depending on the destination country.",
        },
        {
            "id": 4,
            "question": "How can I reset my password?",
            "answer": "Click on 'Forgot Password' on the login page and follow the instructions sent to your email.",
        },
        {
            "id": 5,
            "question": "What payment methods do you accept?",
            "answer": "We accept major credit cards, PayPal, and Apple Pay.",
        },
    ]

    # Pre-compute embeddings for FAQ database
    faq_embeddings = [get_embedding(faq["answer"]) for faq in faq_database]

    # 8 Test Questions (some use synonyms/different phrasing where keyword search might fail)
    test_questions = [
        "What is the return policy?",  # Exact match (Both pass)
        "Can I get my money back?",  # Synonym test (Keyword fails - no shared words with "refund" or "return")
        "Where is my package?",  # Synonym test (Keyword fails - "package" vs "shipment")
        "How do I track my shipment?",  # Exact match (Both pass)
        "Do you ship abroad?",  # Synonym test (Keyword fails - "abroad" vs "international")
        "I forgot my login credentials",  # Synonym test (Keyword fails - "credentials" vs "password")
        "Can I pay with credit card?",  # Partial match (Keyword passes due to "credit card")
        "What are your delivery options overseas?",  # Synonym test (Keyword fails)
    ]

    print(f"=== Comparing Keyword Search vs. Semantic Search using [{EMBED_MODEL}] ===\n")

    for i, q in enumerate(test_questions, start=1):
        print(f"Test {i}: Query = '{q}'")
        
        # 1. Keyword Search
        kw_result = keyword_search(q, faq_database)
        kw_title = f"FAQ #{kw_result['id']}: {kw_result['question']}" if kw_result else "No Match"

        # 2. Semantic Search
        q_emb = get_embedding(q)
        sem_result, sem_score = semantic_search(q_emb, faq_embeddings, faq_database)
        sem_title = f"FAQ #{sem_result['id']}: {sem_result['question']} (Score: {sem_score:.4f})"

        print(f"  👉 Keyword Search  : {kw_title}")
        print(f"  👉 Semantic Search : {sem_title}")
        print("-" * 60)


if __name__ == "__main__":
    main()