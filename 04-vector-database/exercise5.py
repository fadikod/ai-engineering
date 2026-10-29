import chromadb

# 1. Sample FAQ documents
faq_data = [
    {"id": "faq1", "text": "How do I reset my password? Go to account settings and click reset."},
    {"id": "faq2", "text": "What is the refund policy? You can request a refund within 30 days of purchase."},
    {"id": "faq3", "text": "How do I contact customer support? Email support@example.com or call our helpline."},
    {"id": "faq4", "text": "Where can I find my invoice? Invoices are available under billing history in your profile."},
    {"id": "faq5", "text": "How do I update my profile picture? Click on your avatar and select change photo."}
]

questions = [
    "I forgot my password",
    "Can I get my money back?",
    "How to reach support team?",
    "Where is my receipt?",
    "How to edit my profile avatar?"
]

# 2. Setup Chroma in-memory client with cosine space
client = chromadb.Client()
collection = client.create_collection(
    name="faq_comparison",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=[item["id"] for item in faq_data],
    documents=[item["text"] for item in faq_data]
)

print("=== Comparing Chroma Search & Similarity Scores ===")

# 3. Query and convert distance to similarity
for q in questions:
    res = collection.query(query_texts=[q], n_results=1)
    
    top_doc = res['documents'][0][0]
    top_dist = res['distances'][0][0]
    
    # Distance to similarity formula: (1 - distance)
    chroma_similarity = 1.0 - top_dist

    print(f"\nQuery: '{q}'")
    print(f"Top Match: {top_doc}")
    print(f"Chroma Distance: {top_dist:.4f}")
    print(f"Calculated Similarity (1 - distance): {chroma_similarity:.4f}")