import chromadb

# 1. Connect to client and get the notes collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="notes")

# 2. Choose a distance threshold (adjust based on your test outputs)
DISTANCE_THRESHOLD = 1.2

print("=== Search Notes CLI (type 'exit' or 'quit' to stop) ===")

# 3. Interactive query loop
while True:
    user_query = input("\nAsk a question: ").strip()
    if user_query.lower() in ["exit", "quit"]:
        break
    if not user_query:
        continue

    # Query the collection for the top 1 result
    results = collection.query(
        query_texts=[user_query],
        n_results=1
    )

    if results['documents'][0]:
        best_doc = results['documents'][0][0]
        best_source = results['metadatas'][0][0]['source']
        best_dist = results['distances'][0][0]

        # Check against threshold
        if best_dist <= DISTANCE_THRESHOLD:
            print(f"\n[Best Match - Source: {best_source}] (Distance: {best_dist:.4f})")
            print(f"Result: {best_doc}")
        else:
            print(f"\nNothing relevant found. (Best distance was {best_dist:.4f}, above threshold {DISTANCE_THRESHOLD})")
    else:
        print("\nNothing relevant found.")