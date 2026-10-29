import chromadb

# 1. Connect to the persistent client and get the collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="recipes")

# 2. Define your filters and query
cuisine_filter = "asian"
max_minutes = 30
query_text = "something warm and spicy"

# 3. Query with metadata filters ($and + $lte)
results = collection.query(
    query_texts=[query_text],
    n_results=3,
    where={
        "$and": [
            {"cuisine": {"$eq": cuisine_filter}},
            {"minutes": {"$lte": max_minutes}}
        ]
    }
)

# 4. Display results
print(f"=== Search: '{query_text}' | Cuisine: {cuisine_filter} | Max Time: <= {max_minutes} mins ===")
if results['documents'][0]:
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        print(f"- Recipe: {doc}")
        print(f"  Cuisine: {meta['cuisine']} | Minutes: {meta['minutes']} | Distance: {dist:.4f}\n")
else:
    print("No recipes found matching these criteria.")