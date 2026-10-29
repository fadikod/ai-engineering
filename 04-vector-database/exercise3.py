import os
import chromadb

# 1. Initialize persistent client and collection for notes
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="notes")

notes_dir = "notes"

ids = []
documents = []
metadatas = []

# 2. Read each text file from the notes directory
for filename in os.listdir(notes_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(notes_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        ids.append(filename)  # Use filename as unique ID to prevent duplicates
        documents.append(content)
        metadatas.append({"source": filename})  # Store file_name in metadata[cite: 6]

# 3. Upsert into Chroma (running this twice won't duplicate items)[cite: 6]
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"Successfully loaded {len(ids)} note files into ChromaDB.")
print(f"Total items in collection: {collection.count()}")