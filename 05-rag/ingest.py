
from pathlib import Path
import chromadb  # ADDED

# Locate the documents directory
docs_path = Path("docs")

# Find all Markdown files
files = list(docs_path.glob("*.md"))

all_chunks = []  # ADDED


def extract_sections(content):

    sections = []
    current_section = None
    current_content = []

    # Process the document line by line
    for line in content.splitlines():

        # Check if the line is a section heading
        if line.startswith("## "):

            # Save the previous section
            if current_section:
                sections.append({
                    "section": current_section,
                    "content": "\n".join(current_content).strip()
                })

            # Start a new section
            current_section = line.replace("## ", "").strip()
            current_content = []

        # Collect content belonging to the current section
        else:
            if current_section:
                current_content.append(line)

    # Save the final section
    if current_section:
        sections.append({
            "section": current_section,
            "content": "\n".join(current_content).strip()
        })

    # Return all extracted sections
    return sections


def chunks_text( text, chunk_size = 500, overlap = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


# Read each document
for file in files:

    content = file.read_text(encoding="utf-8")

    # Extract sections from the document
    sections = extract_sections(content)

    print(f"\nFile: {file.name}")

    # Process each section
    for section in sections:

        chunks = chunks_text(section["content"])

        # Process each chunk
        for chunk in chunks:

            chunk_data = {
                "text": chunk,
                "source": file.name,
                "section": section["section"]
            }

            all_chunks.append(chunk_data)  # ADDED

    print("*" * 40)


# ADDED: Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# ADDED: Create or retrieve collection
collection = client.get_or_create_collection(
    name="company_policies"
)

# ADDED: Prepare documents, metadata and IDs
documents = [item["text"] for item in all_chunks]

metadatas = [
    {
        "source": item["source"],
        "section": item["section"]
    }
    for item in all_chunks
]

ids = [f"chunk_{i}" for i in range(len(all_chunks))]

# ADDED: Store everything in ChromaDB
if all_chunks:
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print(f"Successfully stored {len(all_chunks)} chunks in ChromaDB.")
print(f"Total chunks in collection: {collection.count()}")


# Your temporary chunking test can now be removed
