
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Define the path to the folder containing our Markdown files
docs_path = Path("docs")

# Configure the Markdown headers used for splitting
headers_to_split_on = [
    ("#", "title"),
    ("##", "section")
]

# Create the Markdown splitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Store all chunks from all Markdown files
all_chunks = []

# Process every Markdown file in the docs folder
for file_path in docs_path.glob("*.md"):

    # Read the current Markdown file
    content = file_path.read_text(encoding="utf-8")

    # Split the content into Document objects
    chunks = splitter.split_text(content)

    # Add the original filename to each chunk's metadata
    for chunk in chunks:
        chunk.metadata["source"] = file_path.name

    # Add all chunks to the combined list
    all_chunks.extend(chunks)


# Configure the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Define where Chroma will store the vector database
persist_directory = "chroma_db"

# Create the vector store and store all chunks
vector_store = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="company_policies"
)

# Display the number of stored chunks
print(f"Successfully stored {len(all_chunks)} chunks in ChromaDB.")
