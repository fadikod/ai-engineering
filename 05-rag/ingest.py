
from pathlib import Path

# Locate the documents directory
docs_path = Path("docs")

# Find all Markdown files
files = list(docs_path.glob("*.md"))


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

    # Display each extracted section
    for section in sections:
        print(f"\nSection: {section['section']}")
        print(section["content"])

    print("*" * 40)


test_text = "A" * 1200
test_chunks = chunks_text(test_text)

print(f"Number of chunks: {len(test_chunks)}")

for i, chunk in enumerate(test_chunks):
    print(f"Chunk{i+1}: {len(chunk)} characters")