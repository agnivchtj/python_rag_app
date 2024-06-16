import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embeddings import create_embeddings
from langchain_community.vectorstores import Chroma

def load_pdfs():
    pdf_loader = PyPDFDirectoryLoader("data")
    return pdf_loader.load()

def split_pdfs(documents: list[Document]):
    parse_text = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=80, 
        length_function=len, 
        is_separator_regex=False
    )
    return parse_text.split_documents(documents)

def get_chunks_with_id(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    curr_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        curr_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if curr_page_id == last_page_id:
            curr_chunk_index += 1
        else:
            curr_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{curr_page_id}:{curr_chunk_index}"
        last_page_id = curr_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def populate_db(chunks: list[Document]):
    db = Chroma(
        persist_directory="chroma", 
        embedding_function=create_embeddings()
    )
    chunks_with_id = get_chunks_with_id(chunks)

    # Add or update existing documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't already exist in the DB.
    new_chunks = []
    for chunk in chunks_with_id:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks) == 0:
        print("There are no new documents")
    
    print(f"Adding new documents: {len(new_chunks)}")
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()

def clear_db():
    if os.path.exists("chroma"):
        shutil.rmtree("chroma")

def main():
    documents = load_pdfs()
    chunks = split_pdfs(documents)
    populate_db(chunks)
    # print(create_embeddings().embed_query("This is a test document"))

if __name__ == "__main__":
    main()