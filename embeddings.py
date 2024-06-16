from langchain_community.embeddings.ollama import OllamaEmbeddings

def create_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    return embeddings