import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embeddings import create_embeddings

PROMPT = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(text: str):
    db = Chroma(
        persist_directory="chroma", 
        embedding_function=create_embeddings()
    )

    # Returns top 5 results
    results = db.similarity_search_with_score(text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(
        context=context_text, 
        question=text
    )

    model = Ollama(model="llama2")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query = args.query_text
    query_rag(query)

if __name__ == "__main__":
    main()