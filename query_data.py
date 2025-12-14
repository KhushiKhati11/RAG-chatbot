import argparse
from dataclasses import dataclass
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a careful research assistant that ONLY answers using the supplied context snippets.

Rules:
- Answer strictly from the given context. Do not use outside knowledge.
- If the answer is not explicitly stated, respond exactly: "Not in context."
- Always be precise and concise. Prefer short, factual answers.
- If multiple entities are present, focus ONLY on the one asked about.
- Never confuse characters, names, or attributes.
- Support your answer with a short, direct quote from the context as evidence.
- If sources conflict, state the conflict briefly.
- Do not reveal chain-of-thought or reasoning steps.

Context:
{context}

Question:
{query}

Answer format:
Answer: <one-line grounded answer or "Not in context">
Evidence: "<verbatim supporting quote from the context>"
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH , embedding_function=embedding_function)
    # print("Number of documents in DB:", db._collection.count())

    results = db.similarity_search_with_relevance_scores(args.query_text, k=8)
    if len(results) == 0 or results[0][1]<0.3:
        print("Unable to find relevant information.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query_text)
    print(prompt)

    model = OllamaLLM(model="phi3:mini") 
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text} \n Sources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
