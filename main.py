"""RAG PDF Question Answering.

Ask natural-language questions about your own PDFs. Answers are generated
only from the retrieved document chunks and cite the source file and page.

Usage:
    python main.py              # build (or load cached) index, then chat
    python main.py --rebuild    # re-read the PDFs and rebuild the index
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
SCORE_THRESHOLD = 0.3

PROMPT_TEMPLATE = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) Cite sources as (filename, page N) using the [source: ..., page: ...] "
    "tags shown in the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def load_documents():
    """Load every PDF under ./data (one Document per page, with page metadata)."""
    loader = DirectoryLoader(
        path=str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def get_vectorstore(embeddings, rebuild=False):
    """Load the cached FAISS index, or build it from the PDFs and cache it."""
    if INDEX_DIR.exists() and not rebuild:
        # The index is a file we created ourselves, so loading it is safe.
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    docs = load_documents()
    if not docs:
        sys.exit(f"No PDF files found in '{DATA_DIR}/'. Add at least one PDF and run again.")
    splits = split_documents(docs)
    print(f"Loaded {len(docs)} pages -> {len(splits)} chunks")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def format_docs(docs):
    """Prefix each chunk with its source file and (1-based) page for citations."""
    parts = []
    for d in docs:
        name = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page")
        page_label = page + 1 if isinstance(page, int) else "?"
        parts.append(f"[source: {name}, page: {page_label}]\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain(rebuild=False):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = get_vectorstore(embeddings, rebuild=rebuild)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the FAISS index")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    rag_chain = build_chain(rebuild=args.rebuild)
    print("\nReady! Type a question (or 'quit' to exit).")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print("\nAnswer:", rag_chain.invoke(question))


if __name__ == "__main__":
    main()
