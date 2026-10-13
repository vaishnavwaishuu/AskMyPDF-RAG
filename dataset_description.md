# Dataset Description — Document Processing in RAG

## Overview
Unlike a traditional ML project with a fixed dataset, RAG works with **any PDF documents you provide**. They are processed at runtime into a searchable vector index. A fictional demo file, `data/sample_employee_handbook.pdf`, is included so the project runs out of the box.

## Supported Input

| Property | Value |
|---|---|
| **File type** | Text-based PDF (`.pdf`); scanned PDFs need OCR first |
| **Location** | `./data/` folder |
| **Discovery** | Recursive — finds PDFs in subfolders |

## Document Processing Pipeline

### Step 1 — Loading
```python
loader = DirectoryLoader(
    path="./data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,    # one Document per page, with source + page metadata
    use_multithreading=True,
)
docs = loader.load()
```

### Step 2 — Chunking
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
)
splits = splitter.split_documents(docs)
```
**Why chunk?**
- LLMs have a limited context window
- Small chunks make retrieval more precise
- Overlap keeps information from being cut at chunk boundaries

### Step 3 — Embedding
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
```
Each chunk becomes a 1536-dimensional vector that captures its meaning. The index is saved to `faiss_index/`.

### Step 4 — Retrieval
```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},
)
```
At query time the question is embedded, FAISS finds the nearest chunks, and only those above the relevance threshold go to the LLM.

## Example Use Cases

| Document Type | Example Questions |
|---|---|
| Company HR handbook | "What is the annual leave policy?" |
| Legal contract | "What are the termination conditions?" |
| Research paper | "What methodology was used in the study?" |
| Product manual | "How do I reset the device to factory settings?" |
| Financial report | "What was the net profit in Q3?" |
