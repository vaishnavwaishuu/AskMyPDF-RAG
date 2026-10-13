# Project Summary — RAG PDF Question Answering

## Objective
Build a Retrieval-Augmented Generation (RAG) pipeline that lets users ask natural-language questions about their own PDF documents and receive answers grounded strictly in the provided content, with source and page citations.

## Technical Stack

| Component | Technology | Role |
|---|---|---|
| **Framework** | LangChain | Orchestrates the RAG pipeline |
| **Document Loader** | PyPDFLoader via DirectoryLoader | Reads PDFs page by page with `source` / `page` metadata |
| **Text Splitter** | RecursiveCharacterTextSplitter | Splits pages into overlapping chunks |
| **Embedding Model** | OpenAI text-embedding-3-small | Converts chunks to 1536-dimensional vectors |
| **Vector Store** | FAISS | Local similarity search, cached to `faiss_index/` |
| **LLM** | OpenAI chat model (default gpt-4o-mini, temperature=0) | Generates grounded answers |
| **Retrieval** | Similarity-score threshold (≥ 0.3), top-5 | Filters out irrelevant chunks |

## Implementation Details

### Document Processing
- Loads all PDFs from `./data/` recursively
- Splits into chunks of **1200 characters** with **200-character overlap**
- Separators (in order): paragraph break, line break, period, space

### Embedding & Retrieval
- Each chunk becomes a 1536-dimensional vector
- Vectors are stored in FAISS and cached to disk after the first build
- At query time the top 5 chunks above the relevance threshold are retrieved

### Answer Generation
- Retrieved chunks are labelled `[source: file, page: N]` and injected into a strict prompt
- The model must answer only from that context and cite `(filename, page N)`
- `temperature=0` keeps answers consistent

### Anti-Hallucination
- The prompt forbids outside knowledge and guessing
- If the answer is not in the context, the reply is "I don't know based on the provided documents."
- This matters most for legal, medical and financial documents, where made-up answers are costly

## Changes in this version
- Fixed the text splitter separators in `main.py` (they were literal `\n` text instead of newlines)
- Switched to PyPDFLoader so page numbers exist for citations (UnstructuredFileLoader gave no page metadata)
- Chunks are labelled with source and page before reaching the LLM
- Added FAISS index caching, a question loop, `--rebuild`, and clear errors for a missing API key or empty `data/`
- Made the chat model configurable and aligned `main.py` with the notebook (gpt-4o-mini)
- Added `requirements.txt` (including `tqdm`, needed for the loader's progress bar), README and a sample PDF

## References
- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS by Meta AI](https://github.com/facebookresearch/faiss)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [RAG — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)
