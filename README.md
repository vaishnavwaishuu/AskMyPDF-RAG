# 📄 RAG PDF Question Answering

Ask natural-language questions about your own PDFs and get answers that come **only from your documents**, with the **source file and page** cited. Built with LangChain, OpenAI embeddings and a local FAISS vector store.

## Features
- 📚 Loads every PDF in `data/` (subfolders included), page by page
- ✂️ Splits text into overlapping chunks (1200 chars, 200 overlap)
- 🔎 Semantic search with FAISS, returning the top 5 chunks above a relevance threshold
- 🛡️ Strict prompt: answers only from the retrieved context, otherwise replies *"I don't know based on the provided documents."*
- 🔖 Citations in the form `(filename, page N)`
- 💾 Index is cached in `faiss_index/`, so embeddings are only paid for once (`--rebuild` to refresh)

## How it works
```
PDFs (data/) → load pages → split into chunks → embed (text-embedding-3-small)
      → FAISS index → question → retrieve top-5 chunks → strict prompt → LLM → cited answer
```

## Tech Stack
| Component | Technology |
|---|---|
| Framework | LangChain |
| PDF loader | PyPDFLoader (page-level metadata) |
| Splitter | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS (local) |
| LLM | OpenAI chat model, `temperature=0` (default `gpt-4o-mini`, configurable) |

## Project Structure
```
├── main.py                  # Full RAG pipeline (CLI chat)
├── notebook.ipynb           # Step-by-step walkthrough
├── data/                    # Put your PDFs here (a fictional sample is included)
├── dataset_description.md   # Document processing details
├── project_summary.md       # Technical summary
├── requirements.txt
├── .env.example
└── README.md
```

## Setup
```bash
git clone https://github.com/vaishnavwaishuu/AskMyPDF-RAG.git
cd AskMyPDF-RAG
pip install -r requirements.txt
cp .env.example .env        # then add your OpenAI API key inside .env
```

## Usage
```bash
python main.py              # build (or load cached) index, then ask questions
python main.py --rebuild    # re-read the PDFs after you add or change files
```
Type `quit` to exit. To use another chat model, set `OPENAI_CHAT_MODEL` in `.env`.

### Try it with the included sample
`data/sample_employee_handbook.pdf` is a **fictional** handbook made for this demo. Try:
- *How many days of annual leave do employees get?*
- *What is the notice period after probation?*
- *What is the learning budget?*
- *Who is the CEO?* → should answer "I don't know based on the provided documents."

To use your own documents, add PDFs to `data/` and run `python main.py --rebuild`.

## Notes & Limitations
- Works with text-based PDFs. Scanned PDFs need OCR first.
- Needs an OpenAI API key; embedding and chat calls are billed by OpenAI.
- Retrieval quality depends on chunking and the score threshold (`CHUNK_SIZE`, `SCORE_THRESHOLD` in `main.py`).

## Future Improvements
- Streamlit / web UI with PDF upload
- Re-ranking and hybrid (keyword + semantic) search
- Conversation memory for follow-up questions
- Evaluation set to measure answer accuracy

## Author
**Vaishnav** — B.Tech CSE (AI), Amrita Vishwa Vidyapeetham
[LinkedIn]( https://www.linkedin.com/in/vaishnav-pv/) · [GitHub](https://github.com/vaishnavwaishuu)

