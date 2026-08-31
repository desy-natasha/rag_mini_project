# Study Assistant RAG

## About
This repository contains the data source and notebooks for a personal project building RAG pipeline as a study assistant that retrieves answers grounded in the information provided in the source text. This produces more focused answers that stem directly from my own notes and textbook, while also being able to transparently indicate when an answer is not available in the source material.

The project is built with both raw Python and LangChain to compare their code efficiency and implementation tradeoffs.

## Live demo

The LangChain version is deployed as an interactive Streamlit app: **[Study Assistant](https://desystudyassistant.streamlit.app)**

The app supports multi-turn conversation: follow-up questions are automatically rewritten into standalone queries before retrieval, and thus the assistant can handle references to earlier queries (e.g. "can you give an example of that?").

## Architecture
Below is the pipeline overview for this project splitted into two main stages. The first stage processes and stores the source text, so it can be reused across different queries. The second stage runs every time a query is received, using the stored data to generate an answer.

**Ingestion**

```
PDF Files → Extract Text → Clean Text → Chunk Text → Embed Chunks → Store in Vector Database
```

**Query**

```
User Query → Embed Query → Retrieve Top-k Chunks → Build Prompt (with History + Context) → Generate Answer → Return Answer with Sources
```

We used `SentenceTransformer` for the embeddings. For answer generation, the notebooks use `Llama3.1:8b` through Ollama for local development. 

The deployed Streamlit app uses `openai/gpt-oss-20b` through the Groq API, since Ollama requires a local model server and is not available for cloud deployment.

## Repository structure

* **data_source/** - Source text materials
* **chroma_db/** - Persisted vector store from raw Python
* **chroma_db_langchain/** - Persisted vector store from LangChain
* **`study_assistant_rag.ipynb`** - RAG pipeline notebook using raw Python
* **`study_assistant_rag_langchain.ipynb`** - RAG pipeline notebook using LangChain
* **`app_langchain.py`** - Streamlit app version of the LangChain pipeline for deployment