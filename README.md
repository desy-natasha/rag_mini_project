# Study Assistant RAG

## About
This repository contains the data source and notebooks for a personal project building RAG pipeline as a study assistant that retrieves answers grounded in the information provided in the source text. This produces more focused answers that stem directly from my own notes and textbook, while also being able to transparently indicate when an answer is not available in the source material.

The project is built with both raw Python and LangChain to compare their code efficiency and implementation tradeoffs.

## Architecture
Below is the pipeline overview for this project splitted into two main stages. The first stage processes and stores the source text, so it can be reused across different queries. The second stage runs every time a query is received, using the stored data to generate an answer.

**Ingestion**

```
PDF Files → Extract Text → Clean Text → Chunk Text → Embed Chunks → Store in Vector Database
```

**Query**

```
User Query → Embed Query → Retrieve Top-k Chunks → Build Prompt → Generate Answer → Return Answer with Sources
```
We used `SentenceTransformer` for the embeddings and `Llama3.1:8b` as the LLM for generating answers through Ollama.

## Repository structure

* **data_source/** - Source text materials
* **chroma_db/** - Persisted vector store from raw Python
* **chroma_db_langchain/** - Persisted vector store from LangChain
* **`study_assistant_rag.ipynb`** - RAG pipeline notebook using raw Python
* **`study_assistant_rag_langchain.ipynb`** - RAG pipeline notebook using LangChain