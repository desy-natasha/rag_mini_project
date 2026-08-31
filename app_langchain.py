import streamlit as st
import logging
import re

logging.getLogger("pypdf").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


### CONFIG ###
CHROMA_PATH = "./chroma_db_langchain"
COLLECTION_NAME = "study_assistant_lc"
GROQ_MODEL = "openai/gpt-oss-20b"
HISTORY_TURNS_TO_KEEP = 3               # the number of past user/assistant exchanges to feed back to the model

st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="centered")


### CACHED RESOURCES ###
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore(_embeddings):
    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
    )

@st.cache_resource
def load_llm():
    # API key comes from Streamlit secrets
    return ChatGroq(model=GROQ_MODEL, api_key=st.secrets["GROQ_API_KEY"])

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm()


### DATA INGESTION HELPERS ###
def extract_pdf_pages(path):
    """Extracts text from a PDF file path and returns a list of cleaned Document objects."""
    loader = PyPDFLoader(path)
    raw_docs = loader.load()  # loader extracts one Document per page
    for d in raw_docs:
        cleaned_text = re.sub(r'(\w)-\n(\w)', r'\1\2', d.page_content)  # fix hyphen breaks
        cleaned_text = re.sub(r'\n', ' ', cleaned_text).strip()          # remove newlines and extra spaces
        d.page_content = cleaned_text
    return raw_docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # characters, not words — roughly 350-400 words
    chunk_overlap=200,    # overlap between chunks to maintain context
    separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph, then line, then sentence, then word
)


### CORE PIPELINE ###
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = PromptTemplate.from_template("""You are a study assistant. Answer the question using ONLY the context below.
If the context does not contain enough information to answer, say so clearly instead of guessing.

When citing information, use ONLY the bracket number, like [1] or [2], directly after the relevant
sentence. Do NOT repeat the source filename or page numbers in your answer text. A source list
will be shown separately after your answer.

Write your answer as clear, direct prose. Do not describe what each numbered source says one by
one and synthesize the information into a single coherent answer.

Conversation so far (for context only. answer the current question, not previous ones):
{history}

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    """Formats retrieved chunks into numbered text for the prompt, e.g. [1] source, page."""

    return "\n\n".join(
        f"[{i+1}] (Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )

def format_page(doc):
    """Returns a chunk's page number, e.g. 'page 12'."""

    return f"page {doc.metadata.get('page', '?')}"

def format_history(messages):
    """Formats recent chat turns as plain text (e.g. 'User: ...\nAssistant: ...') for the LLM to read as prior context. Keeps only the last few turns."""

    if not messages:
        return "(no previous conversation)"
    recent = messages[-(HISTORY_TURNS_TO_KEEP * 2):]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

def condense_question(query, history_messages):
    """Rewrites a follow-up question (e.g. 'what about the second one?') into a standalone question, so retrieval can actually search for it. 
    This to get the top-k relevant chunks based on the follow-up question, not the previous conversation."""

    if not history_messages:
        return query

    history_text = format_history(history_messages)
    condense_prompt = f"""Given this conversation history:
{history_text}

And this follow-up question: "{query}"

Rewrite the follow-up as a standalone question that makes sense without the conversation history.
If it's already standalone, return it unchanged. Reply with ONLY the rewritten question, nothing else."""

    result = llm.invoke(condense_prompt)
    return result.content.strip()

def ask(query, history_messages, k=3):
    standalone_query = condense_question(query, history_messages)
    retrieved_docs = retriever.invoke(standalone_query)
    context = format_docs(retrieved_docs)
    history_text = format_history(history_messages)

    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"history": history_text, "context": context, "question": query})
    return {"query": query, "standalone_query": standalone_query, "answer": answer, "sources": retrieved_docs}

def get_cited_sources(answer, sources):
    """Returns only the sources whose [n] citation number actually appears in the answer text."""

    cited_numbers = {int(n) for n in re.findall(r'\[(\d+)\]', answer)}
    return [
        (i, doc) for i, doc in enumerate(sources, 1)
        if i in cited_numbers
    ]

### UI ###
st.title("📚 Desy's Study Assistant")
st.caption("Hello! I'm a demo RAG assistant built from my personal notes and resources from an NLP course. Ask me anything! 📖")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=5, value=3)
    retriever.search_kwargs["k"] = k
    st.caption(f"Chroma collection: `{COLLECTION_NAME}` — {vectorstore._collection.count()} chunks indexed")

    st.divider()
    st.subheader("Add files (optional)")
    uploaded_files = st.file_uploader("Upload PDF files to index", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Index uploaded PDFs"):
        with st.spinner("Processing PDF files..."):
            documents = []
            for f in uploaded_files:
                temp_path = f"/tmp/{f.name}"
                with open(temp_path, "wb") as out:
                    out.write(f.getbuffer())
                documents.extend(extract_pdf_pages(temp_path))
            new_chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(new_chunks)
        st.success(f"Indexed {len(new_chunks)} new chunks from {len(uploaded_files)} file(s).")
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# session_state holds chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay existing conversation on every rerun (Streamlit reruns the whole script each time)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question, e.g. What is one-hot vector?")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(query, st.session_state.messages, k=k)
            except Exception as e:
                st.error(
                    "Couldn't get a response from Groq. Check that GROQ_API_KEY is set correctly "
                    f"in your Streamlit secrets, and that the model `{GROQ_MODEL}` is valid.\n\nDetails: {e}"
                )
                st.stop()

        st.write(result["answer"])

        with st.expander("Sources"):
            cited = get_cited_sources(result["answer"], result["sources"])
            if not cited:
                st.caption("No specific sources were cited in this answer.")
            for i, d in cited:
                source = d.metadata.get("source", "unknown")
                st.markdown(f"**[{i}] {source}, {format_page(d)}**")
                st.write(d.page_content)

    # Save both turns AFTER a successful answer, so history stays in sync with what was shown
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
