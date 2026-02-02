"""
========================================================
YouTube Video Chatbot (Streamlit + LangChain + Ollama)
========================================================

What this app does:
-------------------
1. Takes a YouTube video ID
2. Fetches its transcript
3. Splits transcript into chunks
4. Creates embeddings using Ollama (local, free)
5. Stores embeddings in FAISS (vector database)
6. Lets user chat with the video using RAG
7. Displays responses in a ChatGPT-style UI

Tech Stack:
-----------
- Streamlit (frontend)
- youtube-transcript-api (data source)
- LangChain (RAG orchestration)
- Ollama (local LLM + embeddings)
- FAISS (vector search)

Author: You 😊
========================================================
"""

# ======================================================
# 1️⃣ Imports
# ======================================================

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

# ======================================================
# 2️⃣ Streamlit Page Configuration
# ======================================================

st.set_page_config(
    page_title="🎥 YouTube Video Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Chat with a YouTube Video")
st.caption("Ask questions or get summaries — powered by Ollama + LangChain (100% local)")

# ======================================================
# 3️⃣ Session State (VERY IMPORTANT)
# ======================================================
# Streamlit reruns the script on every interaction.
# Session state allows us to persist data across reruns.

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ======================================================
# 4️⃣ Sidebar: User Inputs
# ======================================================

with st.sidebar:
    st.header("⚙️ Settings")

    video_id = st.text_input(
        "YouTube Video ID",
        value="Rni7Fz7208c",
        help="Only the video ID, not the full URL"
    )

    build_btn = st.button("🔄 Build Knowledge Base")

# ======================================================
# 5️⃣ Helper Functions
# ======================================================

def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript for a YouTube video and
    return it as a single plain text string.
    """
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)
    transcript_text = " ".join(snippet.text for snippet in transcript_list)
    return transcript_text


def build_vector_store(transcript: str) -> FAISS:
    """
    1. Split transcript into overlapping chunks
    2. Create embeddings
    3. Store them in FAISS
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.create_documents([transcript])

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def format_docs(docs):
    """
    Convert retrieved documents into a single context string
    """
    return "\n\n".join(doc.page_content for doc in docs)

# ======================================================
# 6️⃣ Build Knowledge Base (Indexing Step)
# ======================================================

if build_btn:
    with st.spinner("📥 Fetching transcript and building index..."):
        try:
            transcript = fetch_transcript(video_id)
            st.session_state.vector_store = build_vector_store(transcript)
            st.success("✅ Knowledge base built successfully!")

        except TranscriptsDisabled:
            st.error("❌ Transcripts are disabled for this video.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ======================================================
# 7️⃣ Chatbot Logic (RAG)
# ======================================================

if st.session_state.vector_store is not None:

    # ---- Retriever ----
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # ---- Prompt Template ----
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY using the provided transcript context.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # ---- LLM ----
    llm = Ollama(model="phi")

    # ---- Output Parser ----
    parser = StrOutputParser()

    # ---- Full RAG Chain ----
    rag_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | parser
    )

    # ==================================================
    # 8️⃣ Display Chat History
    # ==================================================

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # ==================================================
    # 9️⃣ User Input
    # ==================================================

    user_query = st.chat_input("Ask something about the video...")

    if user_query:
        # Store user message
        st.session_state.chat_history.append(("user", user_query))

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = rag_chain.invoke(user_query)
                st.markdown(answer)

        # Store assistant response
        st.session_state.chat_history.append(("assistant", answer))

else:
    st.info("👈 Enter a YouTube video ID and click **Build Knowledge Base** to start chatting.")
