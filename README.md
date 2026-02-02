# 🎥 YouTube Video Chatbot

A Streamlit-based chatbot that allows users to **chat with a YouTube video** using its transcript.  
The app uses **RAG (Retrieval-Augmented Generation)** with **local LLMs via Ollama**, ensuring privacy and zero API cost.

---

## 🔍 What This App Does

1. Takes a **YouTube video ID or URL**
2. Fetches the video **transcript**
3. Splits transcript into chunks
4. Creates **embeddings using Ollama (local)**
5. Stores embeddings in **FAISS** (vector database)
6. Answers user queries using **RAG**
7. Displays responses in a **ChatGPT-style Streamlit UI**

---

## 🧠 Tech Stack

- **Streamlit** – frontend UI
- **LangChain** – RAG orchestration
- **Ollama** – local LLM & embeddings
- **FAISS** – vector similarity search
- **youtube-transcript-api** – transcript extraction
- **Python**

---

## 📁 Project Structure

ChatBot/
│── app.py # Experiment
│── summariseYT.py # RAG chatbot logic
│── main.py # Streamlit app entry point
│── requirements.txt # Python dependencies
│── .gitignore
│── README.md
