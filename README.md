# Knowledge-Intelligence-Hub-Chatbot-Real-Time-Web-Search

 <img width="1354" height="648" alt="image" src="https://github.com/user-attachments/assets/9ab5b8b9-89ac-43d6-b0db-aac0722c4748" />


🧠 Project Overview

The Knowledge Intelligence Hub Chatbot is a web-based AI assistant built with Streamlit. It helps users find answers from their own documents by combining document search with AI-generated responses.

The chatbot can also pull live information from the internet, making answers more relevant and updated when required.

This project represents a practical implementation of a RAG-based system using industry-standard tools and clean coding practices.

⭐ Core Highlights

📁 Query Multiple Files Together
Users can upload several PDF or TXT files and ask questions across all of them in one go.

⚡ Fast Meaning-Based Search
Uses vector embeddings and FAISS to quickly locate relevant text from documents.

🌍 Live Internet Data Support
Enhances answers with real-time web results using Tavily when enabled.

🔖 Answer Traceability
Responses include references showing where the information came from.

🗨️ Real-Time Chat Responses
AI answers stream gradually, creating a natural conversation flow.

🧱 Clean and Scalable Design
Code is divided into logical modules for easy maintenance and extension.

🏛️ System Design Overview

The chatbot follows a combined document + web RAG pipeline:

🗂️ File Processing

Users upload documents

Text is cleaned and broken into chunks

Important metadata is retained

🧮 Vector Creation & Storage

Text chunks are transformed into numerical vectors

FAISS stores and retrieves vectors efficiently

🔎 Question Handling

User queries trigger similarity matching

Optional web lookup is executed

🧩 Context Building

Document results and web snippets are merged

Combined context is sent to the AI model

✍️ Response Creation

Groq-powered LLM generates accurate answers

Responses are displayed progressively

🧰 Tools & Technologies
| Area            | Tool               |
| --------------- | ------------------ |
| Programming     | Python             |
| AI Model        | Groq (ChatGroq)    |
| Text Embeddings | HuggingFace Models |
| Vector Index    | FAISS              |
| Web Lookup      | Tavily             |
| Frontend        | Streamlit          |
| AI Framework    | LangChain          |

🗃️ Folder Layout
.
├── app.py                     # Main Streamlit application
├── config/
│   └── settings.py            # Global settings & API keys
├── core/
│   ├── document_loader.py     # File loading & text splitting
│   ├── embedding_engine.py    # Embedding logic
│   ├── faiss_manager.py       # Vector index handling
│   └── rag_pipeline.py        # Retrieval and generation flow
├── tools/
│   └── web_search.py          # Tavily integration
├── ui/
│   ├── layout.py              # UI structure
│   └── chat_ui.py             # Chat handling
├── data/
│   ├── uploads/               # User documents
│   └── vector_db/             # Stored FAISS index
├── requirements.txt
└── README.md


⚙️ Installation Guide
🪜 Step 1: Download the Code
git clone <your-github-repo-url>
cd <project-folder>

🧪 Step 2: Set Up Virtual Environment
python -m venv .venv


Windows

.venv\Scripts\Activate.ps1


macOS / Linux

source .venv/bin/activate

📦 Step 3: Install Dependencies
pip install -r requirements.txt

🔐 Step 4: Add API Keys

Create a .env file:

GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

▶️ Start the Application
streamlit run app.py


Your chatbot will open in the browser and be ready to use 🚀

