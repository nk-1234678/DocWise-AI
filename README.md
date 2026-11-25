# DocWise AI – AI-Powered PDF Question Answering Chatbot

Live Demo: Try DocWise AI

DocWise AI is an intelligent PDF question-answering application built using Python, Streamlit, Google Gemini and Retrieval-Augmented Generation (RAG).

It allows users to upload PDF documents, ask questions about their content, and receive context-aware answers generated using AI. It also supports general-purpose conversations with session-based chat history.

## Features

- **PDF Upload:** Upload PDF documents and extract their text.
- **Intelligent Document Q&A:** Ask questions about uploaded documents using AI.
- **RAG Architecture:** Retrieve relevant document chunks before generating answers.
- **Semantic Search:** Generate embeddings using Google's Gemini embedding model.
- **Vector Search:** Use FAISS to find relevant document chunks.
- **AI-Powered Responses:** Generate explanatory answers using Google Gemini.
- **Normal Chat Mode:** Have general conversations beyond uploaded documents.
- **Conversation History:** Maintain context from previous messages during an active session.
- **Streaming Responses:** Display AI responses progressively.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Google Gemini API | AI-powered answer generation |
| Gemini Embeddings | Text embedding generation |
| FAISS | Vector similarity search |
| NumPy | Numerical operations |
| PyPDF | PDF text extraction |
| python-dotenv | Environment variable management |

## How It Works

DocWise AI follows a Retrieval-Augmented Generation (RAG) workflow.

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** Text is extracted from the uploaded PDF.
3. **Text Chunking:** Extracted text is divided into smaller chunks.
4. **Embedding Generation:** Gemini generates vector embeddings for the chunks.
5. **Vector Storage:** FAISS stores the embeddings for similarity search.
6. **Question Processing:** The user's question is converted into an embedding.
7. **Context Retrieval:** FAISS retrieves the most relevant document chunks.
8. **Answer Generation:** Gemini uses the retrieved context to generate an explanatory response.

## Project Structure

```text
DocWise_AI/
│
├── app.py
├── requirements.txt
├── .env
│
└── utils/
    ├── pdf_reader.py
    ├── chunking.py
    ├── embedding.py
    ├── vector_store.py
    ├── retriever.py
    ├── llm.py
    ├── chat.py
    └── router.py
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DocWise-AI.git
cd DocWise-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Future Improvements

- Improved retrieval accuracy and ranking.
- Conversation memory for document-based follow-up questions.
- Enhanced document understanding and reasoning.
- Support for additional document formats.
- Persistent document storage and chat history.

## Security

- API keys are stored using environment variables.
- Do not upload `.env` or API credentials to GitHub.
- Use Streamlit Secrets for deployed application credentials.

## Author

**Neelam Kumari**

B.Tech – Computer Science and Engineering



---

*DocWise AI – Making document understanding smarter with Generative AI and RAG.*
