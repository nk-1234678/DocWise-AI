
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from utils.chat import generate_general_answer
from utils.router import is_document_question

from utils.pdf_reader import extract_pdf_text
from utils.chunking import split_text
from utils.embedding import generate_embeddings
from utils.vector_store import create_vector_store
from utils.retriever import retrieve_chunks
from utils.llm import generate_answer


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="DocWise AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# GEMINI API
# =====================================================

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "index": None,
    "chunks": None,
    "processed": False,
    "messages": [],
    "uploaded_names": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* MAIN */
.stApp {
    background: #ffffff;
    color: #202123;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #f8f8fa;
    border-right: 1px solid #ededf0;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1rem 0.8rem;
}

.brand {
    font-size: 24px;
    font-weight: 750;
    letter-spacing: -1px;
    color: #202123;
    padding: 8px 8px 25px;
}

.brand span {
    color: #8065c9;
}

.side-label {
    color: #92929a;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 25px 8px 10px;
}

section[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: none;
    color: #414149;
    text-align: left;
    border-radius: 9px;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: #ececf1;
}

/* WELCOME */
.welcome-area {
    text-align: center;
    margin-top: 19vh;
    margin-bottom: 35px;
}

.welcome-icon {
    width: 55px;
    height: 55px;
    margin: 0 auto 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #f0ecfb;
    border-radius: 18px;
    color: #8065c9;
    font-size: 27px;
}

.welcome-title {
    font-size: 34px;
    font-weight: 600;
    letter-spacing: -1.3px;
    color: #202123;
}

.welcome-subtitle {
    color: #898991;
    font-size: 14px;
    margin-top: 12px;
}

/* SUGGESTIONS */
.suggestion {
    background: white;
    border: 1px solid #ececf0;
    padding: 15px;
    border-radius: 14px;
}

.suggestion-title {
    color: #33333a;
    font-size: 13px;
    font-weight: 600;
}

.suggestion-desc {
    color: #92929a;
    font-size: 11px;
    margin-top: 5px;
}

/* CHAT */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 12px 4px;
}

[data-testid="stChatMessage"] p {
    color: #292930;
    line-height: 1.75;
    font-size: 15px;
}

/* CHAT COMPOSER */
[data-testid="stChatInput"] {
    border: 1px solid #e4e2e9 !important;
    border-radius: 25px !important;
    background: #ffffff !important;
    box-shadow: 0 5px 22px rgba(30,25,50,0.08);
}

[data-testid="stChatInput"]:focus-within {
    border-color: #b7a7e0 !important;
    box-shadow: 0 5px 24px rgba(128,101,201,0.12);
}

[data-testid="stChatInput"] textarea {
    color: #292930 !important;
    font-size: 14px !important;
}

/* FILE ATTACHMENT */
[data-testid="stChatInput"] button {
    color: #6f58ad;
}

/* STATUS */
[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #a1a1a8;
    font-size: 11px;
    padding: 20px;
}

/* MOBILE */
@media(max-width:700px) {
    .welcome-area {
        margin-top: 8vh;
    }

    .welcome-title {
        font-size: 27px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">DocWise<span> AI.</span></div>',
        unsafe_allow_html=True
    )

    if st.button("✎   New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="side-label">RECENT CHATS</div>',
        unsafe_allow_html=True
    )

    if st.session_state.messages:

        first_question = next(
            (
                m["content"]
                for m in st.session_state.messages
                if m["role"] == "user"
            ),
            "New conversation"
        )

        st.caption("💬 " + first_question[:35])

    else:
        st.caption("Your conversations will appear here.")

    st.markdown(
        '<div class="side-label">DOCUMENT STATUS</div>',
        unsafe_allow_html=True
    )

    if st.session_state.processed:

        st.success("Knowledge Base Ready")

        for name in st.session_state.uploaded_names:
            st.caption("📄 " + name)

    else:
        st.caption("No documents uploaded yet.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**✦ DocWise AI**")
    st.caption("Powered by Google Gemini")


# =====================================================
# PROCESS PDF FILES
# =====================================================

def process_pdfs(uploaded_files):

    all_text = ""

    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())
            pdf_path = tmp.name

        try:
            text = extract_pdf_text(pdf_path)
            all_text += "\n\n" + text

        finally:
            os.remove(pdf_path)

        progress.progress((i + 1) / len(uploaded_files))

    if not all_text.strip():
        raise ValueError("No readable text found in uploaded PDFs.")

    with st.spinner("Creating Knowledge Base..."):

        chunks = split_text(all_text)

        if not chunks:
            raise ValueError("No text chunks could be created.")

        embeddings = generate_embeddings(client, chunks)

        index = create_vector_store(embeddings)

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.processed = True

    st.session_state.uploaded_names = [
        pdf.name for pdf in uploaded_files
    ]


# =====================================================
# WELCOME SCREEN
# =====================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-area">
            <div class="welcome-icon">✦</div>
            <div class="welcome-title">
                What can I help you with?
            </div>
            <div class="welcome-subtitle">
                Explore your documents and discover information with AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            """
            <div class="suggestion">
                <div class="suggestion-title">
                    ✧ Understand your documents
                </div>
                <div class="suggestion-desc">
                    Find important information in your PDFs.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="suggestion">
                <div class="suggestion-title">
                    ✧ Get quick answers
                </div>
                <div class="suggestion-desc">
                    Ask questions about your uploaded files.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )



# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])

            for pdf_name in message.get("attachments", []):
                st.caption("📎 " + pdf_name)

    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])



# =====================================================
# CHAT INPUT WITH PDF ATTACHMENT
# =====================================================

prompt = st.chat_input(
    "Message DocWise AI...",
    accept_file="multiple",
    file_type=["pdf"]
)



# =====================================================
# HANDLE USER INPUT AND PDF UPLOAD
# =====================================================

if prompt is not None:
    user_text = (prompt.text or "").strip()
    uploaded_files = prompt.files or []

    # Process uploaded PDFs
    if uploaded_files:
        try:
            with st.spinner("Processing your PDF documents..."):
                process_pdfs(uploaded_files)

            st.toast("PDF documents processed successfully!")

        except Exception as e:
            st.error(f"PDF processing failed: {e}")
            st.stop()

    if user_text:
        st.session_state.messages.append({
            "role": "user",
            "content": user_text,
            "attachments": [pdf.name for pdf in uploaded_files]
        })

        # Previous conversation, excluding current question
        previous_messages = st.session_state.messages[:-1]

        with st.chat_message("assistant"):
            try:
                # Identify question type
                with st.spinner("Understanding your question..."):
                    document_question = is_document_question(
                        client,
                        user_text,
                        previous_messages
                    )

                if document_question:
                    if not st.session_state.processed:
                        answer = (
                            "Please upload a PDF document first "
                            "to ask document-related questions."
                        )
                        st.markdown(answer)

                    else:
                        with st.spinner("Searching your documents..."):
                            response = client.models.embed_content(
                                model="gemini-embedding-001",
                                contents=user_text
                            )

                            query_embedding = response.embeddings[0].values

                            retrieved_chunks = retrieve_chunks(
                                st.session_state.index,
                                query_embedding,
                                st.session_state.chunks,
                                k=5
                            )

                            context = "\n\n".join(retrieved_chunks)

                        # Stream PDF answer
                        answer = st.write_stream(
                            generate_answer(
                                client,
                                context,
                                user_text
                            )
                        )

                else:
                    # Stream normal chat answer
                    answer = st.write_stream(
                        generate_general_answer(
                            client,
                            previous_messages,
                            user_text
                        )
                    )

            except Exception as e:
                answer = f"Something went wrong: {e}"
                st.error(answer)

        # Save completed assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    elif uploaded_files:
        st.rerun()




# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    DocWise AI · Powered by Google Gemini
</div>
""", unsafe_allow_html=True)
