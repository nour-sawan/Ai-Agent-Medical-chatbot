import os
import base64
import time  
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# ================= CONFIG =================
PDF_PATH = "data/Guideline-Hand-Hygiene.pdf"

OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")


if OPENAI_KEY is None:
    st.error("❌ API Key not found. Check your .env file.")
    st.stop()


if OPENAI_KEY is None:
    st.error("❌ API Key not found. Check your .env file.")
    st.stop()

st.set_page_config(page_title="Medical Q&A Chatbot", layout="centered")


# ================= LOAD CSS =================
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ================= BACKGROUND =================
def set_background(image_path):
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("assets/medical.avif")


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙ Controls")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ================= MAIN CONTAINER =================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
<div class="fadein-wrapper">
    <div class="fadein-box">
        <div class="fadein-text">
            🧼 Answers are generated ONLY from the CDC Hand Hygiene Guideline
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================= CHECK PDF =================
if not os.path.exists(PDF_PATH):
    st.error("❌ PDF file not found!")
    st.stop()


# ================= LOAD & PROCESS PDF =================
@st.cache_resource(show_spinner=False)
def load_vectorstore():

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ================= SMART + STRICT PROMPT =================
strict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a STRICT medical document assistant.

You MUST understand the intent and answer ONLY what is asked, based on the provided context.

Answer ONLY using the provided context.
Expand the answer ONLY when the question explicitly asks for details or explanation.
Do NOT add extra information that is not directly requested in the question.
Stay focused on answering the exact scope of the question.

If the answer is NOT FOUND, respond exactly:
Not found in the provided document.

Context:
{context}

User Question:
{question}

Answer format:

Answer:
- Start with a short, clear opening sentence
Use bullet points ( • ) and place EACH bullet on a NEW LINE
- Include ONLY information that answers the question directly
"""
)


# ================= LOAD SYSTEM =================
with st.spinner("🔄 Loading document..."):
    vectorstore = load_vectorstore()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    combine_docs_chain_kwargs={"prompt": strict_prompt},
    return_source_documents=False  # ✅ No source/page displayed
)

# ================= TYPING ANIMATION =================
def typewriter(text: str):
    """Show answer with typing animation inside chat-box div."""
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(
            f'<div class="chat-box">{displayed}</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.008) 

# ================= TITLE =================
st.title("🩺 Medical Document Q&A Chatbot")


# ================= ROBOT GIF =================
if os.path.exists("assets/robot.gif"):
    st.markdown(
        f"""
        <div class="gif-container">
            <img src="data:image/gif;base64,
            {base64.b64encode(open('assets/robot.gif', 'rb').read()).decode()}">
        </div>
        """,
        unsafe_allow_html=True
    )


# ================= SHOW CHAT HISTORY =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-box">{question}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-box">{answer}</div>', unsafe_allow_html=True)


# ================= USER INPUT =================
query = st.chat_input("Ask your medical policy question...")


# ================= HANDLE QUERY =================
if query:

    with st.chat_message("user"):
        st.markdown(f'<div class="chat-box">{query}</div>', unsafe_allow_html=True)

    result = qa_chain.invoke({
        "question": query,
        "chat_history": st.session_state.chat_history
    })

    answer = result["answer"].strip()

    with st.chat_message("assistant"):
        typewriter(answer)

    st.session_state.chat_history.append((query, answer))




# ================= END MAIN CONTAINER =================
st.markdown('</div>', unsafe_allow_html=True)
