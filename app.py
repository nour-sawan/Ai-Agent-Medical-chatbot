import os
import base64
import time
import streamlit as st
from dotenv import load_dotenv

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# ================= CONFIG =================
PDF_PATH = "data/Guideline-Hand-Hygiene.pdf"

OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if OPENAI_KEY is None:
    st.error("❌ API Key not found. Add it to Streamlit secrets or .env file")
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
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ================= PROMPT =================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a STRICT medical document assistant.

Answer ONLY from the given context.
If not found, respond exactly:
Not found in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ================= LOAD SYSTEM =================
with st.spinner("🔄 Loading document..."):
    vectorstore = load_vectorstore()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_KEY
)

# Increased k for better results ✅
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ================= TYPING ANIMATION =================
def typewriter(text: str):
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
    robot_b64 = base64.b64encode(open("assets/robot.gif", "rb").read()).decode()

    st.markdown(
        f"""
        <div class="robot-wrapper">
            <img class="robot-img" src="data:image/gif;base64,{robot_b64}">
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

    result = qa_chain.invoke(query)
    answer = result.content.strip()

    with st.chat_message("assistant"):
        typewriter(answer)

    st.session_state.chat_history.append((query, answer))

# ================= END MAIN CONTAINER =================
st.markdown('</div>', unsafe_allow_html=True)