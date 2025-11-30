##imported the libries I need for the code to Run
import os
import base64 ##for background and robot gif (the robot gif is in assets folder)
import time
import streamlit as st
from dotenv import load_dotenv
import re #for citation extraction
from collections import Counter ##for citation extraction


# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnableMap


##My API key is loaded from My .env file (please use your own Key)
# Load environment variables
load_dotenv()


# Configuration for the PDF path and OpenAI Key
# Configuration for the PDF path and OpenAI Key
PDF_PATH = "data/Guideline-Hand-Hygiene.pdf"

# load from Streamlit secrets on cloud
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error(" OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Medical Q&A Chatbot", layout="centered")




  #Loading CSS
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
        
  # BACKGROUND (I have a background genrated by Gemini AI, you can find it in assets folder)
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


  # SIDEBAR (clear chat button in the sidebar to reset the chat history)
with st.sidebar:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
        

   # MAIN CONTAINER 
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

##Here we start th main code for the chatbot 


# ================= CHECK PDF =================
if not os.path.exists(PDF_PATH):
    st.error(" PDF file not found!")
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

    chunks = []
    for i, page in enumerate(pages):
        split_page = splitter.split_documents([page])
        for c in split_page:
            c.metadata["page"] = i + 1   #  store original page number( I did this to restore original PDF page)
            chunks.append(c)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
    model="gpt-4o-mini", #its from OpenAI
    temperature=0,   #temperature 0 for more accurate answers
    openai_api_key=OPENAI_KEY #My API key
)

# Increased k for better results 
retriever = vectorstore.as_retriever(search_kwargs={"k": 15}) #--> k is the number of chunks to retrieve

def format_docs(docs):
    formatted = ""
    for d in docs:
        page = d.metadata.get("page", "?")
        formatted += f"\n\n[Page {page}]\n{d.page_content}"
    return formatted

# ================= QA CHAIN =================
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)
## ================= EXTRACT CITATIONS =================
def extract_citations(docs, answer):
    """
    Evidence-based citation selection:
    - Detects key concepts in the answer
    - Matches only strong medical evidence terms
    - Avoids generic, repeated vocabulary
    - Returns only the true supporting pages
    """ #----> If answer is "Not found" → do NOT cite anything
    if "Not found in the provided document" in answer:
        return "📘 Source: Not applicable"

    # --- 1. Extract meaningful short phrases from the answer ---
    # Keep 2-word and 3-word phrases to detect medical definitions
    words = answer.lower().split()
    phrases = set()

    for i in range(len(words) - 1):
        phrases.add(f"{words[i]} {words[i+1]}")
    for i in range(len(words) - 2):
        phrases.add(f"{words[i]} {words[i+1]} {words[i+2]}")

    # Some generic words to ignore because they appear on every page
    stopwords = {
        "patient", "skin", "organisms", "infection", "hands",
        "care", "health", "worker", "workers", "transmission",
        "associated", "flora", "pathogens"
    }

    # Keep only meaningful terms
    key_terms = [
        p for p in phrases
        if not any(w in stopwords for w in p.split())
        and len(p.split()[0]) > 2
    ]

    # If nothing found, fallback to single important words
    if not key_terms:
        key_terms = [w for w in words if len(w) > 5]

    # --- 2. Score pages based on strong term matches ---
    page_scores = {}

    for d in docs:
        text = d.page_content.lower()
        page = d.metadata.get("page")

        score = 0
        for phrase in key_terms:
            if phrase in text:
                score += 5  # strong match

        # whole-word fallback match
        for w in phrase.split():
            if len(w) > 3 and re.search(rf"\b{re.escape(w)}\b", text):
                score += 1

        if score > 0:
            page_scores[page] = score

    # --- 3. If nothing matched, fallback to first retrieved ---
    if not page_scores:
        fallback = sorted({d.metadata["page"] for d in docs})[:2]
        return f"📘 Source: CDC Hand Hygiene Guideline 📄 Page(s): {', '.join(map(str, fallback))}"

    # --- 4. Sort by highest score ---
    ranked = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)

    # --- 5. Select only top relevant pages ---
    # Stop when score drops sharply
    top_pages = []
    highest = ranked[0][1]

    for p, s in ranked:
        if s >= highest * 0.35:   # only significantly relevant
            top_pages.append(p)

    # Limit maximum to 5
    top_pages = top_pages[:5]

    # Sort numerically for clean output
    top_pages = sorted(top_pages)

    return f"📘 Source: CDC Hand Hygiene Guideline 📄 Page(s): {', '.join(map(str, top_pages))}"



 #TYPING ANIMATIONn(addional feature)
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



   #TITLE 
st.title("🩺 Medical Document Q&A Chatbot")


   #ROBOT GIF (additional feature)
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


#SHOW CHAT HISTORY (additional feature+ so when the user write other question the previous chat will be shown)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-box">{question}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-box">{answer}</div>', unsafe_allow_html=True)


#USER INPUT 
query = st.chat_input("Ask Your Medical Policy Question...")



# ================= HANDLE QUERY =================
if query:

    # Show user message
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-box">{query}</div>', unsafe_allow_html=True)

    # ⿡ Retrieve docs for citation
    retrieved_docs = retriever.invoke(query)
    
    # ⿢ Get answer from QA chain (LLM + context)
    result = qa_chain.invoke(query)
    answer = result.content.strip()
    # ⿣ Extract citations based on the actual answer
    citation = extract_citations(retrieved_docs, answer)

    # ⿣ Combine answer + citation
    full_answer = answer + "\n\n" + citation

    # ⿤ Show assistant message
    with st.chat_message("assistant"):
        typewriter(full_answer)

    # ⿥ Save in history
    st.session_state.chat_history.append((query, full_answer))



# ================= END MAIN CONTAINER =================
st.markdown('</div>', unsafe_allow_html=True)