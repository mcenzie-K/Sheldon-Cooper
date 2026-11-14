import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Safely check Streamlit Cloud Secrets without crashing locally when secrets.toml doesn't exist
if not os.getenv("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

st.set_page_config(page_title="Sheldon Cooper AI", page_icon="⚛️")
st.title("⚛️ Fun with Sheldon Cooper")

# 1. Cache models & vector databases so they only load ONCE
@st.cache_resource
def init_rag_and_models():
    model = ChatGroq(model="openai/gpt-oss-120b")
    embedding_fn = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vectorstore = Chroma(
        collection_name="chat_history",
        embedding_function=embedding_fn,
        persist_directory="./memory",
    )

    ragstore = Chroma(
        collection_name="rag_notes",
        embedding_function=embedding_fn,
        persist_directory="./rag_memory",
    )

    # Use a relative path so it works both locally and on cloud servers
    notes_path = "chat_history.txt"
    if ragstore._collection.count() == 0 and os.path.exists(notes_path):
        loader = TextLoader(notes_path, encoding="utf-8")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        ragstore.add_documents(chunks)

    return model, vectorstore, ragstore

model, vectorstore, ragstore = init_rag_and_models()

# 2. Keep short-term UI chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Optional sidebar showing memory/RAG stats + clear chat button
with st.sidebar:
    st.header("Brain Stats")
    st.write(f"Long-term memories stored: **{vectorstore._collection.count()}**")
    st.write(f"RAG note chunks loaded: **{ragstore._collection.count()}**")
    if st.button("Clear Screen Chat"):
        st.session_state.messages = []
        st.rerun()

# Render existing messages on screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Handle new user input
if user_input := st.chat_input("Say something to Dr. Sheldon Cooper..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Search past memories & RAG notes BEFORE saving the current message
    relevant_memory = vectorstore.similarity_search_with_score(user_input, k=2)
    rag_results = ragstore.similarity_search_with_score(user_input, k=2)

    memories = [r.page_content for r, score in relevant_memory if score < 0.5]
    notes = [doc.page_content for doc, score in rag_results if score < 1.5]

    # Build the prompt cleanly for this turn without polluting session history
    system_prompt = "You are Sheldon Cooper. Stay completely in character."
    if memories:
        system_prompt += "\n\nRelevant past conversation memories:\n" + "\n".join(f"- {m}" for m in memories)
    if notes:
        system_prompt += "\n\nRelevant background notes:\n" + "\n".join(f"- {n}" for n in notes)

    llm_messages = [SystemMessage(content=system_prompt)]
    for m in st.session_state.messages:
        if m["role"] == "user":
            llm_messages.append(HumanMessage(content=m["content"]))
        else:
            llm_messages.append(AIMessage(content=m["content"]))

    # Generate and display Sheldon's response
    with st.chat_message("assistant"):
        with st.spinner("Sheldon is formulating a superior response..."):
            result = model.invoke(llm_messages)
            st.markdown(result.content)

    # Save to UI session and long-term Chroma memory
    st.session_state.messages.append({"role": "assistant", "content": result.content})
    vectorstore.add_texts(
        texts=[user_input, result.content],
        metadatas=[{"role": "user"}, {"role": "ai"}],
    )
