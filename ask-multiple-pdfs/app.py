import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------- Utility Functions ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# ---------- Streamlit App ----------
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon="📚")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.header("📚 Chat With Multiple PDFs")

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.chat_history = []  # reset chat history
            st.success("Processing complete! Ask your questions below.")

    # Display conversation history above input
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

    # Input field at the bottom
    user_q = st.text_input("Ask a question:", key="user_input")

    if st.button("Send"):
        if user_q and st.session_state.conversation:
            response = st.session_state.conversation({"question": user_q})
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state.chat_history.append({"role": "bot", "content": response["answer"]})
            st.session_state.user_input = ""  # clear input
            st.experimental_rerun()
        elif not st.session_state.conversation:
            st.warning("Please upload and process PDFs first.")

if __name__ == '__main__':
    main()
