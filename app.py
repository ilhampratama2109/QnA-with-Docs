from PyPDF2 import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
import streamlit as st
import os

working_dir = os.path.dirname(os.path.abspath(__file__))


def load_document(file_path):
    pdf_reader = PdfReader(file_path)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    return raw_text


def setup_vectorstore(document):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    text = text_splitter.split_text(document)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    document_search = FAISS.from_texts(text, embeddings)
    return document_search


def create_chain(vectorstore):
    llm = Ollama(model="llama3.2", temperature=0.1)
    memory = ConversationBufferMemory(
        llm=llm, output_key="answer", memory_key="chat_history", return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory, verbose=True
    )
    return chain


st.title("🦙 QnA with Doc - LLMA")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

upload_file = st.file_uploader(label="Upload file here", type=["pdf"])

if upload_file:
    file_path = f"{working_dir}/{upload_file.name}"

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Bot...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_response}
        )
