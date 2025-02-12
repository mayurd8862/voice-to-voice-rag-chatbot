import streamlit as st
from src.speech_to_text import transcribe_audio
from src.text_to_speech import text_to_speech_file
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq


import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import tempfile
import asyncio

from langchain_community.document_loaders import UnstructuredMarkdownLoader

# Load environment variables
load_dotenv()

# Initialize Groq
llm = ChatGroq(model_name="Llama3-8b-8192")
client = Groq()


# Header
st.title("🎙️ Voice to voice Chatbot")
# st.markdown("Speak your message to chat!")

@st.cache_resource
def load_and_process_data(files):
    all_splits = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            file_path = temp_file.name
        
        # loader = PyPDFLoader(file_path)
        loader = UnstructuredMarkdownLoader(file_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(data)
        all_splits.extend(splits)
    
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(all_splits, embedding)
    return vectordb


async def response_generator(vectordb, query):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    
    result = await asyncio.to_thread(qa_chain, {"query": query})
    return result["result"]

files = st.file_uploader("Upload PDF File(s)", type=["pdf","md"], accept_multiple_files=True)
submit_pdf = st.checkbox('Submit and chat (PDF)')
st.subheader('',divider="rainbow")

# Page config
# st.set_page_config(page_title="Voice Chatbot", page_icon="🎙️")

if files and submit_pdf:

    vectordb = load_and_process_data(files)
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # Audio input
    audio_value = st.audio_input("Record a voice message")

    if audio_value:
        with st.spinner("Processing voice message..."):
            # Save audio to temporary file
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_value.read())
            
            # Transcribe audio
            transcript = transcribe_audio(client, webm_file_path)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(transcript)
            st.session_state.messages.append({"role": "user", "content": transcript})
            
            # Generate response
            with st.spinner("Thinking..."):
                output = asyncio.run(response_generator(vectordb, transcript))
                
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(output)
                audio_buffer = text_to_speech_file(output)
                st.audio(audio_buffer, format="audio/mp3", autoplay=True)
            
            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": output,
                "audio": audio_buffer
            })



