
import os
import pickle

import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import UnstructuredPowerPointLoader, DirectoryLoader, UnstructuredFileLoader


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# select which embeddings we want to use (GPT - ADA)

# select which embeddings we want to use (GPT - ADA)
embeddings = OpenAIEmbeddings()
# presist_directory is the Local Directory where embeddings will be saved
persist_directory = 'db'


loader = DirectoryLoader('./data')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
db.persist()

print("embeddings created ")

