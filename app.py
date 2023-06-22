# Sidebar contents
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
from embeddings import persist_directory,embeddings
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


with st.sidebar:
    st.title('🤗💬 OPS-GPT')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)

    load_dotenv()

def main():
    st.header("Chat with Your Documents 💬")

    # uploadedfiles = st.file_uploader("Upload Docs", accept_multiple_files=True)
    # for file in uploadedfiles:
    #     if uploadedfiles is not None:
    #         save_uploadedfile(file)

    # Loads the emeddings locally
    db = None
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Accept user questions/query
    add_vertical_space(5)
    query = st.text_input("Ask questions about your Docs:")

    prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, 
        reply saying "I don't know", don't try to make up an answer.
        {context}
        Question: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}


    if query:
        qa = RetrievalQA.from_chain_type(ChatOpenAI(temperature=0.4), chain_type="stuff",
                                          retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                                          chain_type_kwargs=chain_type_kwargs)

        with get_openai_callback() as cb:
            query = query
            response = qa.run(query=query)
            print(cb)
            st.write(response)



def save_uploadedfile(uploadedfile):
    with open(os.path.join("Data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to Data".format(uploadedfile.name))



if __name__ == '__main__':
    main()