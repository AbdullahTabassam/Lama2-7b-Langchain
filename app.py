import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
import os
import torch


def model_response(query,docs_page_content):

### Check if GPU is available run the model on GPU
    if torch.cuda.is_available():
        config={'max_new_tokens':1024,
                              'context_length': 4096,
                              'temperature':0.05,
                              'gpu_layers': 10}
    else:
        config={'max_new_tokens':2048,
                              'context_length': 4096,
                              'temperature':0.05}

    ### LLama2 model
    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config=config)
    
    template= """
                You are a helpful assistant that that can answer questions about documents using the provided transcripts. Answer the following question: {query} by searching the following transcript: {docs_page_content}
                Only use the factual information from the transcript to answer the question. If you feel like you don't have enough information to answer the question, say "I don't know". Your answers should be verbose and detailed.
                """
    prompt=PromptTemplate(input_variables=["query","docs_page_content"],
                          template=template)

    ## Generate the response from the LLama 2 model
    response=llm(prompt.format(query=query,docs_page_content=docs_page_content))

    ## Remove the pdf file after the response is generated
    if response:
        if doc_type == 'PDF File':
            os.remove("files/pdf_file.pdf")

    container = st.container(border=True)
    container.write(response) 

### Function to find similarity between the query and the document
    
def similarity_finder(data):

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    docs=text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    query_similarity = db.similarity_search(query, k= 4)
    docs_page_content = " ".join([d.page_content for d in query_similarity])
    return docs_page_content 

docs_page_content = ' '
embeddings=HuggingFaceEmbeddings()      ## Vector Embeddings from HuggingFace
   
st.title('InsightFlow')
st.markdown(' #####  Conversational Learning Companion')        
st.markdown(' ###### This web app helps you understand the concepts better.')


with st.sidebar:
    st.subheader('Inputs Tab', divider=True)
    doc_type = st.radio('Select source:', ('Select Source', 'Youtube Video', 'PDF File')) 

## For YT video
if doc_type == 'Youtube Video':
    query = st.text_area('What is your question?',placeholder='Ask question from the uploaded pdf')
    with st.sidebar:
        yt_link = st.text_input('YouTube Link:', placeholder='https://www.youtube.com/watch?v=AbCxYz')
        if st.button('Submit'):
            if yt_link is not None:
                if query != '':
                    loader = YoutubeLoader.from_youtube_url(yt_link)
                    data = loader.load()
                    docs_page_content = similarity_finder(data)
    if yt_link is not None:
        if query != '':
            model_response(query,docs_page_content)
                    

## For PDF file
elif doc_type == 'PDF File':
    query = st.text_area('What is your question?','', placeholder='Ask question from the uploaded pdf')
    with st.sidebar:
        pdf_file = st.file_uploader("Choose a PDF file:", accept_multiple_files=False, type=['pdf'])
        if pdf_file is not None:
            with open('files/pdf_file.pdf', "wb") as f:
                f.write(pdf_file.read())
        if st.button('Submit'):
            if pdf_file is not None:
                if query != '':
                    loader = PyPDFLoader("files/pdf_file.pdf")
                    data = loader.load()
                    docs_page_content = similarity_finder(data)
    if pdf_file is not None:
        if query != '':
            model_response(query,docs_page_content)

                  
elif doc_type == 'Select Source':
    with st.sidebar:
        if st.button('Submit'):
            st.write('Please choose a source.')

















