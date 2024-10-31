import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

URL = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
loader = WebBaseLoader(
    web_paths=(URL,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("editedContent") 
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=".chroma_data"
)
retriever = vectorstore.as_retriever()
st.title("RAG")

if "llm" not in st.session_state:
    st.session_state["llm"] = "gpt-4o-mini"
    st.session_state["max_tokens"] = 512  
    st.session_state["temperature"] = 0.1  
    st.session_state["frequency_penalty"] = 0.0  

llm = ChatOpenAI(model=st.session_state["llm"],
                 max_tokens=st.session_state["max_tokens"],
                temperature=st.session_state["temperature"],
                frequency_penalty=st.session_state["frequency_penalty"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_text := st.chat_input("질문을 입력하세요"):
    
    with st.chat_message("user"):
        st.markdown(user_text)
        retrieved_docs = retriever.invoke(user_text)
        
        prompt = hub.pull("rlm/rag-prompt")
        user_prompt = prompt.invoke({
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": user_text
        })
        
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("assistant"):
        response_stream = llm.stream(user_prompt)
        response = st.write_stream(response_stream)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response
    })



            # "context": lambda retrieved_docs: "\n\n".join(doc.page_content for doc in retrieved_docs),
