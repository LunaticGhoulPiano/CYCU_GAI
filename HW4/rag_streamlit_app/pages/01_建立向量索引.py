import streamlit as st
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import shutil

st.set_page_config(page_title = "建立向量索引", layout = "wide")
st.header("上傳PDF並建立向量索引庫")

embedding = HuggingFaceEmbeddings(
    model_name = "shibing624/text2vec-base-chinese",
    encode_kwargs = {"normalize_embeddings": True},
    model_kwargs = {"device": "cpu"}
)
doc_dir = Path("documents")
doc_dir.mkdir(exist_ok = True)

uploaded_files = st.file_uploader("請選擇PDF檔案：", type = "pdf", accept_multiple_files = True)
if uploaded_files:
    if st.button("開始建立索引"):
        with st.spinner("處理與分段中..."):
            documents = []
            for file in uploaded_files:
                file_path = doc_dir / file.name
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file, f)
                loader = PyMuPDFLoader(str(file_path))
                pages = loader.load()
                for page in pages:
                    page.metadata["source"] = file.name
                    documents.append(page)
            splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
            docs = splitter.split_documents(documents)
            db = FAISS.from_documents(docs, embedding)
            db.save_local("faiss_index")
            st.success(f"共上傳{len(uploaded_files)}份PDF，分段為{len(docs)}段，已成功建立向量庫")
            st.balloons()