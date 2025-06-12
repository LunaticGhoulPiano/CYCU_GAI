import streamlit as st
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title = "中文問答查詢", layout = "wide")
st.header("電資四11020107蘇伯勳 - 中文問答查詢介面")

llm = Ollama(model = "mistral")
embedding = HuggingFaceEmbeddings(
    model_name = "shibing624/text2vec-base-chinese",
    encode_kwargs = {"normalize_embeddings": True},
    model_kwargs = {"device": "cpu"}
)

prompt = PromptTemplate.from_template(
    """
    你是一位問答助手，只能使用繁體中文根據以下資料回答問題，不可以使用任何額外知識。
    若資料不足以回答，請誠實回覆「無法從提供資料中找到答案」。
    【資料】：
    {context}
    【問題】：{question}
    【答案】：
    """
)

def rag_ask(question: str, source: list):
    db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization = True)
    retriever = db.as_retriever()
    all_docs = retriever.invoke(question)
    filtered_docs = [doc for doc in all_docs if doc.metadata["source"] in source]
    filtered_docs = sorted(filtered_docs, key = lambda x: x.metadata.get("page", 99))
    if not filtered_docs:
        return "未提供欲查詢的PDF！", "", []
    
    context_parts = []
    for doc in filtered_docs:
        page = doc.metadata.get("page", "未知頁碼")
        source = doc.metadata.get("source", "未知來源")
        context_parts.append(f"【來源：{source} - 第{page}頁】\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        chain: Runnable = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return response, context, filtered_docs

# UI
doc_dir = Path("documents")
all_pdfs = [f.name for f in doc_dir.glob("*.pdf")]
selected_files = st.multiselect("請選擇要檢索的PDF檔案：", all_pdfs, default = all_pdfs[:1])
question = st.text_input("請輸入問題：", "和弦代號中什麼是omit？")
if st.button("查詢"):
    if not selected_files:
        st.warning("請選擇一份PDF檔案")
    else:
        with st.spinner("查詢中..."):
            answer, context, docs = rag_ask(question, selected_files)
        st.subheader("回答：")
        st.markdown(answer)
        with st.expander("檢索段落（來源與頁碼）"):
            for doc in docs:
                source = doc.metadata.get("source", "?")
                page = doc.metadata.get("page", "?")
                st.markdown(f"【來源：{source} - 第{page}頁】\n\n{doc.page_content}")
        with st.expander("Prompt Context 原文"):
            st.code(context)