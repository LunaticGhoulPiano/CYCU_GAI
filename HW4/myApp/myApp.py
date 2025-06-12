import streamlit as st

st.set_page_config(page_title = "RAG中文問答系統", layout = "wide")
st.title("電資四11020107蘇伯勳 - 本地PDF中文問答系統")
st.markdown("""
1. **上傳PDF並建立向量索引庫**（左側選單->建立索引）
2. **進行問答查詢**（左側選單->問答查詢）
系統將使用FAISS向量資料庫 + 中文嵌入模型 + 本地Ollama模型（如mistral）進行回答。
---
文件儲存路徑：`documents/`
向量庫儲存路徑：`faiss_index/`
""")