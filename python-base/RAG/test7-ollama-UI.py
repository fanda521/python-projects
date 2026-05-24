# pip install streamlit pdfminer.six requests numpy faiss-cpu
import os
import json
import numpy as np
import faiss
import requests
import streamlit as st
from pdfminer.high_level import extract_text

# 禁用外网模型下载
os.environ["HF_HUB_OFFLINE"] = "1"
VEC_SAVE_PATH = "local_vec.index"
VEC_DIM = 128

# 文本向量化
def encode(text):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, c in enumerate(text[:VEC_DIM]):
        vec[i] = ord(c)
    return vec

# 读取TXT
def load_txt_content(file_bytes):
    try:
        return [line.strip() for line in file_bytes.decode("utf-8").splitlines() if line.strip()]
    except:
        return []

# 读取PDF
def load_pdf_content(temp_path):
    try:
        text = extract_text(temp_path)
        return [line.strip() for line in text.splitlines() if line.strip()]
    except:
        return []

# 向量库操作
def build_index(docs):
    vecs = np.array([encode(d) for d in docs], dtype=np.float32)
    index = faiss.IndexFlatL2(VEC_DIM)
    index.add(vecs)
    faiss.write_index(index, VEC_SAVE_PATH)
    return index

def get_index():
    if os.path.exists(VEC_SAVE_PATH):
        return faiss.read_index(VEC_SAVE_PATH)
    return None

# 相似度检索
def search_relevant(query, index, docs, top_k=2):
    q_vec = encode(query).reshape(1, -1)
    _, idx = index.search(q_vec, top_k)
    return [docs[i] for i in idx[0]]

# 流式调用Ollama
def ollama_stream(prompt, model_name):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": True}
    resp = requests.post(url, json=payload, stream=True, timeout=1200)
    full_text = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            chunk = data.get("response", "")
            full_text += chunk
            yield chunk
    return full_text

# 页面初始化配置
st.set_page_config(page_title="本地知识库RAG", layout="wide")
st.title("📚 离线RAG知识库问答系统")

# 侧边栏配置
with st.sidebar:
    st.header("参数配置")
    model_name = st.text_input("Ollama模型名称", value="deepseek-r1:7b")
    uploaded_file = st.file_uploader("上传TXT/PDF文档", type=["txt","pdf"])
    rebuild_btn = st.button("重新构建向量库")

# 初始化会话存储
if "doc_list" not in st.session_state:
    st.session_state.doc_list = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 上传文件处理
if uploaded_file:
    suffix = uploaded_file.name.split(".")[-1]
    temp_file = f"temp.{suffix}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())
    if suffix == "txt":
        new_docs = load_txt_content(uploaded_file.getvalue())
    else:
        new_docs = load_pdf_content(temp_file)
    os.remove(temp_file)
    st.session_state.doc_list.extend(new_docs)
    st.success(f"成功加载{len(new_docs)}条文本")

# 重建向量库
if rebuild_btn and st.session_state.doc_list:
    build_index(st.session_state.doc_list)
    st.success("向量库重建完成")

# 聊天界面展示
for user_msg, ai_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(ai_msg)

# 提问输入
user_query = st.chat_input("输入你的问题...")
if user_query:
    st.chat_message("user").write(user_query)
    index = get_index()
    if not index or not st.session_state.doc_list:
        st.warning("请先上传文档并构建向量库")
    else:
        # 检索上下文
        context = search_relevant(user_query, index, st.session_state.doc_list)
        context_str = "\n".join(context)
        # 拼接历史对话
        history_str = ""
        for u,a in st.session_state.chat_history:
            history_str += f"用户：{u}\n助手：{a}\n"
        # 构造提示词
        prompt = f"""依据参考资料回答，禁止编造信息
参考资料：{context_str}
历史对话：{history_str}
当前问题：{user_query}
回答："""
        # 流式输出回复
        with st.chat_message("assistant"):
            resp_placeholder = st.empty()
            full_response = ""
            for chunk in ollama_stream(prompt, model_name):
                full_response += chunk
                resp_placeholder.markdown(full_response)
        # 存入对话记录
        st.session_state.chat_history.append((user_query, full_response))