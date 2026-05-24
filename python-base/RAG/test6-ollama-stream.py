# 全功能离线RAG：TXT+PDF+Ollama+流式输出+对话记忆
# pip install pdfminer.six requests numpy faiss-cpu
import os
import numpy as np
import faiss
import requests
from pdfminer.high_level import extract_text

# 禁用模型联网下载
os.environ["HF_HUB_OFFLINE"] = "1"

# 文本向量化
def encode(text, dim=128):
    vec = np.zeros(dim, dtype=np.float32)
    for i, c in enumerate(text[:dim]):
        vec[i] = ord(c)
    return vec

# 加载TXT文档
def load_txt(file_path="knowledge.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]
    print(f"✅ 读取TXT：{len(chunks)}条")
    return chunks

# 加载PDF文档
def load_pdf(file_path="document.pdf"):
    if not os.path.exists(file_path):
        return []
    text = extract_text(file_path)
    chunks = [line.strip() for line in text.split("\n") if line.strip()]
    print(f"✅ 读取PDF：{len(chunks)}条")
    return chunks

# 向量库操作
def build_vector_db(docs):
    vec_arr = np.array([encode(doc) for doc in docs], dtype=np.float32)
    index = faiss.IndexFlatL2(128)
    index.add(vec_arr)
    return index

def save_index(index, path="local_vec.index"):
    faiss.write_index(index, path)

def load_index(path="local_vec.index"):
    return faiss.read_index(path) if os.path.exists(path) else None

# 文本相似度检索
def search_context(query, index, docs, top_k=2):
    query_vec = encode(query).reshape(1, -1)
    _, idx_list = index.search(query_vec, top_k)
    return [docs[idx] for idx in idx_list[0]]

# 流式调用Ollama
def stream_ollama(prompt, model_name="deepseek-r1:7b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }
    try:
        response = requests.post(url, json=payload, stream=True, timeout=1200)
        full_resp = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                import json
                data = json.loads(line)
                chunk = data.get("response", "")
                print(chunk, end="", flush=True)
                full_resp += chunk
        return full_resp.strip()
    except Exception:
        print("\n⚠️ Ollama服务未启动，请先执行ollama serve")
        return ""

# RAG问答整合对话记忆
def rag_chat(question, index, docs, history, model_name):
    context = search_context(question, index, docs)
    context_content = "\n".join(context)
    history_text = "\n".join([f"用户：{h[0]}\n助手：{h[1]}" for h in history])

    prompt = f"""基于参考资料回答问题，禁止编造内容
参考资料：{context_content}
历史对话：{history_text}
当前问题：{question}
回答："""
    print("AI回复：", end="")
    reply = stream_ollama(prompt, model_name)
    return reply

if __name__ == "__main__":
    print("=" * 60)
    print(" 全功能离线RAG系统 | TXT+PDF+流式对话+记忆")
    print("=" * 60)
    print("启动Ollama服务后再提问，quit退出\n")

    # 合并所有文档
    all_docs = []
    all_docs.extend(load_txt())
    all_docs.extend(load_pdf())

    # 无文档使用默认知识库
    if not all_docs:
        all_docs = [
            "Python是数据分析与人工智能主流编程语言。",
            "Numpy负责多维数值计算，Pandas处理表格数据。",
            "机器学习流程：数据预处理→数据集划分→模型训练→评估。",
            "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
            "过拟合解决方法：Dropout、L2正则、早停、简化网络结构。"
        ]

    index_file = "local_vec.index"
    vec_index = load_index(index_file)
    # 修改文档后删除索引文件自动重建
    if not vec_index:
        vec_index = build_vector_db(all_docs)
        save_index(vec_index)
        print("💾 向量库已保存\n")

    # 替换为你本地Ollama模型名
    use_model = "deepseek-r1:7b"
    chat_history = []

    # 交互式对话
    while True:
        user_input = input("\n请提问：")
        if user_input.lower() == "quit":
            print("👋 对话结束")
            break
        ans = rag_chat(user_input, vec_index, all_docs, chat_history, use_model)
        chat_history.append((user_input, ans))