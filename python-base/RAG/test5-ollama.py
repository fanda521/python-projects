import os
import numpy as np
import faiss
import requests

# --------------------------
# 1. 加载 TXT 知识库
# --------------------------
def load_txt(file_path="knowledge.txt"):
    if not os.path.exists(file_path):
        print("❌ 未找到 knowledge.txt")
        return [
            "Python是数据分析与人工智能主流编程语言。",
            "Numpy负责多维数值计算，Pandas处理表格数据。",
            "机器学习流程：数据预处理→数据集划分→模型训练→评估。",
            "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
            "过拟合解决方法：Dropout、L2正则、早停、简化网络结构。"
        ]
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]
    print(f"✅ 加载 TXT 成功：{len(chunks)} 条")
    return chunks

# --------------------------
# 2. 向量化 + 向量库
# --------------------------
def encode(text, dim=128):
    vec = np.zeros(dim, dtype=np.float32)
    for i, c in enumerate(text[:dim]):
        vec[i] = ord(c)
    return vec

def build_db(docs):
    vecs = np.array([encode(d) for d in docs])
    index = faiss.IndexFlatL2(128)
    index.add(vecs)
    return index

# --------------------------
# 3. 检索
# --------------------------
def retrieve(query, index, docs, top_k=2):
    q_vec = encode(query).reshape(1, -1)
    _, idx = index.search(q_vec, top_k)
    return [docs[i] for i in idx[0]]

# --------------------------
# 4. 调用 OLLAMA 本地模型（已100%修复）
# --------------------------
def ollama_chat(prompt, model="deepseek-r1:7b"):
    try:
        url = "http://127.0.0.1:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(url, json=data, timeout=1200)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print("🔴 真实错误信息：", str(e))  # 现在会显示真实错误！
        return "⚠️ 连接 Ollama 失败"

# --------------------------
# 5. RAG 问答
# --------------------------
def rag_qa(question, index, docs):
    context = retrieve(question, index, docs)
    context = "\n".join(context)

    prompt = f"""
根据以下资料回答问题，不要编造。

资料：
{context}

问题：{question}
回答：
"""
    return ollama_chat(prompt, model="deepseek-r1:7b")


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("     RAG + Ollama 本地知识库问答")
    print("=" * 50)

    docs = load_txt("knowledge.txt")
    index = build_db(docs)

    while True:
        q = input("请提问：")
        if q.lower() == "quit":
            break
        print("AI：", rag_qa(q, index, docs), "\n")