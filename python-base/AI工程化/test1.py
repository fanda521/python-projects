# pip install fastapi uvicorn requests numpy faiss-cpu

from fastapi import FastAPI
import requests
import re
import numpy as np
import faiss

# 初始化 FastAPI
app = FastAPI(title="RAG + Agent AI 服务", version="1.0")

# ===================== 配置（使用你微调后的模型）=====================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"  # 你自己微调的模型

# ===================== RAG 知识库 =====================
KNOWLEDGE_BASE = [
    "Python是数据分析与人工智能主流编程语言。",
    "Numpy负责多维数值计算，Pandas处理表格数据。",
    "机器学习流程：数据预处理→数据集划分→模型训练→评估。",
    "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
    "过拟合解决方法：Dropout、L2正则、早停、简化网络结构。"
]

def encode(text):
    vec = np.zeros(128, dtype=np.float32)
    for i, c in enumerate(text[:128]):
        vec[i] = ord(c)
    return vec

index = faiss.IndexFlatL2(128)
vecs = np.array([encode(d) for d in KNOWLEDGE_BASE])
index.add(vecs)

def rag_search(question):
    q_vec = encode(question).reshape(1, -1)
    _, idx = index.search(q_vec, 2)
    return [KNOWLEDGE_BASE[i] for i in idx[0]]

# ===================== 工具集 =====================
def calculator(expression):
    try:
        expr = re.sub(r"[^\d\+\-\*/\(\)]", "", expression)
        return str(eval(expr))
    except:
        return "计算错误"

# ===================== AI 主逻辑 =====================
def ai_answer(question):
    q = question.lower()
    # 数学计算
    if any(k in q for k in ["+", "-", "*", "/", "等于", "多少"]):
        return f"计算结果：{calculator(question)}"
    # RAG 检索
    context = rag_search(question)
    context_str = "\n".join(context)
    prompt = f"""根据资料回答，不要编造。
资料：{context_str}
问题：{question}
回答："""
    # 调用模型
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt, "stream": False
        })
        return resp.json().get("response", "模型无响应")
    except:
        return "错误：请启动 Ollama 服务"

# ===================== API 接口 =====================
@app.get("/ai")
def get_ai(question: str):
    result = ai_answer(question)
    return {"code": 200, "question": question, "answer": result}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)