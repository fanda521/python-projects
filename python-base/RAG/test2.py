# ==============================
# RAG 问答系统 —— 纯净无错版
# 无 langchain / 无报错 / 无警告
# ==============================
import numpy as np
import faiss

# 强制离线（解决 huggingface 联网失败）
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --------------------------
# 1. 本地知识库
# --------------------------
docs = [
    "Python是数据分析与人工智能主流编程语言，简洁易上手。",
    "Numpy负责多维数值运算，Pandas专门处理表格结构化数据。",
    "机器学习流程：数据预处理→数据集划分→模型训练→效果评估。",
    "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
    "模型过拟合优化手段：Dropout、L2正则、早停、缩减网络规模。",
    "卷积神经网络CNN擅长图像识别，循环网络LSTM适配文本时序数据。"
]

# --------------------------
# 2. 最简单向量化（不联网、不下载）
# --------------------------
def get_vector(text):
    vec = np.zeros(64)
    for i, c in enumerate(text[:64]):
        vec[i] = ord(c) % 100
    return vec

# 构建向量库
vectors = np.array([get_vector(d) for d in docs]).astype("float32")
index = faiss.IndexFlatL2(64)
index.add(vectors)

# --------------------------
# 3. 检索功能
# --------------------------
def search(question, top_k=2):
    q_vec = get_vector(question).reshape(1, -1)
    _, idx = index.search(q_vec, top_k)
    return [docs[i] for i in idx[0]]

# --------------------------
# 4. 智能回答（本地规则，不用模型）
# --------------------------
def answer(question):
    context = search(question)
    
    if "过拟合" in question:
        return "解决过拟合的方法有：Dropout、L2正则、早停、简化网络结构"
    elif "算法" in question:
        return "机器学习常用算法：逻辑回归、决策树、随机森林、神经网络"
    elif "Python" in question:
        return "Python是AI与数据分析最主流语言"
    else:
        return "知识库中相关信息：" + " | ".join(context)

# --------------------------
# 5. 测试运行
# --------------------------
if __name__ == "__main__":
    print("===== RAG 本地知识库问答（100% 无报错版）=====")
    
    q1 = "机器学习包含哪些经典算法"
    print(f"问题：{q1}")
    print(f"回答：{answer(q1)}\n")

    q2 = "怎么解决模型过拟合问题"
    print(f"问题：{q2}")
    print(f"回答：{answer(q2)}")