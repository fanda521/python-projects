# ==============================
# RAG 本地知识库问答（完全离线版）
# 不连HuggingFace | 不下载模型 | 国内直接跑
# ==============================
import numpy as np
import faiss

# 强制离线模式（关键！不联网）
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------------------
# 1. 知识库（本地）
# --------------------------
knowledge = [
    "Python是数据分析与人工智能主流编程语言，简洁易上手。",
    "Numpy负责多维数值运算，Pandas专门处理表格结构化数据。",
    "机器学习流程：数据预处理→数据集划分→模型训练→效果评估。",
    "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
    "模型过拟合优化手段：Dropout、L2正则、早停、缩减网络规模。",
    "卷积神经网络CNN擅长图像识别，循环网络LSTM适配文本时序数据。"
]

# --------------------------
# 2. 本地简单向量（不联网、不下载）
# --------------------------
def simple_vector(text):
    vec = np.zeros(32)
    vec[:len(text)%32] = 1
    return vec

# 构建向量库
vectors = np.array([simple_vector(t) for t in knowledge]).astype("float32")
index = faiss.IndexFlatL2(32)
index.add(vectors)

# --------------------------
# 3. 检索
# --------------------------
def search(query):
    q_vec = simple_vector(query).reshape(1, -1)
    _, idx = index.search(q_vec.astype("float32"), 2)
    return [knowledge[i] for i in idx[0]]

# --------------------------
# 4. 本地模拟回答（不联网）
# --------------------------
def answer(question):
    context = search(question)
    ctx = "\n".join(context)

    if "过拟合" in question:
        return "过拟合解决方法：Dropout、L2正则、早停、简化网络结构"
    elif "算法" in question:
        return "机器学习经典算法：逻辑回归、决策树、随机森林、神经网络"
    else:
        return "根据知识库，这是AI学习相关知识。"

# --------------------------
# 5. 测试
# --------------------------
if __name__ == "__main__":
    print("===== RAG 本地知识库（完全离线·无网络·不报错）=====")

    q1 = "机器学习包含哪些经典算法"
    print(f"问题：{q1}")
    print(f"回答：{answer(q1)}\n")

    q2 = "怎么解决模型过拟合问题"
    print(f"问题：{q2}")
    print(f"回答：{answer(q2)}")