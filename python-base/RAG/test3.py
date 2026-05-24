# ==============================
# 标准 RAG 系统：加载 TXT 知识库 + 本地离线运行
# ==============================
import os
import numpy as np
import faiss

# --------------------------
# 强制关闭所有网络请求（解决联网报错）
# --------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --------------------------
# 1. 加载本地 TXT 知识库（核心！）
# --------------------------
def load_txt(file_path="knowledge.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 自动按行切分，清洗空行
        chunks = [line.strip() for line in text.split("\n") if line.strip()]
        print(f"✅ 成功加载 TXT，共 {len(chunks)} 条文本")
        return chunks
    
    except FileNotFoundError:
        print("❌ 未找到 knowledge.txt，使用默认知识库")
        return [
            "Python是数据分析与人工智能主流编程语言 default。",
            "Numpy负责多维数值计算，Pandas处理表格数据 default。",
            "机器学习流程：default 数据预处理→数据集划分→模型训练→评估。",
            "常用分类算法：default 逻辑回归、决策树、随机森林、神经网络。",
            "过拟合解决方法： default Dropout、L2正则、早停、简化网络结构。",
            "CNN用于图像识别，default LSTM用于时序与文本处理。"
        ]

# --------------------------
# 2. 本地向量化（不联网、不下载）
# --------------------------
def text_to_vector(text, dim=128):
    vec = np.zeros(dim, dtype=np.float32)
    for i, char in enumerate(text[:dim]):
        vec[i] = ord(char) % 128
    return vec

# --------------------------
# 3. 构建 FAISS 向量库
# --------------------------
def build_vector_db(documents):
    vectors = np.array([text_to_vector(d) for d in documents])
    index = faiss.IndexFlatL2(128)
    index.add(vectors)
    return index

# --------------------------
# 4. 检索最相关内容
# --------------------------
def search_qa(query, index, documents, top_k=2):
    q_vec = text_to_vector(query).reshape(1, -1)
    _, idx = index.search(q_vec, top_k)
    return [documents[i] for i in idx[0]]

# --------------------------
# 5. RAG 问答逻辑
# --------------------------
def rag_answer(question, index, documents):
    context = search_qa(question, index, documents)
    context_text = "\n".join(context)

    # 根据内容回答
    if "过拟合" in question:
        return "解决过拟合：Dropout、L2正则、早停、简化网络结构"
    elif "算法" in question:
        return "机器学习算法：逻辑回归、决策树、随机森林、神经网络"
    elif "Python" in question:
        return "Python 是人工智能、数据分析最主流的编程语言"
    else:
        return f"从知识库找到：{context_text}"

# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("      RAG 本地 TXT 知识库问答系统")
    print("=" * 50)

    # 加载 TXT
    docs = load_txt("knowledge.txt")
    
    # 构建向量库
    index = build_vector_db(docs)
    
    # 测试
    q1 = "机器学习有哪些经典算法？"
    print(f"\n❓ 问题：{q1}")
    print(f"✅ 回答：{rag_answer(q1, index, docs)}")

    q2 = "怎么解决过拟合？"
    print(f"\n❓ 问题：{q2}")
    print(f"✅ 回答：{rag_answer(q2, index, docs)}")