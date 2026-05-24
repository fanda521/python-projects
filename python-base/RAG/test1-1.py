# ==============================
# 标准 RAG 实现（干净、稳定、无报错）
# ==============================
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------
# 1. 知识库
# --------------------------
documents = [
    "Python是数据分析与人工智能主流编程语言。",
    "Numpy负责多维数值计算，Pandas处理表格数据。",
    "机器学习流程：数据预处理→数据集划分→模型训练→评估。",
    "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
    "过拟合解决方法：Dropout、L2正则、早停、简化网络结构。",
    "CNN用于图像识别，LSTM用于时序与文本处理。"
]

# --------------------------
# 2. 向量模型
# --------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# 3. 构建 FAISS 向量库
# --------------------------
embeddings = embed_model.encode(documents, convert_to_numpy=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype(np.float32))

# --------------------------
# 4. 检索模块
# --------------------------
def retrieve(query, k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, idx = index.search(q_emb.astype(np.float32), k)
    return [documents[i] for i in idx[0]]

# --------------------------
# 5. 大模型
# --------------------------
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

generate = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.1,
    repetition_penalty=1.1
)

# --------------------------
# 6. 标准 RAG 问答
# --------------------------
def rag(question):
    # 1. 检索
    context = retrieve(question)
    context = "\n".join(context)

    # 2. 构造提示词
    prompt = f"""
根据以下已知信息回答问题，不要编造。

已知信息：
{context}

问题：{question}
回答：
"""

    # 3. 生成
    output = generate(prompt)
    return output[0]["generated_text"].split("回答：")[-1].strip()

# --------------------------
# 7. 测试
# --------------------------
if __name__ == "__main__":
    print("===== 标准 RAG 问答系统 =====")
    q1 = "机器学习有哪些经典算法？"
    print(f"Q: {q1}")
    print(f"A: {rag(q1)}\n")

    q2 = "如何解决过拟合问题？"
    print(f"Q: {q2}")
    print(f"A: {rag(q2)}")