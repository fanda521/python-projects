# ==============================
# RAG离线系统：适配合理检索阈值
# ==============================
import os
import numpy as np
import faiss

# 强制离线禁止联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 文本向量化
def encode(text, dim=128):
    vec = np.zeros(dim, dtype=np.float32)
    for i, c in enumerate(text[:dim]):
        vec[i] = ord(c) % 128
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

# 加载本地TXT知识库
def load_txt_knowledge(file_path="knowledge.txt"):
    if not os.path.exists(file_path):
        print("❌ 未找到knowledge.txt，使用默认知识库")
        return [
            "Python是数据分析与人工智能主流编程语言。",
            "Numpy负责多维数值计算，Pandas处理表格数据。",
            "机器学习流程：数据预处理→数据集划分→模型训练→评估。",
            "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
            "过拟合解决方法：Dropout、L2正则、早停、简化网络结构。",
            "CNN用于图像识别，LSTM用于时序与文本处理。"
        ]
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✅ 成功加载TXT，共{len(chunks)}条文本")
    return chunks

# 构建向量库
def build_vector_store(documents):
    vectors = np.array([encode(d) for d in documents])
    index = faiss.IndexFlatL2(128)
    index.add(vectors)
    return index

# 保存向量库
def save_index(index, save_path="vec_index.faiss"):
    faiss.write_index(index, save_path)
    print("💾 向量库已保存至本地")

# 加载向量库
def load_index(load_path="vec_index.faiss"):
    if os.path.exists(load_path):
        print("📂 读取本地已有向量库")
        return faiss.read_index(load_path)
    return None

# 调低阈值，放宽匹配范围
def retrieve(question, index, documents, top_k=2, threshold=1.2):
    q_vec = encode(question).reshape(1, -1)
    dis_arr, idx_arr = index.search(q_vec, top_k)
    res = []
    for dis, idx in zip(dis_arr[0], idx_arr[0]):
        if dis < threshold:
            res.append(documents[idx])
    return res

# 问答逻辑
def rag_chat(question, index, documents):
    context = retrieve(question, index, documents)
    if not context:
        return "⚠️ 知识库中暂无相关内容"
    
    q_low = question.lower()
    if "python" in q_low:
        return "✅ Python是数据分析与人工智能领域的主流编程语言"
    elif "过拟合" in question:
        return "✅ 解决过拟合：Dropout、L2正则、早停、简化网络结构"
    elif "算法" in question:
        return "✅ 机器学习算法：逻辑回归、决策树、随机森林、神经网络"
    elif "机器" in question:
        return "✅ 机器学习包含数据处理、模型训练与效果评估完整流程"
    elif "numpy" in q_low or "pandas" in q_low:
        return "✅ Numpy做多维数值运算，Pandas专注表格结构化数据处理"
    else:
        context_str = "\n".join(context)
        return f"📑 检索内容：\n{context_str}"

if __name__ == "__main__":
    print("=" * 60)
    print("    RAG知识库问答｜适配检索阈值版")
    print("=" * 60)
    print("输入quit退出对话\n")

    doc_list = load_txt_knowledge("knowledge.txt")
    index_file = "vec_index.faiss"
    # 清空旧库重建
    if os.path.exists(index_file):
        os.remove(index_file)
    faiss_index = build_vector_store(doc_list)
    save_index(faiss_index)

    while True:
        user_q = input("请提问：")
        if user_q.lower() == "quit":
            print("👋 对话结束")
            break
        res = rag_chat(user_q, faiss_index, doc_list)
        print(f"AI回复：{res}\n")