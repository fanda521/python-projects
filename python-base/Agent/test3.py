import requests
import re
import numpy as np
import faiss

# ====================== 配置 ======================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"  # 你的模型名

# ====================== RAG 知识库模块 ======================
# 你的知识库
KNOWLEDGE_BASE = [
    "Python是数据分析与人工智能主流编程语言。",
    "Numpy负责多维数值计算，Pandas处理表格数据。",
    "机器学习流程：数据预处理→数据集划分→模型训练→评估。",
    "常用分类算法：逻辑回归、决策树、随机森林、神经网络。",
    "过拟合解决方法：Dropout、L2正则、早停、简化网络结构。"
]

# 向量化
def encode(text):
    vec = np.zeros(128, dtype=np.float32)
    for i, c in enumerate(text[:128]):
        vec[i] = ord(c)
    return vec

# 构建向量库
index = faiss.IndexFlatL2(128)
vecs = np.array([encode(d) for d in KNOWLEDGE_BASE])
index.add(vecs)

# 检索
def rag_search(question):
    q_vec = encode(question).reshape(1, -1)
    _, idx = index.search(q_vec, 2)
    return [KNOWLEDGE_BASE[i] for i in idx[0]]

# ====================== 工具集 ======================
def calculator(expression):
    try:
        expr = re.sub(r"[^\d\+\-\*/\(\)]", "", expression)
        return str(eval(expr))
    except:
        return "计算错误"

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "文件不存在"

def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "写入成功"
    except:
        return "写入失败"

# ====================== Agent 主逻辑（RAG + 工具 + LLM） ======================
def agent(question):
    q = question.lower()

    # 1. 数学计算
    if any(k in q for k in ["+", "-", "*", "/", "等于", "多少"]):
        return f"🧮 计算结果：{calculator(question)}"

    # 2. 读取文件
    if "读取" in q and ".txt" in q:
        match = re.search(r"(\w+\.txt)", question)
        if match:
            return f"📄 文件内容：\n{read_file(match.group(1))}"

    # 3. 写入文件
    if "写入" in q and ".txt" in q:
        match = re.search(r"(\w+\.txt)", question)
        if match and "：" in question:
            content = question.split("：")[-1]
            return f"✅ {write_file(match.group(1), content)}"

    # 4. RAG 检索知识库
    context = rag_search(question)
    context_str = "\n".join(context)

    # 5. 交给大模型回答
    prompt = f"""根据以下资料回答问题，不要编造。
资料：{context_str}
问题：{question}
回答："""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=1200)
        return f"📚 AI回答：{resp.json().get('response', '无结果').strip()}"
    except:
        return "⚠️ 请启动 Ollama 服务"

# ====================== 测试 ======================
if __name__ == "__main__":
    print(agent("机器学习有哪些算法？"))  # RAG知识库
    print(agent("35*78等于多少？"))       # 计算器
    print(agent("读取 knowledge.txt"))    # 读文件
    print(agent("写入 note.txt：AI学习")) # 写文件