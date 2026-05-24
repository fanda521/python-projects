import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"

# ---------- 工具：计算器 ----------
def calculator(expression):
    try:
        expr = re.sub(r"[^\d\+\-\*/\(\)]", "", expression)
        return str(eval(expr))
    except:
        return "计算错误"

# ---------- 工具：读文件 ----------
def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "文件不存在或读取失败"

# ---------- 工具：写文件 ----------
def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "写入成功"
    except:
        return "写入失败"

# ---------- Agent 判断逻辑 ----------
def agent_route(question):
    q = question.lower()
    # 计算器
    if any(k in q for k in ["+", "-", "*", "/", "等于", "多少", "加", "减", "乘", "除"]):
        return ("calc", question)
    # 读文件
    elif "读取" in q and "文件" in q:
        # 提取文件名（简单匹配）
        m = re.search(r"(\w+\.txt)", question)
        if m:
            return ("read", m.group(1))
        else:
            return ("none", "")
    # 写文件
    elif "写入" in q and "文件" in q:
        m = re.search(r"(\w+\.txt)", question)
        if m:
            # 简易提取内容：“写入 a.txt：内容”
            content = question.split("：")[-1] if "：" in question else ""
            return ("write", (m.group(1), content))
        else:
            return ("none", "")
    # 直接回答
    else:
        return ("llm", question)

# ---------- Agent 主逻辑 ----------
def agent_run(question):
    route, data = agent_route(question)
    if route == "calc":
        return f"计算结果：{calculator(data)}"
    elif route == "read":
        return f"文件内容：\n{read_file(data)}"
    elif route == "write":
        path, content = data
        return write_file(path, content)
    elif route == "llm":
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": data, "stream": False
        })
        return resp.json()["response"].strip()
    else:
        return "无法识别指令"

# ---------- 测试 ----------
if __name__ == "__main__":
    print(agent_run("35 * 78 等于多少？"))
    print(agent_run("读取 knowledge.txt"))
    print(agent_run("写入 note.txt：Java+AI 很强大"))
    print(agent_run("什么是RAG？"))