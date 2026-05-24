import requests
import json
import re

# 本地 Ollama 服务
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"  # 改成你本地的模型

# --------------------------
# 工具：计算器
# --------------------------
def calculator(expression):
    try:
        expression = re.sub(r"[^\d\+\-\*/\(\)]", "", expression)
        return str(eval(expression.strip()))
    except:
        return "计算错误"

# --------------------------
# 工具执行
# --------------------------
def use_tool(tool_name, params):
    if tool_name == "calculator":
        return calculator(params)
    return "无工具"

# --------------------------
# Agent 判断是否计算（超级稳定）
# --------------------------
def agent_think(question):
    q = question.lower()
    if any(k in q for k in ["+", "-", "*", "/", "等于", "多少", "加", "减", "乘", "除"]):
        return {"tool": "calculator", "params": question}
    return {"tool": "none", "params": ""}

# --------------------------
# Agent 主流程
# --------------------------
def agent_run(question):
    plan = agent_think(question)

    if plan["tool"] == "calculator":
        res = calculator(question)
        return f"计算结果：{res}"
    else:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": question,
            "stream": True
        })
        return resp.json()["response"].strip()

# --------------------------
# 测试
# --------------------------
if __name__ == "__main__":
    print(agent_run("35 * 78 等于多少？"))
    print(agent_run("什么是RAG？"))