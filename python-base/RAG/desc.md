RAG 检索增强生成 入门实战
先理清核心概念，再上手可运行代码，衔接前面深度学习基础，全程注释 + 参数详解
一、RAG 核心认知
传统大模型问题：知识截止固定时间、易幻觉、无法私有知识库问答
RAG 作用：先检索私有文档真实内容，再交给模型生成回答，减少胡说、贴合自有数据
标准流程
文档切块 → 向量化嵌入 → 存入向量库 → 问题向量化检索 → 拼接上下文 + 提问 → LLM 生成答案
二、环境安装
激活虚拟环境后执行
bash
运行
pip install langchain faiss-cpu sentence-transformers


##2
真实 LLM 接入 + 本地开源大模型 RAG 实战
脱离模拟模型，接入真实推理能力，完整可运行代码，逐行注释参数
一、安装额外依赖
bash
运行
pip install sentence-transformers faiss-cpu transformers torch

