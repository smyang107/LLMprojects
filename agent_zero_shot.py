'''
Agent 的核心组件包括：

LLM (Large Language Model): 作为核心的推理引擎，通常由 OpenAI、Hugging Face 或其他大语言模型提供。
工具 (Tools): Agent 可以使用一系列预定义的工具，如网络搜索、计算工具、数据库、API 接口等。
策略 (Policies): Agent 根据策略（如用户输入或环境）决定调用哪些工具以及以何种顺序调用。
2. LangChain 中的常用 Agents
LangChain 提供了几种常见的 Agent 类型，每种类型适用于不同场景：

零射程代理 (Zero-shot Agent): 无需预先提供任何背景信息，Agent 直接基于输入内容和工具调用进行推理。通常用于简单任务。
反射链代理 (ReAct Agent): ReAct 是一种先进的策略，Agent 在生成答案时会一步一步地解释其推理过程，便于处理复杂任务。
工具链代理 (Tool Agent): 允许 Agent 动态调用一系列工具完成任务，比如搜索引擎、API 等。

'''
import os
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI,HuggingFaceHub
#from langchain_experimental import PythonREPLTool
from langchain.tools import PythonREPLTool

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 使用 Hugging Face 上的模型
llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7}) #repo_id 模型的ID

# 创建一个计算器工具
tools = [
    Tool(
        name="Calculator",
        func=PythonREPLTool().run,
        description="Use this tool for mathematical calculations"
    )
]

# 初始化 Agent，使用零射程模式并加载工具
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,#AgentType.
    verbose=True
)

# 测试使用
result = agent.run("What is 25 raised to the power of 3?")
print(result)