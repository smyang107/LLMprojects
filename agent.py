from langchain.agents import AgentType
from langchain import PromptTemplate, LLMChain
from langchain.utilities import DynamicTool

class ZeroShotAgent:
    def respond_to(self, llm, prompt):
        chain = LLMChain(llm, AgentType.ZERO_SHOT)
        return chain.run(prompt)

class ReActAgent:
    def respond_to(self, llm, prompt):
        template = PromptTemplate(
            input_variables=["question"],
            template="Step 1: Understand the question\nStep 2: Analyze the question\nStep 3: Generate the answer\nFinal answer:",
        )
        chain = LLMChain(llm, AgentType.REACT)
        chain.add_template(template)
        return chain.run(prompt)

class ToolAgent:
    def respond_to(self, llm, prompt):
        tool = DynamicTool(name="search", func=self.search)
        chain = LLMChain(llm, AgentType.TOOL)
        chain.add_tool(tool)
        return chain.run(prompt)

    def search(self, prompt):
        # 这里应该是调用搜索引擎的代码，现在只是一个示例
        return f"Search result for: {prompt}"

# 使用示例
llm = ...  # 这里应该是你的语言模型实例，例如OpenAI或其他

zero_shot_agent = ZeroShotAgent()
print(zero_shot_agent.respond_to(llm, "What is the capital of France?"))

react_agent = ReActAgent()
print(react_agent.respond_to(llm, "Explain the water cycle."))

tool_agent = ToolAgent()
print(tool_agent.respond_to(llm, "What is the current weather in San Francisco?"))