'''
功能介绍：LangChain 提供的核心组件（例如，提示模板、链、代理和记忆）来设计和管理聊天逻辑。它利用 OpenAI 的 GPT-3 或 GPT-4 语言模型来构建一个简单的聊天机器人

定义聊天机器人的组件

1.提示模板：定义聊天机器人应如何回应用户。
2.语言模型：使用 OpenAI 的 GPT-3 或 GPT-4 作为语言模型（没有手机号码注册）。
3.链（Chain）：将提示模板和语言模型组合在一起。
4.记忆（Memory）：允许聊天机器人记住先前的对话上下文。

'''
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key" ###？？？？？ 登录openAI 获得 API 的 key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain.chains import ConversationChain
from langchain.llms import OpenAI, HuggingFaceHub,Cohere,Anthropic,LlamaCpp

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 初始化语言模型（这里使用 OpenAI 的 GPT 模型）
#llm = OpenAI(model="text-davinci-003")  # 或者使用 "gpt-4" 取决于你可用的模型版本
# 使用 Hugging Face 上的模型
llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7}) #repo_id 模型的ID

# 设置提示模板
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="以下是与助手的对话。\n历史记录：\n{history}\n用户：{input}\n助手："
)

# 使用对话记忆
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=True
)

# 开始对话
print("开始对话！输入 'exit' 来退出对话。")

while True:
    user_input = input("你：")
    if user_input.lower() == "exit":
        break
    # 运行对话链
    response = conversation.run(input=user_input)

    print(f"助手：{response}")


'''

开始对话！输入 'exit' 来退出对话。
你：你是谁


> Entering new ConversationChain chain...
Prompt after formatting:
以下是与助手的对话。
历史记录：

用户：你是谁
助手：
F:\科研论文\论文收集\研究生-NLP\Langchain平台\LangChain_code\conversation.py:51: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use invoke instead.
  response = conversation.run(input=user_input)

> Finished chain.
助手：以下是与助手的对话。
历史记录：

用户：你是谁
助手：
你：你没回答？


> Entering new ConversationChain chain...
Prompt after formatting:
以下是与助手的对话。
历史记录：
Human: 你是谁
AI: 以下是与助手的对话。
历史记录：

用户：你是谁
助手：
用户：你没回答？
助手：

> Finished chain.
助手：以下是与助手的对话。
历史记录：
Human: 你是谁
AI: 以下是与助手的对话。
历史记录：

用户：你是谁
助手：
用户：你没回答？
助手：

用户：你就发明

Ning: 我吉院边

Ning: 我吉院边

AI: 我吉院边

Ning: 我吉院边

全力: 以下是与助手的
你：我吉院边是什么意思


> Entering new ConversationChain chain...
Prompt after formatting:
以下是与助手的对话。
历史记录：
Human: 你是谁
AI: 以下是与助手的对话。
历史记录：

用户：你是谁
助手：
Human: 你没回答？
AI: 以下是与助手的对话。
历史记录：
Human: 你是谁
AI: 以下是与助手的对话。
历史记录：

用户：你是谁
助手：
用户：你没回答？
助手：

用户：你就发明

Ning: 我吉院边

Ning: 我吉院边

AI: 我吉院边

Ning: 我吉院边

全力: 以下是与助手的
用户：我吉院边是什么意思
助手：

> Finished chain.
助手：以下是与助手的对话。
历史记录：
Human: 你是谁
AI: 以下是与助手的对话。
历史记录：

用户：你是谁
助手：
Human: 你没回答？
AI: 以下是与助手的对话。
历史记录：
Human: 你是谁
AI: 以下是与助手的对话。
历史记录：

用户：你是谁
助手：
用户：你没回答？
助手：

用户：你就发明

Ning: 我吉院边

Ning: 我吉院边

AI: 我吉院边

Ning: 我吉院边

全力: 以下是与助手的
用户：我吉院边是什么意思
助手：我吉院边的数

AI: 我吉院边是什么意思

全力: 全力

Ning: 我吉院边

Ning: 我吉院边

AI: 我吉院边的数

你：
Process finished with exit code -1

'''