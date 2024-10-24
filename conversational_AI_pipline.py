'''
Hugging face平台的transformers.pipline聚合了预训练模型和对应的文本预处理（当task为question-answering）


目的：对话 AI 旨在与用户保持多轮、动态的对话。它根据对话上下文生成回应，并致力于实现更类似人类的对话。
工作原理：模型经过训练可以处理对话任务，能够根据对话的上下文生成回复。它通常在包含自然对话的数据集上进行微调，能够理解上下文、保持连续性，并处理多样化的输入。
典型用例：
    用于客户服务的聊天机器人或虚拟助手。
    用于娱乐或陪伴的对话系统。
    模拟自然人类对话的对话代理。
    示例模型：DialoGPT、BlenderBot、GPT-2、GPT-3（在对话数据集上微调）。

方面	        问答（Question Answering）	            对话 AI（Conversational AI）
目标	        基于给定的上下文提供具体答案。	            维持多轮对话，生成动态回复。
输入格式	    需要一个问题和上下文段落。	                接受对话历史或之前的对话上下文。
输出格式	    从上下文中提取的直接、简洁的答案。	        生成的响应，用于继续或维持对话。
训练数据	    在 QA 数据集（如 SQuAD）上训练，包含问答对。	在多轮对话数据集上训练，包含多轮对话。
用例	信息检索、FAQ 机器人、文档问答。	聊天机器人、虚拟助手、交互式对话系统。
模型行为	从固定上下文中提取和定位答案。	根据对话流生成开放式回复。
'''

from transformers import pipeline


conversational_AI = pipeline(task="conversational")#,model="microsoft/DialoGPT-medium")

# 初始化对话
chat_history = ""

while(True):
    user_input = input("You:")
    chat_history += f"{user_input}\n"
    response= conversational_AI(chat_history, pad_token_id=50256)
    print(response)
    bot_response = response[0]["generated_text"]
    print("AI:",bot_response)

    #加入对话中
    chat_history += f"{bot_response}\n"