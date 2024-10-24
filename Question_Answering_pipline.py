'''
Hugging face平台的transformers.pipline聚合了预训练模型和对应的文本预处理（当task为question-answering）

目的：问答用于根据给定的上下文或文档找到特定问题的答案。
工作原理：模型接收一个问题和上下文（文本段落）作为输入，并尝试在提供的上下文中找到确切的答案。模型经过训练能够理解文本并从中提取相关的答案。
典型用例：
    在文档或知识库中查找信息。
    构建 FAQ 系统，其中答案直接从内容中提取。
    从结构化或半结构化文本中提供具体的细节或事实。

示例模型：BERT、DistilBERT、ALBERT、RoBERTa（训练于 SQuAD 或类似数据集）。
示例：
输入：
    问题："Hugging Face 以什么闻名？"
    上下文："Hugging Face 以其开源的自然语言处理工具和库闻名。"


'''

from transformers import pipeline

question_answerer = pipeline(task='question-answering',device="cuda:0",model="distilbert-base-cased-distilled-squad")

## 测试,在context中寻找问题的答案 ##
result = question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline has been included in the huggingface/transformers repository'
})
print(result['answer'])

# {'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}

