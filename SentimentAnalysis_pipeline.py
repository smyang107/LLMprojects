'''
Hugging face平台的transformers.pipline聚合了预训练模型和对应的文本预处理

'''
from transformers import pipeline

#使用pipline进行情绪分析
classifier = pipeline('sentiment-analysis') # 下载并缓存了流水线使用的预训练模型
result = classifier('We are very happy to introduce pipeline to the transformers repository.') #进行情绪分析
print(result)

# [{'label': 'POSITIVE', 'score': 0.9996980428695679}]