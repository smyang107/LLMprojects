'''
torch.hub.load 是 PyTorch 提供的一个方便的函数，用于加载预训练模型或者第三方库中的模型，而不需要用户手动下载模型文件。

（1）从torch.hub自动下载：

bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')

（2）从torch.hub手动下载：

from fairseq.models.bart import BARTModel
model_path = 'path/to/model.pt'  # 替换为你下载的模型文件路径
model = BARTModel.from_pretrained(model_path)


'''


### 使用 BART 执行句子对分类任务：

import torch
from fairseq.models.bart import BARTModel


# Download BART already finetuned for MNLI
#bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli') #自动从hub下载预训练模型

model_path = 'path/to/model.pt'  # 替换为你下载的模型文件路径
bart = BARTModel.from_pretrained(model_path)
bart.eval()  # disable dropout for evaluation

# Encode a pair of sentences and make a prediction
tokens = bart.encode('BART is a seq2seq model.', 'BART is not sequence to sequence.')
bart.predict('mnli', tokens).argmax()  # 0: contradiction

# Encode another pair of sentences
tokens = bart.encode('BART is denoising autoencoder.', 'BART is version of autoencoder.')
bart.predict('mnli', tokens).argmax()  # 2: entailment


