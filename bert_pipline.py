'''
BERT 是一个以自监督方式在大量英语数据上进行预训练的 Transformer 模型。这意味着它只在原始文本上进行预训练，没有任何人工标记（这就是它可以使用大量公开数据的原因），并有一个自动流程从这些文本中生成输入和标签。更准确地说，它进行了两个预训练目标：
	掩码语言建模 (Masked language modeling，MLM)：取一个句子，模型随机掩码输入中的 15% 的单词，然后通过模型运行整个掩码句子，并预测被掩码的单词。这与通常一个接一个地看到单词的传统循环神经网络 (RNN) 或内部掩码未来标记的 GPT 等自回归模型不同。它允许模型学习句子的双向表示。
	下一句预测 (Next sentence prediction，NSP)：模型在预训练期间将两个掩码句子连接起来作为输入。有时它们对应于原文中彼此相邻的句子，有时则不是。然后，模型必须预测这两个句子是否彼此相连。
通过这种方式，模型可以学习英语的内部表征，然后可以使用该表征提取对下游任务有用的特征：例如，如果您有一个带标签的句子数据集，则可以使用 BERT 模型生成的特征作为输入来训练标准分类器。

	用途：
您可以使用原始模型进行掩码语言建模或下一句预测，但它主要用于在下游任务中进行微调。

'''


from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
print(unmasker("Hello I'm a [MASK] model."))
'''
[{'sequence': "[CLS] hello i'm a fashion model. [SEP]",
  'score': 0.1073106899857521,
  'token': 4827,
  'token_str': 'fashion'},
 {'sequence': "[CLS] hello i'm a role model. [SEP]",
  'score': 0.08774490654468536,
  'token': 2535,
  'token_str': 'role'},
 {'sequence': "[CLS] hello i'm a new model. [SEP]",
  'score': 0.05338378623127937,
  'token': 2047,
  'token_str': 'new'},
 {'sequence': "[CLS] hello i'm a super model. [SEP]",
  'score': 0.04667217284440994,
  'token': 3565,
  'token_str': 'super'},
 {'sequence': "[CLS] hello i'm a fine model. [SEP]",
  'score': 0.027095865458250046,
  'token': 2986,
  'token_str': 'fine'}]
'''