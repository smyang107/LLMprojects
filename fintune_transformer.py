'''
定义新的分类模型，对分类模型进行微调，调整了最后一层

'''
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import pipeline

# 加载IMDb数据集
dataset = load_dataset('imdb')

# 加载预训练模型和分词器
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) ## 微调模型 ##
tokenizer = AutoTokenizer.from_pretrained(model_name) # 定义分词器

def preprocess_data(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# 使用 map 函数对整个数据集进行预处理
encoded_dataset = dataset.map(preprocess_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch进行评估
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=8,   # 训练时的批量大小
    per_device_eval_batch_size=8,    # 评估时的批量大小
    num_train_epochs=3,              # 训练的epoch数
    weight_decay=0.01,               # 权重衰减
)

trainer = Trainer(
    model=model,                         # 微调的模型
    args=training_args,                  # 训练参数
    train_dataset=encoded_dataset['train'],  # 训练数据集
    eval_dataset=encoded_dataset['test']     # 测试数据集
)

# 开始训练
trainer.train()
# 评估模型
eval_results = trainer.evaluate()
print(eval_results)

# 保存模型
trainer.save_model("./fine-tuned-model")


# 加载微调后的模型
fine_tuned_model = "./fine-tuned-model"
text_classification_pipeline = pipeline("sentiment-analysis", model=fine_tuned_model)

# 测试微调后的模型
result = text_classification_pipeline("This is a fantastic movie!")
print(result)


