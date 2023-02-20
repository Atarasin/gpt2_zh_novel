from transformers import (
    BertTokenizer, GPT2LMHeadModel,
    TrainingArguments, DataCollatorForLanguageModeling,
    Trainer
)
from datasets import load_dataset
from utils import *


def train():
    """
    Step 1: 加载预训练模型
    Step 2: 加载数据集
    Step 3: 设置训练超参数
    """
    # Step 1: 加载预训练模型
    tokenizer = BertTokenizer.from_pretrained("./pretrained")
    gpt2_distil_zh_model = GPT2LMHeadModel.from_pretrained("./pretrained")     # 待训练


    # Step 2: 加载数据集
    data_file = {
    'train': 'train.txt',
    'test': 'test.txt'
    }
    mydataset = load_dataset("text", data_dir="data", data_files=data_file)
    mydataset = mydataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=1024),
                                batched=True, remove_columns=['text'])
    print(mydataset)

    # Step 3: 设置训练超参数
    training_args = TrainingArguments('checkpoints', 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # per_device_eval_batch_size=4,
        num_train_epochs=20,
        seed=2023,
        logging_steps=500,
        label_names=['labels'],
        save_strategy='steps',
        save_steps=2000,
        save_total_limit=5
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=gpt2_distil_zh_model,
        args=training_args,
        train_dataset=mydataset['train'],
        # eval_dataset=mydataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    result = trainer.train()
    print_summary(result)

if __name__ == "__main__":
    train()
