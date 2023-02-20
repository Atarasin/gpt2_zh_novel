# README

- transformers：4.24.0
- datasets：2.6.1

本次项目的主要目的为：使用金庸先生的14部小说对中文gpt-2模型进行fine-tuning，获得一个金庸小说生成器。

训练过程可以参考：[Training a causal language model from scratch - Hugging Face Course](https://huggingface.co/course/chapter7/6?fw=pt)

## 1.预训练模型

本次采用的预训练模型为蒸馏版的中文gpt-2模型，模型权重来源于[uer/gpt2-distil-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-distil-chinese-cluecorpussmall)。

```python
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
text_generator = TextGenerationPipeline(model, tokenizer)   
text_generator("这是很久之前的事情了", max_length=100, do_sample=True)
```

为了便于使用，可以先将模型相关文件下载到本地。

```shell
git clone https://huggingface.co/uer/gpt2-distil-chinese-cluecorpussmall ./pretrained
```

之后直接加载本地模型即可使用。

```python
tokenizer = BertTokenizer.from_pretrained("pretrained")
model = GPT2LMHeadModel.from_pretrained("pretrained")
```

## 2.自定义数据集

数据集来源为金庸先生的14部三联版小说的txt文件，放在raw文件夹内。raw文件夹也可以放入自己想训练的txt文件。

```powershell
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
------          2023/1/3     10:25        1568129 书剑恩仇录（三联版）.txt
------          2023/1/3     10:25        1118071 侠客行（三联版）.txt
------          2023/1/3     10:25        2934061 倚天屠龙记（三联版）.txt
------          2023/1/3     10:25        3719484 天龙八部（三联版）.txt
------          2023/1/3     10:25        2817163 射雕英雄传（三联版）.txt
------          2023/1/3     10:25         207056 白马啸西风（三联版）.txt
------          2023/1/3     10:25        1499681 碧血剑（三联版）.txt
------          2023/1/3     10:25        2912152 神雕侠侣（三联版）.txt
------          2023/1/3     10:25        3008104 笑傲江湖（三联版）.txt
------          2023/1/3     10:25         707559 连城诀（三联版）.txt
------          2023/1/3     10:25         400606 雪山飞狐（三联版）.txt
------          2023/1/3     10:25        1349116 飞狐外传（三联版）.txt
------          2023/1/3     10:25         106309 鸳鸯刀（三联版）.txt
------          2023/1/3     10:25        3766619 鹿鼎记（三联版）.txt
```

当然，这些txt文件不能直接作为数据集进行训练，需要做一些预处理。

在做预处理之前，我们首先需要明确`Text Generation`这个NPL任务的数据集格式。

与其它的深度学习任务不同，文本生成的任务是根据已知的一段文本来预测下一个token。因此，**文本生成的数据集不需要额外的标注，它的label就是数据本身，只不过将数据向后移动一个token而已**。

transformers提供了专门用于语言模型的collator，除了可以stacking与padding之外，还能生成专门的label。

> We can use the `DataCollatorForLanguageModeling` collator, which is designed specifically for language modeling (as the name subtly suggests). Besides stacking and padding batches, it also takes care of creating the language model labels — **in causal language modeling the inputs serve as labels too (just shifted by one element)**, and this data collator creates them on the fly during training so we don’t need to duplicate the `input_ids`.

因此，该自定义数据集只需要将以上所有的txt文本数据放在一起，然后根据比例分割为train、validate、test数据集即可。预处理函数如下所示：

```python
from pathlib import Path
from typing import Union
from math import floor

def data_split(data_dir: Union[str, Path], output_dir: Union[str, Path], train_size: float = 0.7):
    """
    读取指定目录的所有txt文件, 按照一定比例划分为train, test
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise Exception('data_dir is not a dictory')

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise Exception('output_dir is not a dictory')

    lines = []

    for file in data_dir.glob('*.txt'):
        with file.open('r', encoding='utf-8') as f:
            lines.extend(f.readlines())

    total_lines_num = len(lines)
    total_train_lines_num = floor(total_lines_num * train_size)

    # create 'train.txt' and 'test.txt'
    train_file = Path(output_dir, 'train.txt')
    test_file = Path(output_dir, 'test.txt')

    with train_file.open('w', encoding='utf-8') as f:
        f.writelines(lines[:total_train_lines_num])
    
    with test_file.open('w', encoding='utf-8') as f:
        f.writelines(lines[total_train_lines_num:])
```

获得训练数据后，可以通过`load_dataset()`将数据加载到dataset中。

```python
from datasets import load_dataset

data_file = { 'train': 'train.txt', 'test': 'test.txt' }
mydataset = load_dataset("text", data_dir="data", data_files=data_file)
```

> DatasetDict({
>     train: Dataset({
>         features: ['text'],
>         num_rows: 71208
>     })
>     test: Dataset({
>         features: ['text'],
>         num_rows: 12567
>     })
> })

以上还只是纯文本数据，无法直接用于训练，必须使用分词器将文本数据分割为一个个单词，并根据词典转换为相应的token id值。文本全部转换为id后，文本内容就可以删除了，已经没有意义了。

```python
mydataset = mydataset.map(
    lambda examples: tokenizer(examples['text'], truncation=True, max_length=1024), 		batched=True, remove_columns=['text'])
```

> DatasetDict({
>     train: Dataset({
>         features: ['input_ids', 'token_type_ids', 'attention_mask'],
>         num_rows: 71208
>     })
>     test: Dataset({
>         features: ['input_ids', 'token_type_ids', 'attention_mask'],
>         num_rows: 12567
>     })
> })

## 3.设置训练超参数

```python
training_args = TrainingArguments('checkpoints', 
   per_device_train_batch_size=1,
   gradient_accumulation_steps=4,
   num_train_epochs=3,			# 总共训练3轮
   seed=2023,
   logging_steps=500,			# 每隔500步输出一次loss值
   label_names=['labels'],
   save_strategy='steps',		# 每隔2000步保存一次模型, 保存模型数最多为5个, 超过则覆盖
   save_steps=2000,
   save_total_limit=5
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=gpt2_distil_zh_model,
    args=training_args,
    train_dataset=mydataset['train'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()		# 开始训练
```

