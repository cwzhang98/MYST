# Fairseq

## 任务/模型/损失函数等在框架中的构建流程

### 预备知识

- 遵循***先注册再构建***
- register_model()等注册装饰器，在文件被导入时就会执行

### 构建流程

1. train.py作为程序的入口，在其中会导入fairseq.tasks，此时所有的task都完成注册
2. `task = tasks.setup_task(cfg.task)`,先在已注册的task中寻找目标task，再返回指定任务的实例化对象
3. 调用`task.build_model()`,其中会调用`super().build_model()`，即`FairseqTask.build_model()`，在此方法内：
    - 导入fairseq.models，完成所有模型的注册
    - 调用`models.build_model()`,在已注册的模型中寻找参数中指定的model(验证合法性)
    - 找到参数指定的模型后，调用`model.build_model()`，也就是我们在模型中自定义的`build_model()`方法，返回模型实例
4. criterion/optim等，注册及构建过程类似

# 预处理

## 配置

|             |   Tokenizer   |        Dictionary size         | Vocabulary type |
| ----------- | :-----------: | :----------------------------: | :-------------: |
| Source Side |    g2p_en     |    according to **g2p_en**     |     Phoneme     |
| Target Side | SentencePiece | 10k (trained on MuSTC and WMT) |     Subword     |

- Speech Input: Raw speech
- Fairseq在训练中怎么tokenized(在dataset类中实现)
    - bpe_tokenizer.encode()方法实际上调用sentencepiece.EncodeAsPieces()方法，先把文本转换为对应的sub-word列表
    - 根据词典文件，使用Dictionary.encode_line()方法生成tensor

### 步骤

1. `g2p_encoder.py`处理训练集中的源端文本
