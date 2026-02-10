import torch
import numpy as np
import datasets
from build_dataset import MathDataset
class Tokenizer:
    def __init__(self):
        #1.定义字符集，我们会用到这么多字符
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')','=','<EOS>','<PAD>','<UNK>']
        #2.构建字符到整数的映射和整数到字符的映射，编码和解码分别用到
        self.char2int = {c: i for i, c in enumerate(self.chars)}
        self.int2char = {i: c for i, c in enumerate(self.chars)}
        #3.记录词表大小和特殊标记的ID，后续模型训练和推理时会用到，nn.Embedding需要知道词表大小，pad_token_id和eos_token_id在处理序列时会用到
        self.vocab_size = len(self.chars)
        self.pad_token_id = self.char2int['<PAD>']
        self.eos_token_id = self.char2int['<EOS>']
    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            if text[i:i+5] == "<EOS>":
                ids.append(self.eos_token_id)
                i += 5
            elif text[i:i+5] == "<PAD>":
                ids.append(self.pad_token_id)
                i += 5
            else:
                ids.append(self.char2int.get(text[i], self.char2int['<UNK>']))
                i += 1
        return ids
    def decode(self, ids):
        """
        将输入的整数ID列表转换回字符串，遇到<UNK>的ID会被映射为<UNK>字符
        """
        return ''.join([self.int2char.get(i, '<UNK>') for i in ids])

tokenizer = Tokenizer()
#做测试，看看编码和解码是否正确
example = "3+5*2=13<EOS>"
encoded = tokenizer.encode(example)
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
#接下来是数据集处理
dataset = MathDataset(tokenizer,num_samples=1000,max_length=20)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,num_workers=0)
#测试一下数据加载是否正常
for x_batch,y_batch in dataloader:
    print("Batch shape:", x_batch.shape)
    print("Example input:", x_batch[0])
    print("Example target:", y_batch[0])
    break
