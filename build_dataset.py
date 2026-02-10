import torch
import random
from torch.utils.data import Dataset, DataLoader
class MathDataset(Dataset):
    def __init__(self,tokenizer,num_samples=1000,max_length=20):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.data = self.generate_data()
    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            #随机生成一个数学表达式，格式为 "a+b=c<EOS>"，其中a和b是0-9的数字，c是a和b的和
            a = torch.randint(0, 99, (1,)).item()
            b = torch.randint(0, 99, (1,)).item()
            expression = f"{a}+{b}={a+b}<EOS>"
            data.append(expression)
        return data
    def __len__(self):#所需的方法，返回数据集的大小，训练时会用到
        return len(self.data)
    def __getitem__(self, idx):#所需的方法，返回指定索引的数据样本，训练时会用到
        equation = self.data[idx]
        tokens = self.tokenizer.encode(equation)
        #对编码后的整数ID列表进行填充，使其长度达到max_length，填充使用pad_token_id
        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        x = torch.tensor(tokens[:-1],dtype=torch.long)  #输入序列，不包括最后一个<EOS>，长度为max_length-1
        y = torch.tensor(tokens[1:],dtype=torch.long)   #目标序列，不包括第一个字符，长度为max_length-1
        return x, y