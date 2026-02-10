import torch
import torch.nn as nn
import torch.nn.functional as F
class Block(nn.Module):
    """
Transformer模型中的基本模块，包含多头自注意力机制和前馈神经网络，以及层归一化和残差连接
这些组件是Transformer模型的核心部分，负责处理输入序列的表示，并捕捉序列中不同位置之间的依赖关系。
通过使用多头自注意力机制，模型能够在不同的表示子空间中关注输入序列的不同部分，从而更好地理解序列的结构和语义。
前馈神经网络则对每个位置的表示进行非线性变换，增强模型的表达能力。
层归一化有助于稳定训练过程，而残差连接则缓解了深层网络中的梯度消失问题，使得训练更深的网络成为可能。
    """
    def __init__(self, embed_size, num_heads):
        #1.super:调用父类的构造函数，确保Block类正确继承nn.Module的功能
        super(Block, self).__init__()
        #2.定义多头自注意力机制和前馈神经网络，以及层归一化，后续Transformer模型会用到这些组件
        #多头自注意力机制允许模型在不同的表示子空间中关注输入序列的不同部分，前馈神经网络用于对每个位置的表示进行非线性变换，层归一化有助于稳定训练过程
        self.attn = nn.MultiheadAttention(embed_size, num_heads,batch_first=True)
        self.ffn = nn.Sequential(#Sequential是一个容器，按顺序包含多个层，前馈神经网络通常由两层线性变换和一个激活函数组成
            nn.Linear(embed_size, embed_size * 4),  #前馈神经网络的隐藏层维度通常是输入维度的4倍
            nn.ReLU(),
            #激活函数引入非线性，使模型能够学习更复杂的表示。激活函数:ReLU（Rectified Linear Unit）是深度学习中常用的激活函数，定义为f(x) = max(0, x)，即当输入x大于0时输出x，否则输出0。ReLU的优点包括计算效率高、能够缓解梯度消失问题，并且在实践中表现良好。
            nn.Linear(embed_size * 4, embed_size)#将前馈神经网络的输出维度恢复到输入维度，以便与残差连接相加
        )
        self.ln1 = nn.LayerNorm(embed_size)
        #层归一化（Layer Normalization）是一种正则化技术，应用于神经网络的每一层，特别是在Transformer模型中。它通过对每个样本的特征进行归一化来稳定和加速训练过程。
        #层归一化的计算方式是：对于输入的特征向量，计算其均值和标准差，然后使用这些统计量对特征进行归一化，使其具有零均值和单位方差。
        #这样可以帮助模型更快地收敛，并且在训练过程中减少内部协变量偏移（Internal Covariate Shift）的影响，从而提高模型的性能和稳定性。
        self.ln2 = nn.LayerNorm(embed_size)
        #残差连接（Residual Connection）是一种神经网络结构设计技巧，常用于深度学习模型中，特别是在Transformer模型中。它通过在网络层之间添加直接的跳跃连接，使得输入可以绕过一个或多个层直接传递到后续层。
        #这种设计有助于缓解深层网络中的梯度消失问题，并且使得训练更深的网络成为可能。
        #在Transformer模型中，残差连接通常与层归一化（Layer Normalization）结合使用。
        #在每个子层（如多头自注意力机制和前馈神经网络）之后，都会添加一个残差连接，将输入直接加到子层的输出上，然后再进行层归一化。
        # 这种结构使得模型能够更有效地学习和优化，从而提高性能和稳定性。
    def forward(self, x,mask=None):
        """
        前向传播函数，定义了Block模块的计算过程
        输入x是一个形状为(batch_size, seq_length, embed_size)的张量
        表示输入序列的嵌入表示
        输出也是一个形状为(batch_size, seq_length, embed_size)的张量
        表示经过Block模块处理后的序列表示
        """
        norm_x = self.ln1(x)  #对输入进行层归一化，得到norm_x
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)  #计算多头自注意力机制的输出，输入是norm_x，输出是attn_output
        x = x + attn_output  #添加残差连接，将输入x与注意力输出相加
        norm_x = self.ln2(x)  #对相加后的结果进行层归一化，得到norm_x
        ffn_out = self.ffn(norm_x)  #计算前馈神经网络的输出，输入是norm_x，输出是ffn_output
        x = x + ffn_out  #添加残差连接，将输入x与前馈神经网络的输出相加
        return x  #返回经过Block模块处理后的序列表示
class SimpleTransformer(nn.Module):
    def __init__(self,vocab_size,embed_size,num_heads,num_layers,block_size):
        super().__init__()
        self.block_size = block_size
        #1.定义输入嵌入层和位置嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        #2.位置编码，为序列添加位置信息，使模型能够区分序列中不同位置的元素
        self.position_embedding = nn.Embedding(block_size, embed_size)
        #3.堆叠block，num_layers表示Block模块的数量
        self.blocks = nn.ModuleList([Block(embed_size, num_heads) for _ in range(num_layers)])
        #4.输出层，将Transformer的输出映射到词表大小的维度，以便进行分类或生成任务
        self.head = nn.Linear(embed_size, vocab_size)
    def forward(self, idx):
        """
        前向传播函数，定义了SimpleTransformer模型的计算过程
        输入idx是一个形状为(batch_size, seq_length)的张量，表示输入序列的整数ID
        输出是一个形状为(batch_size, seq_length, vocab_size)的张量，表示每个位置上每个词的预测概率分布
        """
        batch_size,seq_len = idx.shape
        #1.计算输入的token嵌入和位置嵌入
        positions = torch.arange(0, seq_len, device=idx.device).expand(batch_size, seq_len)
        #生成位置索引，形状为(batch_size, seq_len)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        #向量叠加：将token嵌入和位置嵌入相加，得到输入序列的嵌入表示

        #2.生成注意力掩码，防止模型在预测时看到未来的信息
        mask = torch.triu(torch.ones((seq_len, seq_len), device=idx.device),diagonal=1).bool()
        for block in self.blocks:
            x = block(x,mask)  #依次通过每个Block模块，输入是x和mask，输出是经过Block处理后的x
        logits = self.head(x)  #通过输出层，将Transformer的输出映射到词表大小的维度，得到每个位置上每个词的预测概率分布
        return logits  #返回预测概率分布
    
