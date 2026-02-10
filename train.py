import torch
from build_tokenizer import tokenizer, dataloader
from model import SimpleTransformer
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    'embed_size': 128,  #嵌入维度，表示每个字符的向量表示的维度
    'num_heads': 8,  #多头自注意力机制中的头数，表示模型在不同的表示子空间中关注输入序列的不同部分的能力
    'num_layers': 4,  #Transformer模型中Block模块的数量，表示模型的深度
    'block_size': 20,  #Transformer模型中输入序列的最大长度，表示模型能够处理的最长序列长度
    'vocab_size': tokenizer.vocab_size,  #词表大小，表示模型能够处理的不同字符的数量
}
model = SimpleTransformer(**config)
model.to(device)
print("模型参数量:", sum(p.numel() for p in model.parameters()))
#2.损失函数设计

#交叉熵损失函数（CrossEntropyLoss）是一种常用的分类问题的损失函数，适用于多类分类任务。
# 在训练过程中，模型会输出每个位置上每个词的预测概率分布，而交叉熵损失函数会计算模型预测的概率分布与真实标签之间的差异
# 并将其作为损失值进行优化。
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#3.优化器设计
#Adam是transformer标配
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        #将输入数据和目标数据移动到设备上（GPU或CPU），以便进行计算
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        #接下来是核心4步
        #1.梯度清零：在每次迭代开始时，使用optimizer.zero_grad()将模型参数的梯度清零，以防止梯度累积。
        optimizer.zero_grad()
        #2.前向传播：将输入数据x_batch传递给模型，得到模型的输出logits。
        # logits是一个形状为(batch_size, seq_length, vocab_size)的张量，表示每个位置上每个词的预测概率分布。
        logits = model(x_batch)
        #3.计算损失：使用定义好的损失函数criterion计算模型的输出logits与目标数据y_batch之间的损失。
        loss = criterion(logits.view(-1, tokenizer.vocab_size), y_batch.view(-1))
        #4.反向传播和优化：调用loss.backward()计算损失的梯度，然后使用optimizer.step()更新模型参数以最小化损失。
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
def solve_equation(model, tokenizer, equation,max_new_tokens=50):
    model.eval()
    with torch.no_grad():
        #将输入的数学表达式编码为整数ID列表，并转换为张量
        input_ids = tokenizer.encode(equation)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  #添加batch维度并移动到设备上
        with torch.no_grad():#在这个上下文管理器内，所有的计算都不会被记录用于梯度计算，这样可以节省内存和计算资源，因为在推理阶段我们不需要进行反向传播。
            for _ in range(max_new_tokens):
                #使用模型进行预测，得到每个位置上每个词的预测概率分布
                logits = model(input_tensor)
                predicted_ids = torch.argmax(logits[:,-1,:], dim=-1).unsqueeze(0)  #获取最后一个位置的预测结果,得到一个形状为(1,)的张量，表示模型预测的下一个字符的ID
                input_tensor = torch.cat([input_tensor, predicted_ids], dim=-1)  #将预测结果拼接到输入序列末尾，继续生成后续字符
                if predicted_ids.item() == tokenizer.eos_token_id:  #如果预测结果是<EOS>，表示生成结束，跳出循环
                    break
        predicted_equation = tokenizer.decode(input_tensor.tolist()[0])  #将最终的输入序列解码为字符串，得到模型生成的完整数学表达式
        return predicted_equation
#测试一下模型的推理能力
test_equation = ["7+8=","11+45=","91+78=","11+45=","13+78="]  #测试输入，可以修改为其他表达式
for eq in test_equation:
    predicted = solve_equation(model, tokenizer, eq)
    print(f"输入: {eq}  预测: {predicted}")
