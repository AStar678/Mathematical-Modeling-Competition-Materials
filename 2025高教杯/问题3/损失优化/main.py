from pandas.core.resample import Day
import torch
from 主模型 import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 新增导入
import os

DATA_PATH_SMOTE = "问题3/损失优化/关键信息提取结果（完整信息）.csv"
DATA_PATH = "问题3/损失优化/关键信息提取结果（完整信息）.csv"


def mlp_train(Model, alpha):  # 修改函数参数，添加alpha
    data = pd.read_csv(DATA_PATH_SMOTE)

    y = data['Y染色体浓度']
    x = data.drop('Y染色体浓度',axis=1)

    xs = torch.tensor(x.values,dtype=torch.float32)
    ys = torch.tensor(y.values,dtype=torch.float32)

    optimizer = torch.optim.Adam(Model.parameters(),lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    # 新增：初始化损失历史记录
    loss_history = []

    for epoch in range(10):
        for x,y in zip(xs,ys):
            y_pred = Model(x,'mlp')
            y = y.unsqueeze(0)
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        # 新增：记录损失值
        loss_history.append(loss.item())
    
    # 新增：保存损失历史到CSV
    loss_df = pd.DataFrame({
        'Epoch': range(len(loss_history)),
        'Loss': loss_history
    })
    # 创建保存目录（如果不存在）
    os.makedirs("表格结果/问题3/mlp_loss", exist_ok=True)
    # 保存CSV文件，包含alpha参数标识
    loss_df.to_csv(f"表格结果/问题3/mlp_loss/mlp_loss_history_alpha_{alpha}.csv", index=False)
    
    torch.save(Model.state_dict(),f"mlp_model_{epoch}.pth")
    

def target_train(Model):
    # 加载数据（复用mlp_train的数据加载逻辑）
    data = pd.read_csv(DATA_PATH)
    x = data.drop('Y染色体浓度', axis=1)
    xs = torch.tensor(x.values, dtype=torch.float32)
    
    # # 冻结MLP层所有参数
    for param in Model.mlp.parameters():
        param.requires_grad = False
    
    # 只优化非冻结参数（如激活函数层参数）
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Model.parameters()),lr=0.01)
    print(f"优化器参数数量: {len(optimizer.param_groups[0]['params'])}")

    # 初始化损失历史记录
    loss_history = []  # 新增

    # 训练循环（保持与mlp_train相同的训练逻辑）
    for epoch in range(10):
        for x in xs:
            loss = Model(x, 'loss')  # 使用损失模式输出
            #print(f"loss{loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Target Epoch {epoch}, Loss: {loss.item()}")
        loss_history.append(loss.item())  # 新增

    return loss_history  # 新增返回损失历史

def main(alpha):
    Model = model(alpha)
    mlp_train(Model, alpha)  # 修改：传递alpha参数
    loss_history = target_train(Model)  # 接收损失历史

    # 绘制损失下降曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o', linestyle='-', color='b')
    plt.title(f'Loss Curve (alpha={alpha})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # 保存到可视化结果目录
    plt.savefig(f"可视化结果/问题3/损失下降/损失下降曲线_alpha_{alpha}.png")
    plt.close()  # 关闭图像避免内存占用

    activate1 = Model.activate1

    BMI = activate1.d.detach().numpy()
    Day = activate1.t.detach().numpy()

    print(f"划分{BMI}")
    print(f"天数{Day}")

    result = {"检测孕周":Day,"孕妇BMI":np.append(BMI,1)}
    result = pd.DataFrame(result,columns=["检测孕周","孕妇BMI"])
    
    result.to_csv(f"问题3/损失优化/损失优化结果/损失优化模型结果_{alpha}.csv",index=False)


if __name__ == '__main__':
    os.makedirs("可视化结果/问题3/损失下降",exist_ok=True)
    os.makedirs("问题3/损失优化/损失优化结果",exist_ok=True)
    for alpha in [1]:
        main(alpha)
    