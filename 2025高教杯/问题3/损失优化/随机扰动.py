from pandas.core.resample import Day
import torch
from 主模型 import model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 用于计算差距

DATA_PATH_SMOTE = "问题3/损失优化/SMOTE过采样数据.csv"
DATA_PATH = "问题3/损失优化/关键信息提取结果（完整信息）.csv"

def add_perturbation(data, std=1):
    """为数据添加高斯扰动"""
    # 生成与数据形状相同的高斯噪声
    noise = torch.normal(mean=0.0, std=std, size=data.shape)
    return data + noise

def mlp_train():
    data = pd.read_csv(DATA_PATH_SMOTE)

    y = data['Y染色体浓度']
    x = data.drop('Y染色体浓度',axis=1)

    xs = torch.tensor(x.values,dtype=torch.float32)
    ys = torch.tensor(y.values,dtype=torch.float32)

    Model = model()
    optimizer = torch.optim.Adam(Model.parameters(),lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(10):
        for x,y in zip(xs,ys):
            y_pred = Model(x,'mlp')
            y = y.unsqueeze(0)
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(Model.state_dict(),f"mlp_model_{epoch}.pth")

def target_train(perturb_std=0.01):
    # 加载数据
    data = pd.read_csv(DATA_PATH)
    x = data.drop('Y染色体浓度', axis=1)
    xs = torch.tensor(x.values, dtype=torch.float32)
    
    # 创建扰动数据
    xs_perturbed = add_perturbation(xs, std=perturb_std)
    
    # 初始化模型
    Model = model()
    
    # 优化器设置
    optimizer = torch.optim.Adam(Model.parameters(),lr=0.001)
    print(f"优化器参数数量: {len(optimizer.param_groups[0]['params'])}")

    # 训练循环
    for epoch in range(10):
        for x in xs:
            loss = Model(x, 'loss')  # 使用损失模式输出
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Target Epoch {epoch}, Loss: {loss.item()}")

    # 获取原始数据的结果
    activate1_original = Model.activate1
    BMI_original = activate1_original.d.detach().numpy()
    Day_original = activate1_original.t.detach().numpy()

    # 使用扰动数据获取结果（仅前向传播，不更新参数）
    Model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        for x in xs_perturbed:
            _ = Model(x, 'loss')  # 前向传播获取扰动后的激活值
        activate1_perturbed = Model.activate1
        BMI_perturbed = activate1_perturbed.d.detach().numpy()
        Day_perturbed = activate1_perturbed.t.detach().numpy()

    # 计算扰动前后的差距
    bmi_mse = mean_squared_error(BMI_original, BMI_perturbed)
    bmi_mae = mean_absolute_error(BMI_original, BMI_perturbed)
    day_mse = mean_squared_error(Day_original, Day_perturbed)
    day_mae = mean_absolute_error(Day_original, Day_perturbed)

    # 打印结果和差距
    print("\n原始结果:")
    print(f"划分BMI: {BMI_original}")
    print(f"检测孕周: {Day_original}")
    
    print("\n扰动后结果:")
    print(f"划分BMI: {BMI_perturbed}")
    print(f"检测孕周: {Day_perturbed}")
    
    print("\n扰动前后差距:")
    print(f"BMI - MSE: {bmi_mse:.6f}, MAE: {bmi_mae:.6f}")
    print(f"检测孕周 - MSE: {day_mse:.6f}, MAE: {day_mae:.6f}")

    # 保存原始结果
    result = {"检测孕周": Day_original, "孕妇BMI": np.append(BMI_original, 1)}
    result = pd.DataFrame(result, columns=["检测孕周", "孕妇BMI"])
    result.to_csv("表格结果/问题3/损失优化模型结果.csv", index=False)
    
    # 保存扰动后结果
    result_perturbed = {"检测孕周": Day_perturbed, "孕妇BMI": np.append(BMI_perturbed, 1)}
    result_perturbed = pd.DataFrame(result_perturbed, columns=["检测孕周", "孕妇BMI"])
    result_perturbed.to_csv("表格结果/问题3/损失优化模型扰动后结果.csv", index=False)


if __name__ == '__main__':
    mlp_train()
    # 可以通过调整perturb_std参数控制扰动强度，默认为0.01
    target_train(perturb_std=0.01)