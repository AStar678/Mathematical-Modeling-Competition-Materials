from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class activate1(nn.Module):
    def __init__(self, class_num=6, min_val=0, max_val=1, k=1000):
        super().__init__()
        self.class_num = class_num
        self.min = min_val
        self.max = max_val
        self.k = k
        
        # t参数在[-0.5, 1]范围且保持排序
        self.t = nn.Parameter(torch.linspace(min_val, max_val, class_num))
        
        # d参数修改为在[0, 1]范围生成
        d_values = torch.linspace(0, 1, class_num + 1)[1:-1]  # 修改范围为0到1
        self.d = nn.Parameter(d_values)

    def parameter_loss(self):
        # 对t参数施加范围惩罚
        t_lower = torch.relu(-self.t)  # t < 0 时的惩罚
        t_upper = torch.relu(self.t - 1)  # t > 1 时的惩罚
        t_penalty = torch.sum(t_lower + t_upper) * 1e6  # 放大惩罚系数

        # 对d参数施加范围惩罚
        d_lower = torch.relu(-self.d)  # d < 0 时的惩罚
        d_upper = torch.relu(self.d - 1)  # d > 1 时的惩罚
        d_penalty = torch.sum(d_lower + d_upper) * 1e6

        # 添加t参数均匀分布惩罚
        sorted_t, _ = torch.sort(self.t)
        ideal_t = torch.linspace(0, 1, self.class_num, device=self.t.device)
        t_uniform_penalty = F.mse_loss(sorted_t, ideal_t) * 0  # 较小的惩罚系数

        # 添加d参数均匀分布惩罚
        sorted_d, _ = torch.sort(self.d)
        ideal_d = torch.linspace(0, 1, self.class_num - 1, device=self.d.device)
        d_uniform_penalty = F.mse_loss(sorted_d, ideal_d) * 1e-2

        return t_penalty + d_penalty + t_uniform_penalty + d_uniform_penalty

    def forward(self, x):
        # 确保d在使用前按从小到大排序
        sorted_d, _ = torch.sort(self.d)
        
        result = 0
        for i in range(self.class_num - 1):
            # 使用排序后的参数进行计算
            result += (self.t[i + 1] - self.t[i]) * F.sigmoid(self.k * (x - sorted_d[i]))
        
        return result + self.t[0]

import torch
import torch.nn as nn

class activate2(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = -0.055
        self.end = 0.055
        self.dangerous = 0.777
        self.smoothness = 5.0  
        
        # 调整各阶段增长的控制参数，确保递增趋势
        self.mid_slope = 0.5    
        self.high_slope = 1  
        self.accel_factor = 0.2 

    def forward(self, x):
        # 计算各区间的平滑权重（通过sigmoid实现连续过渡）
        # 1. x < start 的权重
        w1 = torch.sigmoid(-self.smoothness * (x - self.start))
        
        # 2. start ≤ x < end 的权重
        w2 = torch.sigmoid(self.smoothness * (x - self.start)) * \
             torch.sigmoid(-self.smoothness * (x - self.end))
        
        # 3. end ≤ x < dangerous 的权重
        w3 = torch.sigmoid(self.smoothness * (x - self.end)) * \
             torch.sigmoid(-self.smoothness * (x - self.dangerous))
        
        # 4. x ≥ dangerous 的权重
        w4 = torch.sigmoid(self.smoothness * (x - self.dangerous))
        
        # 各区间的基础函数（确保自身严格递增）
        f1 = (self.start - x)  # x < 10：递减
        # 将0.1转换为Tensor，修复sigmoid输入类型错误
        f2 = self.mid_slope * (x - self.start) + f1 * torch.sigmoid(torch.tensor(-self.smoothness * 0.1))  # 10~12：递增
        f3 = self.mid_slope * (self.end - self.start) + self.high_slope * (x - self.end)  # 12~25：递增，确保与前一区间衔接
        f4 = f3 + (x - self.dangerous) + torch.exp(self.accel_factor * (x - self.dangerous)) - 1  # 25以上：递增，确保与前一区间衔接
        
        # 加权组合，确保整体严格单调
        result = f1 * w1 + f2 * w2 + f3 * w3 + f4 * w4
        
        return result

class activate3(nn.Module):
    def __init__(self):
        super().__init__()
        self.limit = 0.133
        self.smoothness = 500.0  # 对于小数值需要更大的平滑系数

    def forward(self, x):
        # 计算两个区间的平滑权重
        w1 = torch.sigmoid(-self.smoothness * (x - self.limit))  # x < limit
        w2 = torch.sigmoid(self.smoothness * (x - self.limit))   # x ≥ limit
        
        # 计算各个区间的输出并加权求和
        part1 = (-x + self.limit) * w1
        part2 = torch.zeros_like(x) * w2  # 恒为0
        
        return part1 + part2 + 0.1

    

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图像并设置大小
    plt.figure(figsize=(12, 6))
    
    # 生成数据
    x = torch.linspace(0, 1, 1000)  # 使用更多点使曲线更平滑
    model = activate1()
    y = model(x)
    
    # 绘制阶梯函数曲线
    plt.plot(x.numpy(), y.detach().numpy(), 'b-', linewidth=3, label='激活函数输出')
    
    # 添加分界点垂直线
    d_values = model.d.detach().numpy()
    for i, d in enumerate(d_values):
        plt.axvline(x=d, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        plt.text(d, plt.ylim()[1]*0.95, f'分界点 {i+1}: {d:.2f}', 
                 color='red', ha='center', rotation=45, fontweight='bold')
    
    # 添加水平参考线
    t_values = model.t.detach().numpy()
    for i, t in enumerate(t_values):
        plt.axhline(y=t, color='g', linestyle=':', alpha=0.5, linewidth=1)
        plt.text(plt.xlim()[0]*1.02, t, f'水平段 {i+1}: {t:.2f}', 
                 color='green', va='center', fontweight='bold')
    
    # 添加标题和标签
    plt.title('阶梯状激活函数可视化', fontsize=18, pad=20)
    plt.xlabel('输入值', fontsize=14, labelpad=10)
    plt.ylabel('输出值', fontsize=14, labelpad=10)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 调整布局并保存图像
    plt.tight_layout()
    output_dir = '可视化结果/问题3'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/激活函数1可视化.png', dpi=300, bbox_inches='tight')

    #尝试反向传播
    x = torch.tensor([25.0])
    y = model(x)
    loss = nn.MSELoss()(y, torch.tensor([30.0]))
    loss.backward()
    print(model.t.grad)
    print(model.d.grad)
    
    # 可视化激活函数2
    plt.figure(figsize=(12, 6))
    
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 生成数据（覆盖激活函数2的所有分段区域）
    x2 = torch.linspace(-1, 1, 1000)
    model2 = activate2()
    
    # 计算输出（使用循环处理每个x值，因activate2使用Python条件判断）
    y2 = torch.tensor([model2(x) for x in x2])  # 移除.item()转换
    
    # 绘制激活函数曲线
    plt.plot(x2.numpy(), y2.numpy(), 'purple', linewidth=3, label='activate2输出')
    
    # 标记关键分段点
    key_points = [
        (model2.start, 'start=10'),
        (model2.end, 'end=12'),
        (model2.dangerous, 'dangerous=25')
    ]
    
    for x_val, label in key_points:
        # 垂直线标记
        plt.axvline(x=x_val, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
        # 文本标注
        plt.text(x_val, plt.ylim()[1]*0.95, label, 
                 color='darkorange', ha='center', rotation=45, fontweight='bold')
        # 标记点
        plt.scatter([x_val], [model2(torch.tensor(x_val)).item()], 
                   color='red', s=100, zorder=5)
    
    # 添加标题和标签
    plt.title('激活函数2 (activate2) 可视化', fontsize=18, pad=20)
    plt.xlabel('输入值', fontsize=14, labelpad=10)
    plt.ylabel('输出值', fontsize=14, labelpad=10)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 保存图像
    output_dir = '可视化结果/问题3'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/激活函数2可视化.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"激活函数2可视化结果已保存至: {output_dir}/激活函数2可视化.png")

    # 可视化激活函数3
    plt.figure(figsize=(12, 6))
    
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 生成数据（覆盖激活函数3的定义域）
    x3 = torch.linspace(0, 1, 1000)  # 从0到0.08，覆盖分界点两侧
    model3 = activate3()
    
    # 计算输出
    y3 = torch.tensor([model3(x) for x in x3])  # 移除.item()转换
    
    # 绘制函数曲线
    plt.plot(x3.numpy(), y3.numpy(), 'b-', linewidth=3, label='activate3输出')
    
    # 标记关键分界点
    limit_value = model3.limit
    plt.axvline(x=limit_value, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.text(limit_value, plt.ylim()[1]*0.95, f'分界点: {limit_value}', 
             color='red', ha='center', rotation=45, fontweight='bold')
    plt.scatter([limit_value], [model3(torch.tensor(limit_value)).item()], 
               color='red', s=100, zorder=5)
    
    # 添加标题和标签
    plt.title('激活函数3 (activate3) 可视化', fontsize=18, pad=20)
    plt.xlabel('输入值', fontsize=14, labelpad=10)
    plt.ylabel('输出值', fontsize=14, labelpad=10)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 保存图像
    output_dir = '可视化结果/问题3'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/激活函数3可视化.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"激活函数3可视化结果已保存至: {save_path}")
