import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import matplotlib.pyplot as plt

# 读取数据
file_path = '问题2/关键信息提取结果.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

y_data = df['Y染色体浓度']

edge = 0.04
sigma = 0.03

def main(p):

    # 计算比值
    ratio = norm.cdf(edge, loc=p, scale=sigma)  # 计算P(X < edge)的概率
    
    # 生成正态分布数据用于可视化
    x = np.linspace(p - 4*sigma, p + 4*sigma, 1000)  # 生成4倍标准差范围内的x值
    y = norm.pdf(x, loc=p, scale=sigma)  # 计算概率密度函数值
    
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制正态分布曲线
    plt.plot(x, y, 'b-', linewidth=2, label=f'N(μ={p:.2f}, σ={sigma:.2f})')
    
    # 填充edge左侧区域
    x_fill = np.linspace(p - 4*sigma, edge, 1000)
    y_fill = norm.pdf(x_fill, loc=p, scale=sigma)
    plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.5, label=f'P(X < {edge:.2f}) = {ratio:.4f}')
    
    # 添加垂直参考线
    plt.axvline(x=edge, color='red', linestyle='--', linewidth=1.5, label=f'边界值: {edge:.2f}')
    plt.axvline(x=p, color='green', linestyle='-', linewidth=1.5, label=f'均值: {p:.2f}')
    
    # 设置图表标题和标签
    plt.title(f'正态分布概率可视化 (μ={p:.2f}, σ={sigma:.2f})', fontsize=14)
    plt.xlabel('X值', fontsize=12)
    plt.ylabel('概率密度', fontsize=12)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 调整布局并保存图像
    plt.tight_layout()
    output_dir = '可视化结果/问题2'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/正态分布_{p:.2f}_{sigma:.2f}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return ratio

results = {}
for p in np.arange(0.04, 0.08, 0.01):
    
    ratio = main(p)
    results[p] = ratio

if not os.path.exists('表格结果/问题2'):
    os.makedirs('表格结果/问题2')
results = pd.DataFrame(results, index=['概率比值']).T
results.to_csv(f'表格结果/问题2/正态分布_概率比值.csv', index=True, encoding='utf-8-sig')

    

