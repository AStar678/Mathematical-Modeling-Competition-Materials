import pandas as pd
import os 

# 加载数据
df = pd.read_csv('数据预处理/男胎特征工程后数据.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 定义特征列表
features = [
    '孕妇BMI', '原始读段数', '在参考基因组上比对的比例', '重复读段的比例',
    '唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
    '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '被过滤掉读段数的比例'
]

# 创建一个包含多个子图的画布
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义不同行的色系
row_colors = ['Blues', 'Oranges', 'Greens', 'Purples']

# 初始化归一化器
scaler = MinMaxScaler()

# 全局设置文字大小
plt.rcParams.update({'font.size': 14})

# 绘制散点图和趋势线
for i, feature in enumerate(features):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    cmap = plt.get_cmap(row_colors[row])
    color = cmap(0.6)

    # 设置点的颜色为对应色系并添加透明度，同时调小点的大小
    sns.scatterplot(data=df, x=feature, y='Y染色体浓度', ax=ax, color=color, alpha=0.4, s=5)
    # 趋势线颜色使用对应色系较深的颜色
    line_color = cmap(0.8)
    sns.regplot(
        data=df,
        x=feature,
        y='Y染色体浓度',
        ax=ax,
        scatter=False,
        color=line_color,
        line_kws={"lw": 2}
    )

    # 对数据进行归一化
    X = scaler.fit_transform(df[[feature]].values)
    y = scaler.fit_transform(df[['Y染色体浓度']].values)

    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # 计算评估指标
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # 在子图上显示评估指标
    text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Y染色体浓度', fontsize=12)

# 隐藏多余的子图
for i in range(len(features), 16):
    row = i // 4
    col = i % 4
    axes[row, col].axis('off')
if not os.path.exists('可视化结果/问题1'):
    os.makedirs('可视化结果/问题1')
plt.savefig('可视化结果/问题1/散点图.png', dpi=300, bbox_inches='tight')