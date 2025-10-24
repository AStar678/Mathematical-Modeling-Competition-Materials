import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取男胎检测数据
data_path_female = '数据预处理/女胎检测数据.csv'
data_path_male = '数据预处理/男胎检测数据.csv'

# 读取数据并添加性别标签
df_male = pd.read_csv(data_path_male)
df_male['性别'] = '男'
df_female = pd.read_csv(data_path_female)
df_female['性别'] = '女'

df = pd.concat([df_male, df_female], ignore_index=True)

# 定义需要可视化的指标列表
indicators = [
    '年龄', '身高', '体重', '孕妇BMI', '原始读段数', 
    '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 'GC含量', 
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 
    'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度', 'X染色体浓度', 
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 
    '被过滤掉读段数的比例'
]

# 筛选存在的指标
existing_indicators = [ind for ind in indicators if ind in df.columns]
missing_indicators = [ind for ind in indicators if ind not in df.columns]
for ind in missing_indicators:
    print(f"警告：'{ind}' 列不存在于数据中，已跳过")

if not existing_indicators:
    print("没有找到有效的指标列，程序退出")
    exit()

# 创建输出目录
output_dir = '可视化结果/数据预处理'
os.makedirs(output_dir, exist_ok=True)

# 生成颜色方案 - 使用更现代和谐的viridis调色板，优化透明度
n_indicators = len(existing_indicators)
colors = [(r, g, b, 0.75) for r, g, b in sns.color_palette("viridis", n_indicators)]

# 生成性别颜色方案 - 更协调的蓝粉配色，增强对比度
GENDER_COLORS = {'男': '#2563eb', '女': '#db2777'}  # 深蓝色为男，玫瑰红为女

# ---------------------- 箱型图组合（Subplot版） ----------------------
# 计算子图网格布局 (设置为4列，自动计算行数)
cols = 4
rows = math.ceil(n_indicators / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()  # 将2D数组转换为1D以便循环

# 绘制每个指标的箱型图
for i, (indicator, color) in enumerate(zip(existing_indicators, colors)):
    sns.boxplot(data=df, y=indicator, ax=axes[i], color=color)
    axes[i].set_title(indicator, fontsize=30)
    axes[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=30)
    axes[i].tick_params(axis='y', labelsize=30)
    axes[i].set_ylabel('')  # 删除Y轴标签文字
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=2.0)
plt.savefig(f'{output_dir}/箱型图_subplot组合.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 小提琴图组合（Subplot版） ----------------------
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

# 绘制每个指标的小提琴图
for i, (indicator, color) in enumerate(zip(existing_indicators, colors)):
    sns.violinplot(data=df, y=indicator, ax=axes[i], color=color, inner='quartile')
    axes[i].set_title(indicator, fontsize=30)
    axes[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axes[i].tick_params(axis='y', labelsize=30)
    axes[i].set_ylabel('')  # 删除Y轴标签文字
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=2.0)
plt.suptitle('所有指标小提琴图组合 (Subplot)', fontsize=30, y=1.02)
plt.savefig(f'{output_dir}/小提琴图_subplot组合.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 合并可视化（箱型图+小提琴图） ----------------------
# 计算子图网格布局 (设置为4列，自动计算行数)
cols = 4
rows = math.ceil(n_indicators / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
axes = axes.flatten()  # 将2D数组转换为1D以便循环

# 为每个指标绘制合并图表
for i, indicator in enumerate(existing_indicators):
    ax = axes[i]
    # 绘制小提琴图
    sns.violinplot(
        data=df, x='性别', y=indicator, ax=ax,
        palette=GENDER_COLORS, inner='quartile', alpha=0.6
    )
    # 绘制箱型图（叠加在小提琴图上）
    sns.boxplot(
        data=df, x='性别', y=indicator, ax=ax,
        palette=GENDER_COLORS, width=0.2, linewidth=2
    )
    ax.set_title(indicator, fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_ylabel('')  # 保留无纵轴标签设置
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=2.0)
plt.savefig(f'{output_dir}/性别对比_箱型图小提琴图合并.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Subplot组合图表已保存至 {output_dir} 目录")
