import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 顶级期刊风格配置 (Nature Style) ---
def set_nature_style():
    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif' 
    # 如果系统没有Times New Roman，这段会自动回退到默认衬线体，不会报错
    try:
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    except:
        pass
    
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    sns.set_palette(colors)
    
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11

def save_fig(fig, filename):
    plt.tight_layout()
    # 增加 bbox_inches='tight' 防止标签被截断
    fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved {filename}")

# --- 2. 数据加载与容错处理 ---
df = pd.read_csv('dataset/processed_data_long.csv')

# 【关键修复】: 检查是 'placement' 还是 'final_placement'
rank_col = 'final_placement' if 'final_placement' in df.columns else 'placement'
print(f"正在使用排名列: {rank_col}")

# 重新计算关键特征（确保数据完整性）
# Z-Score
grouped = df.groupby(['season', 'week'])['total_score']
df['week_mean'] = grouped.transform('mean')
df['week_std'] = grouped.transform('std').fillna(1)
df['performance_z_score'] = (df['total_score'] - df['week_mean']) / df['week_std']

# Momentum
df = df.sort_values(['season', 'contestant', 'week'])
df['prev_score_ratio'] = df.groupby(['season', 'contestant'])['score_ratio'].shift(1)
df['momentum'] = df['score_ratio'] - df['prev_score_ratio']

# 评委分歧度 (如果预处理没算，这里补算)
if 'judge_controversy' not in df.columns:
    judge_cols = [c for c in df.columns if 'judge' in c and 'score' not in c and 'count' not in c and 'controversy' not in c]
    # 通常是 judge1, judge2...
    if not judge_cols:
        # 尝试找 judge1_score 格式
        judge_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
    
    if judge_cols:
        df['judge_controversy'] = df[judge_cols].std(axis=1)
    else:
        df['judge_controversy'] = 0 # 无法计算

# --- 3. 分组逻辑 (修复了KeyError) ---
def get_group(row):
    # 使用动态获取的 rank_col
    rank = row[rank_col]
    if rank == 1:
        return 'Winner'
    elif rank <= 3:
        return 'Finalist'
    else:
        return 'Eliminated'

df['group'] = df.apply(get_group, axis=1)

# 应用风格
set_nature_style()

# --- 图表 1: 冠军之路 ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=df, 
    x='week', 
    y='performance_z_score', 
    hue='group', 
    style='group',
    markers=True,
    dashes=False,
    palette=['#E64B35', '#4DBBD5', 'gray'],
    alpha=0.8,
    ax=ax1
)
ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
ax1.set_xlabel('Competition Week')
ax1.set_ylabel('Performance Z-Score (Standardized)')
ax1.set_title('Evolution of Competitive Dominance', fontweight='bold')
ax1.legend(title='Status', loc='lower right', frameon=False)
ax1.set_xlim(1, 11)
save_fig(fig1, 'Fig1_Champion_Trajectory.pdf')

# --- 图表 2: 争议热力图 ---
if 'judge_controversy' in df.columns:
    pivot_controversy = df.pivot_table(index='season', columns='week', values='judge_controversy', aggfunc='mean')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_controversy, cmap='Reds', cbar_kws={'label': 'Avg. Judge Std. Dev'}, ax=ax2)
    ax2.set_title('Heatmap of Judicial Controversy', fontweight='bold')
    ax2.invert_yaxis()
    save_fig(fig2, 'Fig2_Controversy_Heatmap.pdf')

# --- 图表 3: 分数收敛趋势 ---
fig3, ax3 = plt.subplots(figsize=(8, 5))
subset_df = df[df['week'] <= 10]
sns.violinplot(
    data=subset_df, x='week', y='score_ratio', hue='week',
    palette='viridis', legend=False, inner='quartile', linewidth=0.8, ax=ax3
)
ax3.set_title('Convergence of Performance Scores', fontweight='bold')
ax3.grid(axis='y', alpha=0.2)
save_fig(fig3, 'Fig3_Score_Distribution.pdf')

# --- 图表 4: 危险区域散点图 ---
fig4, ax4 = plt.subplots(figsize=(7, 7))
sns.scatterplot(
    data=df, x='performance_z_score', y='momentum',
    hue='is_eliminated', style='is_eliminated',
    palette={False: '#4DBBD5', True: '#E64B35'},
    alpha=0.7, s=40, ax=ax4
)
ax4.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax4.set_title('The Danger Zone Analysis', fontweight='bold')
# 手动添加注释，避免因坐标轴范围问题报错
try:
    ax4.text(-1.5, -0.4, 'High Risk Zone', fontsize=9, color='#E64B35', ha='center')
    ax4.text(1.5, 0.4, 'Safe Zone', fontsize=9, color='#4DBBD5', ha='center')
except:
    pass
save_fig(fig4, 'Fig4_Danger_Zone_Scatter.pdf')

print("所有图表生成完毕！")