import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# --- 全局设置 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 如果环境支持中文，请取消注释

DATA_PATH = 'estimated_fan_votes_final.csv'

def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        print("Error: File not found.")
        return pd.DataFrame()

df = load_data()

# ==============================================================================
# PART 1: 四大争议案例的独立微观画像 (Dual-Axis Survival Analysis)
# ==============================================================================

def plot_controversial_cases(df):
    cases = [
        {'name': 'Jerry Rice', 'season': 2, 'color': 'green', 'title': 'The "Rank Shield" (S2)'},
        {'name': 'Billy Ray Cyrus', 'season': 4, 'color': 'orange', 'title': 'The "Country Music" Force (S4)'},
        {'name': 'Bristol Palin', 'season': 11, 'color': 'purple', 'title': 'The "Political" Survivor (S11)'},
        {'name': 'Bobby Bones', 'season': 27, 'color': 'red', 'title': 'The "Cardinal" Overlord (S27)'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, case in enumerate(cases):
        ax1 = axes[i]
        name = case['name']
        season = case['season']
        
        # 提取数据
        data = df[(df['season'] == season) & (df['contestant'] == name)].sort_values('week')
        if data.empty: continue
        
        # 准备数据
        weeks = data['week']
        j_rank = data['judge_rank']
        f_share = data['estimated_fan_share']
        
        # --- 左轴：评委排名 (倒序，1在上面) ---
        color_j = 'tab:blue'
        l1 = ax1.plot(weeks, j_rank, color=color_j, marker='o', linestyle='-', linewidth=2, label='Judge Rank (Technical)')
        ax1.set_xlabel('Week', fontsize=12)
        ax1.set_ylabel('Judge Rank (Lower is Better)', color=color_j, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_j)
        ax1.set_ylim(data['judge_rank'].max() + 1, 0.5) # 倒序设置
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # --- 右轴：粉丝份额 (正序，越高越好) ---
        ax2 = ax1.twinx()
        color_f = case['color']
        l2 = ax2.plot(weeks, f_share, color=color_f, marker='s', linestyle='--', linewidth=2, label='Fan Share (Popularity)')
        ax2.set_ylabel('Estimated Fan Vote Share', color=color_f, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_f)
        ax2.set_ylim(0, max(f_share.max()*1.2, 0.4)) # 留出空间
        
        # --- 标注“生存缺口 (Survival Gap)” ---
        # 标记出评委给低分但粉丝给高分的区域
        # 简单逻辑：如果 Judge Rank > 3 (差) 且 Fan Share > 20% (高)
        gap_weeks = data[(data['judge_rank'] >= data['judge_rank'].mean()) & (data['estimated_fan_share'] > 0.15)]
        if not gap_weeks.empty:
            ax1.fill_between(weeks, 0, 1, where=(data['judge_rank'] >= 3), 
                             color='gray', alpha=0.1, transform=ax1.get_xaxis_transform(), label='Controversy Zone')
            
        # 标题与图例
        ax1.set_title(f"{name} (Season {season}): {case['title']}", fontsize=14, fontweight='bold')
        
        # 合并图例
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig('q2_case_studies_dual_axis.png')
    print("Generated: q2_case_studies_dual_axis.png")

# ==============================================================================
# PART 2: 创新分析 - 民粹主义象限轨迹 (The Populism Quadrant Trajectory)
# ==============================================================================
# 理论：我们计算每周的 Z-Score (标准分)。
# X轴：评委 Z-Score (越右越强)
# Y轴：粉丝 Z-Score (越上越红)
# 正常选手应该在 y=x 线附近。争议选手会向“左上角”移动。

def plot_populism_trajectory(df):
    target_names = ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones']
    
    # 1. 计算全量 Z-Score
    df_z = df.copy()
    
    # 按赛季-周分组计算Z-Score
    # Judge Score越大越好，Fan Share越大越好
    # 注意：judge_share 已经是归一化的，但在不同周之间方差不同，用Z-Score更公平
    df_z['j_zscore'] = df_z.groupby(['season', 'week'])['judge_share'].transform(lambda x: zscore(x, nan_policy='omit'))
    df_z['f_zscore'] = df_z.groupby(['season', 'week'])['estimated_fan_share'].transform(lambda x: zscore(x, nan_policy='omit'))
    
    plt.figure(figsize=(12, 10))
    
    # 绘制背景（所有其他选手的点，淡灰色）
    plt.scatter(df_z['j_zscore'], df_z['f_zscore'], color='lightgray', alpha=0.3, s=10, label='Regular Contestants')
    
    # 绘制坐标轴线
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    # 绘制四个象限标签
    plt.text(1.5, 1.5, 'Elite Icons\n(High Skill, High Fans)', ha='center', color='green', fontweight='bold')
    plt.text(-1.5, -1.5, 'Elimination Zone\n(Low Skill, Low Fans)', ha='center', color='gray', fontweight='bold')
    plt.text(1.5, -1.5, 'Technical Experts\n(High Skill, Low Fans)', ha='center', color='blue', fontweight='bold')
    plt.text(-1.5, 1.5, 'Populist Heroes\n(Low Skill, High Fans)', ha='center', color='red', fontweight='bold')
    
    # 绘制争议选手的轨迹
    colors = {'Jerry Rice': 'green', 'Billy Ray Cyrus': 'orange', 'Bristol Palin': 'purple', 'Bobby Bones': 'red'}
    markers = ['o', 's', '^', 'D']
    
    for i, name in enumerate(target_names):
        track = df_z[df_z['contestant'] == name].sort_values('week')
        if track.empty: continue
        
        c = colors[name]
        # 绘制轨迹线
        plt.plot(track['j_zscore'], track['f_zscore'], color=c, alpha=0.8, linewidth=2, label=f"{name} Trajectory")
        # 绘制起点和终点
        plt.scatter(track['j_zscore'].iloc[0], track['f_zscore'].iloc[0], color=c, marker=markers[i], s=100, label='_nolegend_')
        plt.text(track['j_zscore'].iloc[0], track['f_zscore'].iloc[0], 'Start', fontsize=8, color=c)
        
        plt.scatter(track['j_zscore'].iloc[-1], track['f_zscore'].iloc[-1], color=c, marker='X', s=150, edgecolors='black', label='_nolegend_')
        plt.text(track['j_zscore'].iloc[-1], track['f_zscore'].iloc[-1], 'End', fontsize=9, color=c, fontweight='bold')
        
        # 添加箭头表示时间流向
        for j in range(len(track)-1):
            if j % 2 == 0: # 减少箭头密度
                plt.arrow(track['j_zscore'].iloc[j], track['f_zscore'].iloc[j], 
                          track['j_zscore'].iloc[j+1]-track['j_zscore'].iloc[j], 
                          track['f_zscore'].iloc[j+1]-track['f_zscore'].iloc[j],
                          shape='full', lw=0, length_includes_head=True, head_width=0.1, color=c)

    plt.title('The "Populism Quadrant": Trajectories of Controversy', fontsize=16)
    plt.xlabel('Technical Merit (Judge Score Z-Score)', fontsize=12)
    plt.ylabel('Popularity (Fan Vote Z-Score)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig('q2_innovation_quadrant.png')
    print("Generated: q2_innovation_quadrant.png")

# --- 执行 ---
if not df.empty:
    plot_controversial_cases(df)
    plot_populism_trajectory(df)