import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_judge_save_impact():
    print("=== 开始运行评委拯救规则（Judges' Save）影响分析 ===")
    
    # 1. 加载预处理后的数据
    try:
        df = pd.read_csv('dataset/processed_data_long.csv')
    except FileNotFoundError:
        print("Error: 未找到 'processed_data_long.csv'，请先运行数据预处理。")
        return

    # 2. 全局设置
    np.random.seed(2026) # 保证结果可复现
    simulation_stats = []
    
    # 只分析发生了单人淘汰的周次
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        # 过滤非标准淘汰周（如双淘汰或无淘汰）
        if group['is_eliminated'].sum() != 1:
            continue
            
        # 获取基础信息
        actual_elim_name = group[group['is_eliminated'] == True]['contestant'].iloc[0]
        contestants = group['contestant'].values
        raw_scores = group['total_score'].values
        
        # 计算评委分数占比 (Judge Shares)
        if np.sum(raw_scores) == 0: continue
        judge_shares = raw_scores / np.sum(raw_scores)
        
        n_contestants = len(contestants)
        elim_idx = np.where(contestants == actual_elim_name)[0][0]
        
        # --- 核心：蒙特卡洛模拟粉丝投票 ---
        # 目标：生成与历史结果一致（即导致 actual_elim 被淘汰）的粉丝投票分布
        valid_samples = 0
        save_flips = 0
        
        # 尝试生成 100 个有效的粉丝投票场景
        target_valid_samples = 100
        max_attempts = 20000 
        attempts = 0
        
        while valid_samples < target_valid_samples and attempts < max_attempts:
            attempts += 1
            
            # 随机生成粉丝份额 (Dirichlet 分布模拟不同选手的受欢迎程度)
            fan_shares = np.random.dirichlet(np.ones(n_contestants))
            
            # 计算总分 (假设 50/50 权重，直接相加份额即可)
            combined_shares = judge_shares + fan_shares
            
            # 验证：在这个虚拟场景下，被淘汰的是否是实际那个人？
            if np.argmin(combined_shares) == elim_idx:
                valid_samples += 1
                
                # --- 应用 "Judges' Save" 规则 ---
                # 1. 找到倒数两名 (Bottom Two)
                sorted_indices = np.argsort(combined_shares)
                bottom_1_idx = sorted_indices[0] # 必定是 elim_idx
                bottom_2_idx = sorted_indices[1] # 倒数第二名
                
                # 2. 评委进行选择
                # 假设评委拯救评委分更高的选手
                j_score_1 = raw_scores[bottom_1_idx]
                j_score_2 = raw_scores[bottom_2_idx]
                
                if j_score_1 > j_score_2:
                    # 评委拯救了原本的淘汰者（Bottom 1）
                    # 结果发生翻转
                    save_flips += 1
                elif j_score_1 == j_score_2:
                    # 分数相同时，通常粉丝分低的走，即维持原判
                    pass
        
        # 如果该周次很难通过随机模拟复现（说明由于极端数据导致），则跳过
        if valid_samples < 10:
            continue
            
        flip_prob = save_flips / valid_samples
        
        # 记录该选手的技术统计，用于后续分析
        # 计算 Z-Score 以衡量该选手相对于当周对手的实力
        week_mean = np.mean(raw_scores)
        week_std = np.std(raw_scores)
        z_score = (raw_scores[elim_idx] - week_mean) / week_std if week_std != 0 else 0
        
        simulation_stats.append({
            'season': season,
            'week': week,
            'contestant': actual_elim_name,
            'judge_z_score': z_score,
            'flip_prob': flip_prob
        })

    # 3. 结果汇总与保存
    sim_df = pd.DataFrame(simulation_stats)
    sim_df.to_csv('judge_save_impact.csv', index=False)
    print(f"分析完成：共处理 {len(sim_df)} 个有效周次，结果已保存至 judge_save_impact.csv")

    # 4. 绘制分析图表
    plt.figure(figsize=(14, 6))
    
    # 图1：安全网效应 (The Safety Net Effect)
    # 展示翻转概率与评委打分(Z-Score)的关系
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=sim_df, x='judge_z_score', y='flip_prob', 
                    hue='flip_prob', palette='coolwarm', s=80, edgecolor='k')
    
    # 添加辅助线与标注
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.5, 0.8, 'High Skill, Low Popularity\n(Protected Zone)', 
             fontsize=10, color='darkgreen', ha='left')
    plt.text(-1.5, 0.2, 'Low Skill\n(Elimination Likely)', 
             fontsize=10, color='darkred', ha='left')
    
    plt.title('Impact of Judges\' Save: The "Safety Net" for Talent', fontsize=14)
    plt.xlabel('Contestant Judge Score (Z-Score)', fontsize=12)
    plt.ylabel('Probability of Being Saved', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend([],[], frameon=False) # 隐藏图例

    # 图2：影响分布直方图
    plt.subplot(1, 2, 2)
    sns.histplot(sim_df['flip_prob'], bins=15, kde=True, color='teal')
    plt.title('Distribution of Outcome Reversals', fontsize=14)
    plt.xlabel('Probability of Result Flip', fontsize=12)
    plt.ylabel('Frequency (Weeks)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    avg_flip = sim_df['flip_prob'].mean()
    plt.axvline(avg_flip, color='red', linestyle='--', label=f'Avg Impact: {avg_flip:.1%}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('judge_save_analysis.pdf')
    print("图表已保存至 judge_save_analysis.pdf")

if __name__ == "__main__":
    analyze_judge_save_impact()