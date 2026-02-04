import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_voting_methods_enhanced():
    print("=== 开始执行赛制深度对比分析 (Rank vs Percent) ===")
    
    # 1. 数据加载
    try:
        df = pd.read_csv('dataset/processed_data_long.csv')
    except FileNotFoundError:
        print("Error: 数据文件未找到。")
        return

    np.random.seed(2026)
    comparison_results = []
    
    # 按周次分组分析
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        # 只分析单人淘汰的标准周次
        if group['is_eliminated'].sum() != 1:
            continue
            
        # 准备基础数据
        contestants = group['contestant'].values
        judge_scores = group['total_score'].values
        actual_elim = group[group['is_eliminated'] == True]['contestant'].iloc[0]
        elim_idx = np.where(contestants == actual_elim)[0][0]
        n_contestants = len(contestants)
        
        # 计算评委数据的两种形态
        # Rank: 1 is Best. 
        judge_ranks = pd.Series(judge_scores).rank(ascending=False, method='min').values 
        # Percent: Higher is Best.
        total_j = np.sum(judge_scores)
        judge_percents = judge_scores / total_j if total_j > 0 else np.zeros(n_contestants)
        
        # 确定当季实际使用的规则 (用于验证模拟有效性)
        current_method_is_rank = (season <= 2)
        
        # --- 蒙特卡洛模拟：反推粉丝投票 ---
        valid_samples = []
        attempts = 0
        
        while len(valid_samples) < 50 and attempts < 2000:
            attempts += 1
            # 随机生成粉丝得票分布 (Dirichlet分布)
            fan_shares = np.random.dirichlet(np.ones(n_contestants))
            fan_ranks = pd.Series(fan_shares).rank(ascending=False, method='min').values
            
            # 验证：该分布是否会导致历史真实的淘汰结果？
            valid = False
            if current_method_is_rank:
                # 排名制：Rank Sum 最大者淘汰 (假设Rank 1为最优)
                # 注：实际规则细节可能有变种，此处统一逻辑为"Sum of Ranks, Max is Worst"
                # (对应 1+1=2 Best, N+N=2N Worst)
                if np.argmax(judge_ranks + fan_ranks) == elim_idx:
                    valid = True
            else:
                # 百分比制：Percent Sum 最小者淘汰
                if np.argmin(judge_percents + fan_shares) == elim_idx:
                    valid = True
            
            if valid:
                valid_samples.append(fan_shares)
        
        if not valid_samples:
            continue
            
        # --- 核心对比：在相同的粉丝投票下，两种赛制的表现 ---
        rank_elim_scores = []   # 排名制淘汰者的评委排名
        percent_elim_scores = [] # 百分比制淘汰者的评委排名
        flips = 0
        
        for fan_shares in valid_samples:
            fan_ranks = pd.Series(fan_shares).rank(ascending=False, method='min').values
            
            # 1. 模拟排名制结果
            rank_sums = judge_ranks + fan_ranks
            # 处理并列情况：取并列者中索引最小的（简化处理，大样本下误差可忽略）
            r_elim = np.argmax(rank_sums)
            
            # 2. 模拟百分比制结果
            p_sums = judge_percents + fan_shares
            p_elim = np.argmin(p_sums)
            
            # 记录差异
            if r_elim != p_elim:
                flips += 1
            
            # 记录被淘汰者的评委排名 (衡量"误杀"程度)
            # 排名数值越大 = 表现越差。
            # 如果某方法淘汰了Rank 5，另一方法淘汰了Rank 3，前者更合理（淘汰了更差的）。
            rank_elim_scores.append(judge_ranks[r_elim])
            percent_elim_scores.append(judge_ranks[p_elim])
            
        # 汇总该周数据
        comparison_results.append({
            'season': season,
            'week': week,
            'flip_rate': flips / len(valid_samples),
            'avg_judge_rank_elim_by_RankMethod': np.mean(rank_elim_scores),
            'avg_judge_rank_elim_by_PercentMethod': np.mean(percent_elim_scores),
            # 差异指标：正值表示排名制淘汰的人"更应该被淘汰"（评委排名更靠后）
            'merit_protection_gap': np.mean(rank_elim_scores) - np.mean(percent_elim_scores)
        })

    # 4. 结果可视化
    res_df = pd.DataFrame(comparison_results)
    res_df.to_csv('method_comparison.csv', index=False)
    
    plt.figure(figsize=(14, 6))
    
    # 图1：不一致率分布
    plt.subplot(1, 2, 1)
    sns.histplot(res_df['flip_rate'], bins=10, color='#4c72b0', kde=True)
    plt.title('Disagreement Rate between Ranking & Percentage Methods')
    plt.xlabel('Probability of Different Outcome')
    plt.ylabel('Frequency (Weeks)')
    plt.axvline(res_df['flip_rate'].mean(), color='r', linestyle='--', 
                label=f'Avg Disagreement: {res_df["flip_rate"].mean():.1%}')
    plt.legend()
    
    # 图2：谁在保护技术流？
    # 比较被淘汰者的平均评委排名
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=res_df, x='avg_judge_rank_elim_by_RankMethod', 
                    y='avg_judge_rank_elim_by_PercentMethod', 
                    hue='flip_rate', size='merit_protection_gap', 
                    palette='coolwarm', alpha=0.8)
    
    # 添加对角线
    max_val = max(res_df['avg_judge_rank_elim_by_RankMethod'].max(), 
                  res_df['avg_judge_rank_elim_by_PercentMethod'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Identity Line')
    
    plt.text(max_val*0.6, max_val*0.2, "Below Line:\nRank Method eliminates\nWORSE dancers than Percent Method", 
             color='darkred', fontsize=9)
    
    plt.title('Meritocracy Check: Judge Rank of Eliminated Contestant')
    plt.xlabel('Avg Judge Rank of Eliminated (Rank Method)\n(Higher Value = Worse Dancer)')
    plt.ylabel('Avg Judge Rank of Eliminated (Percent Method)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('method_comparison_visuals.pdf')
    print("图表已保存至 method_comparison_visuals.pdf")

if __name__ == "__main__":
    compare_voting_methods_enhanced()