import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 数据加载与预处理 (Data Loading)
# ==========================================
def generate_synthetic_data():
    """
    生成模拟数据，用于在没有原始数据的情况下演示代码功能。
    """
    print("正在生成模拟数据...")
    np.random.seed(42)
    data = []
    contestants = [f'Dancer_{i}' for i in range(1, 16)]
    weeks = range(1, 11)
    
    for w in weeks:
        # 模拟每周剩余人数 (从15人逐渐减少到5人)
        num_dancers = max(5, 15 - w) 
        current_contestants = contestants[:num_dancers]
        
        # 1. 模拟粉丝投票 (帕累托分布：少数人拥有大量粉丝)
        fan_raw = np.random.pareto(a=2, size=num_dancers)
        fan_shares = fan_raw / fan_raw.sum()
        
        # 2. 模拟评委打分 (正态分布)
        judge_raw = np.random.normal(loc=0.5, scale=0.15, size=num_dancers)
        # 归一化
        judge_shares = np.clip(judge_raw, 0.1, 0.9)
        judge_shares = judge_shares / judge_shares.sum()
        
        for i, person in enumerate(current_contestants):
            data.append({
                'season': 32,
                'week': w,
                'contestant': person,
                'estimated_fan_share': fan_shares[i],
                'judge_share': judge_shares[i]
            })
            
    return pd.DataFrame(data)

def load_and_prep_data():
    """
    尝试读取本地文件，失败则使用模拟数据。
    """
    file_path_votes = 'dataset/estimated_fan_votes_final.csv'
    file_path_processed = 'dataset/processed_data_long.csv'
    
    if os.path.exists(file_path_votes) and os.path.exists(file_path_processed):
        try:
            print(f"正在读取数据文件...")
            fan_votes_df = pd.read_csv(file_path_votes)
            processed_data_df = pd.read_csv(file_path_processed)
            # 合并
            df = pd.merge(fan_votes_df, processed_data_df, 
                          on=['season', 'week', 'contestant'], how='inner')
            # 筛选第32季
            df_season = df[df['season'] == 32].copy()
            if df_season.empty:
                print("警告: 第32季数据为空，切换至模拟数据。")
                return generate_synthetic_data()
            return df_season
        except Exception as e:
            print(f"读取出错: {e}，切换至模拟数据。")
            return generate_synthetic_data()
    else:
        print("未找到数据文件，使用模拟数据运行。")
        return generate_synthetic_data()

# ==========================================
# 2. 核心模型：商业价值计算 (The Model)
# ==========================================
def calculate_commercial_value_with_surprise(eliminated_row, fair_eliminated_contestant):
    """
    计算商业价值。
    逻辑：价值 = (基础热度 * 惊讶倍率) - 风险惩罚
    """
    # 获取排名 (Rank 1 = 最高人气/最强)
    rank_actual = eliminated_row['estimated_fan_rank']
    rank_expected = fair_eliminated_contestant['estimated_fan_rank']
    
    # --- 1. 惊讶指数 (Surprise Index) ---
    # 定义：实际淘汰者比预期淘汰者强多少？
    # 如果预期淘汰 Rank 10，实际淘汰 Rank 4，惊讶度 = 6
    surprise_index = max(0, rank_expected - rank_actual)
    
    # --- 2. 基础热度 (Base Buzz) ---
    # 淘汰人气越高的人(Rank值越小)，基础热度越高
    base_buzz = 200 * np.exp(-0.25 * (rank_actual - 1))
    
    # --- 3. 惊讶倍率 (Multiplier) ---
    # 惊讶是热度的放大器
    surprise_multiplier = 1 + 0.5 * surprise_index  
    total_buzz = base_buzz * surprise_multiplier
    
    # --- 4. 风险惩罚 (Churn Risk) ---
    churn_risk = 0
    
    # 阈值 A: 绝对底线 (Top 2 不可动)
    if rank_actual <= 2:
        churn_risk += 500 # 毁灭性打击
        
    # 阈值 B: 认知失调 (黑幕嫌疑)
    if surprise_index >= 6: 
        churn_risk += 300 # "This is rigged!"
    elif surprise_index >= 4:
        churn_risk += 50  # "Robbed!" (争议区，高收益低风险 -> Sweet Spot)
        
    net_value = total_buzz - churn_risk
    
    return net_value, total_buzz, churn_risk, surprise_index

# ==========================================
# 3. 策略模拟 (Simulation)
# ==========================================
def run_full_simulation(df_season):
    print("正在运行策略模拟...")
    weeks = sorted(df_season['week'].unique())
    weekly_optimal = []   # 存储我们的动态策略结果
    comparison_data = []  # 存储用于对比的数据
    
    df_season = df_season.copy()
    # 预计算：每周内的粉丝排名 (estimated_fan_share 越大，Rank 越小)
    df_season['estimated_fan_rank'] = df_season.groupby('week')['estimated_fan_share'].rank(ascending=False)

    for week in weeks[:-1]: # 决赛周通常不适用此逻辑
        week_data = df_season[df_season['week'] == week].copy()
        
        # --- A. 基准：公平策略 (Fair/Percent Method, w=0.5) ---
        w_fair = 0.5
        week_data['fair_score'] = w_fair * week_data['judge_share'] + (1 - w_fair) * week_data['estimated_fan_share']
        fair_eliminated = week_data.sort_values('fair_score').iloc[0]
        
        # 计算公平策略的价值 (作为对比)
        val_fair, _, _, _ = calculate_commercial_value_with_surprise(fair_eliminated, fair_eliminated)
        comparison_data.append({
            'Week': week, 'Method': 'Traditional (Percent)', 
            'Value': val_fair, 'Eliminated': fair_eliminated['contestant']
        })
        
        # --- B. 对比：传统排名和法 (Rank Sum Method) ---
        week_data['rank_judge'] = week_data['judge_share'].rank(ascending=False)
        week_data['rank_fan'] = week_data['estimated_fan_share'].rank(ascending=False)
        week_data['rank_sum'] = week_data['rank_judge'] + week_data['rank_fan']
        # 排名和最大者被淘汰 (因为 Rank 1 是最好，Rank N 是最差，Sum 越大越差)
        # 注意：这里假设 Rank 1 = Best. 
        # 如果是淘汰，应该是淘汰最差的。最差的人 Rank 数字最大。所以 Rank Sum 最大者被淘汰。
        rank_sum_eliminated = week_data.sort_values('rank_sum', ascending=False).iloc[0]
        
        val_rank_sum, _, _, _ = calculate_commercial_value_with_surprise(rank_sum_eliminated, fair_eliminated)
        comparison_data.append({
            'Week': week, 'Method': 'Traditional (Rank Sum)', 
            'Value': val_rank_sum, 'Eliminated': rank_sum_eliminated['contestant']
        })
        
        # --- C. 我们的策略：动态寻找最优权重 (Dynamic Optimization) ---
        best_w = 0.5
        best_val = -float('inf')
        best_stats = {}
        
        # 网格搜索权重 w 从 0 到 1
        for w in np.linspace(0, 1, 101):
            week_data['manipulated_score'] = w * week_data['judge_share'] + (1 - w) * week_data['estimated_fan_share']
            # 分数最低者被淘汰
            curr_eliminated = week_data.sort_values('manipulated_score').iloc[0]
            
            # 计算价值
            val, buzz, risk, surprise = calculate_commercial_value_with_surprise(
                curr_eliminated, fair_eliminated
            )
            
            if val > best_val:
                best_val = val
                best_w = w
                best_stats = {
                    'buzz': buzz, 
                    'risk': risk, 
                    'surprise': surprise, 
                    'eliminated_contestant': curr_eliminated['contestant'], 
                    'expected_contestant': fair_eliminated['contestant'],
                    'rank_actual': curr_eliminated['estimated_fan_rank'],
                    'rank_expected': fair_eliminated['estimated_fan_rank']
                }
        
        # 记录最优结果
        weekly_optimal.append({
            'week': week, 
            'optimal_weight': best_w, 
            'max_value': best_val,
            **best_stats
        })
        
        # 添加到对比数据
        comparison_data.append({
            'Week': week, 'Method': 'Proposed (Dynamic)', 
            'Value': best_val, 'Eliminated': best_stats['eliminated_contestant']
        })
    
    return pd.DataFrame(weekly_optimal), pd.DataFrame(comparison_data)

# ==========================================
# 4. 绘图与可视化 (Visualization)
# ==========================================
def plot_all_results(df_dynamic, df_comparison):
    print("正在生成图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Figure 1: 商业价值与惊讶指数双轴图 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_val = '#2ca02c' # Green
    color_sur = '#9467bd' # Purple
    
    ax1.plot(df_dynamic['week'], df_dynamic['max_value'], color=color_val, marker='o', linewidth=3, label='Net Commercial Value')
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('Commercial Value ($)', color=color_val, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_val)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx() 
    ax2.bar(df_dynamic['week'], df_dynamic['surprise'], color=color_sur, alpha=0.3, width=0.5, label='Surprise Index')
    ax2.set_ylabel('Surprise Index (Rank Deviation)', color=color_sur, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_sur)
    
    plt.title('Figure 1: Maximizing Profit via Controlled Surprise', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figure1_Value_Surprise_Dual.pdf')
    
    # --- Figure 2: 动态权重变化 (Rollercoaster) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_dynamic['week'], df_dynamic['optimal_weight'], marker='s', linestyle='-', color='#d62728', linewidth=2.5)
    plt.axhline(0.5, color='gray', linestyle='--', label='Fairness Baseline (0.5)')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Week')
    plt.ylabel('Optimal Judge Weight ($w$)')
    plt.title('Figure 2: The "Rollercoaster" Manipulation Strategy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure2_Weight_Dynamics.pdf')

    # --- Figure 3: 甜蜜点散点图 (Sweet Spot) ---
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(df_dynamic['risk'], df_dynamic['buzz'], 
                          c=df_dynamic['max_value'], cmap='viridis', s=150, edgecolors='k')
    plt.colorbar(scatter, label='Net Value')
    plt.xlabel('Churn Risk (Penalty)')
    plt.ylabel('Buzz Generated')
    plt.title('Figure 3: The "Sweet Spot" Trade-off', fontsize=14, fontweight='bold')
    # 标注区域
    plt.axvline(x=100, color='red', linestyle=':', alpha=0.5)
    plt.text(300, 50, 'Danger Zone\n(High Risk)', color='red', ha='center')
    plt.text(25, 200, 'Sweet Spot\n(Max Profit)', color='green', ha='center')
    plt.tight_layout()
    plt.savefig('Figure3_Sweet_Spot.pdf')
    
    # --- Figure 4: 淘汰对比 (Fair vs Optimal) ---
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = df_dynamic['week']
    plt.bar(x - width/2, df_dynamic['rank_expected'], width, label='Fair Elimination (Expected)', color='gray', alpha=0.6)
    plt.bar(x + width/2, df_dynamic['rank_actual'], width, label='Optimal Elimination (Actual)', color='#ff7f0e')
    plt.xlabel('Week')
    plt.ylabel('Fan Rank of Eliminated Contestant\n(Lower # = More Popular)', fontsize=10)
    plt.title('Figure 4: Who Gets Cut? Fair vs. Profit-Driven', fontsize=14, fontweight='bold')
    plt.legend()
    plt.gca().invert_yaxis() # 翻转Y轴，让Rank 1在上方
    plt.tight_layout()
    plt.savefig('Figure4_Elimination_Comparison.pdf')

    # --- Figure 5: 总价值对比 (Total Value Comparison) ---
    total_val = df_comparison.groupby('Method')['Value'].sum().reset_index()
    order = ['Traditional (Percent)', 'Traditional (Rank Sum)', 'Proposed (Dynamic)']
    colors_bar = ['#A9A9A9', '#808080', '#E64B35'] # Grey, Dark Grey, Red
    
    plt.figure(figsize=(9, 6))
    sns.barplot(data=total_val, x='Method', y='Value', order=order, palette=colors_bar)
    plt.title('Figure 5: Total Season Commercial Value Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Total Commercial Value')
    # 添加数值标签
    for i, method in enumerate(order):
        val = total_val[total_val['Method'] == method]['Value'].values[0]
        plt.text(i, val + 5, f"{val:.0f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figure5_Method_Comparison.pdf')

    # --- Figure 6: 累积价值增长 (Cumulative Value) ---
    plt.figure(figsize=(10, 6))
    for method, color, style in zip(order, colors_bar, ['--', ':', '-']):
        subset = df_comparison[df_comparison['Method'] == method].copy()
        subset['Cumulative'] = subset['Value'].cumsum()
        lw = 3 if method == 'Proposed (Dynamic)' else 2
        plt.plot(subset['Week'], subset['Cumulative'], label=method, color=color, linestyle=style, linewidth=lw)
    
    plt.title('Figure 6: Cumulative Value Growth Over Season', fontsize=14, fontweight='bold')
    plt.xlabel('Week')
    plt.ylabel('Cumulative Commercial Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure6_Cumulative_Growth.pdf')
    
    print("所有图表已保存为 PDF 文件。")

# ==========================================
# 主程序 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    df_season = load_and_prep_data()
    
    if df_season is not None and not df_season.empty:
        # 2. 运行模拟
        df_results, df_compare = run_full_simulation(df_season)
        
        # 3. 保存数据
        df_results.to_csv('optimization_results.csv', index=False)
        df_compare.to_csv('method_comparison_results.csv', index=False)
        print("结果已保存至 CSV 文件。")
        
        # 4. 生成图表
        plot_all_results(df_results, df_compare)
        
        print("\n=== 最优策略摘要 (部分) ===")
        print(df_results[['week', 'optimal_weight', 'surprise', 'max_value']].head())
    else:
        print("数据加载失败。")