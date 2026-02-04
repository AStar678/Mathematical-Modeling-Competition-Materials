import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与预处理
# ==========================================
def load_and_prep_data():
    # 读取数据
    fan_votes_df = pd.read_csv('dataset/estimated_fan_votes_final.csv')
    processed_data_df = pd.read_csv('dataset/processed_data_long.csv')

    # 合并数据集
    df = pd.merge(fan_votes_df, processed_data_df, 
                  on=['season', 'week', 'contestant'], 
                  how='inner')
    
    # 筛选第32季作为案例研究 (因为数据较全且有代表性)
    target_season = 32
    df_season = df[df['season'] == target_season].copy()
    
    return df_season

# ==========================================
# 2. 核心模型：商业价值计算函数
# ==========================================
def calculate_commercial_value_tuned(eliminated_row, week_data):
    """
    输入：被淘汰的选手行数据，当周所有选手数据
    输出：净商业价值 (Net Value), 热度 (Buzz), 风险 (Risk)
    理论：
    - 淘汰人气一般的选手 -> 无热度，无风险 (Boring)
    - 淘汰人气较高的选手 -> 高热度，低风险 (Sweet Spot)
    - 淘汰人气最高的选手 -> 极高热度，但在灾难性风险 (Disaster)
    """
    # 1. 重新计算当周的人气排名 (1 = 最高人气)
    week_data = week_data.copy()
    week_data['fan_rank_current'] = week_data['estimated_fan_share'].rank(ascending=False)
    
    # 获取被淘汰者的人气排名
    person_stats = week_data[week_data['contestant'] == eliminated_row['contestant']].iloc[0]
    R_F = person_stats['fan_rank_current']
    
    # --- 模型公式 ---
    
    # A. 热度 (Buzz): 指数衰减模型
    # 排名越靠前，淘汰时的震惊程度越高
    # Rank 1 buzz ~ 200, Rank 5 buzz ~ 60
    buzz = 200 * np.exp(-0.3 * (R_F - 1)) 
    
    # B. 风险 (Risk): 阶跃惩罚函数
    # 只有触碰了观众底线(Top 2)才会导致严重脱粉
    churn_risk = 0
    if R_F <= 1.5:   # 淘汰了第1名 (绝对顶流)
        churn_risk = 300 # 惩罚 > 热度 (得不偿失)
    elif R_F <= 2.5: # 淘汰了第2名
        churn_risk = 150 # 高风险
    elif R_F <= 3.5: # 淘汰了第3名
        churn_risk = 20  # 风险极低，但有热度 -> Sweet Spot!
    else:
        churn_risk = 0   # 无人关心
        
    net_value = buzz - churn_risk
    return net_value, buzz, churn_risk

# ==========================================
# 3. 模拟策略：寻找最优权重
# ==========================================
def simulate_strategies(df_season):
    # A. 静态策略模拟 (整个赛季固定一个 w)
    static_results = []
    weights = np.linspace(0, 1, 101)
    
    for w in weights:
        season_total = 0
        weeks = sorted(df_season['week'].unique())
        for week in weeks[:-1]: # 决赛周除外
            week_data = df_season[df_season['week'] == week].copy()
            # 计算混合得分
            week_data['combined_score'] = w * week_data['judge_share'] + (1 - w) * week_data['estimated_fan_share']
            # 模拟淘汰分数最低者
            eliminated = week_data.sort_values('combined_score').iloc[0]
            val, _, _ = calculate_commercial_value_tuned(eliminated, week_data)
            season_total += val
        static_results.append((w, season_total))
    
    df_static = pd.DataFrame(static_results, columns=['weight', 'total_value'])
    
    # B. 动态策略模拟 (每周寻找最优 w)
    weeks = sorted(df_season['week'].unique())
    weekly_optimal = []
    
    for week in weeks[:-1]:
        best_w = 0
        best_val = -float('inf')
        
        week_data = df_season[df_season['week'] == week].copy()
        
        # 遍历权重寻找当周最优解
        for w in np.linspace(0, 1, 101):
            week_data['combined_score'] = w * week_data['judge_share'] + (1 - w) * week_data['estimated_fan_share']
            eliminated = week_data.sort_values('combined_score').iloc[0]
            val, _, _ = calculate_commercial_value_tuned(eliminated, week_data)
            
            if val > best_val:
                best_val = val
                best_w = w
                
        weekly_optimal.append({'week': week, 'optimal_weight': best_w, 'max_value': best_val})
    
    df_dynamic = pd.DataFrame(weekly_optimal)
    
    return df_static, df_dynamic

# ==========================================
# 4. 绘图代码
# ==========================================
def plot_results(df_static, df_dynamic, df_season):
    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 图表 1: 静态 vs 动态策略对比 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：静态权重优化曲线
    ax1.plot(df_static['weight'], df_static['total_value'], color='#6A0DAD', linewidth=3)
    ax1.set_title('Static Strategy: Total Season Value vs. Fixed Weight', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Judge Weight ($w$)', fontsize=12)
    ax1.set_ylabel('Total Commercial Value', fontsize=12)
    
    # 标记最优静态点
    max_static = df_static.loc[df_static['total_value'].idxmax()]
    ax1.axvline(max_static['weight'], color='orange', linestyle='--', linewidth=2, 
                label=f'Optimal Static $w={max_static["weight"]:.2f}$')
    ax1.legend(loc='upper right')

    # 右图：动态权重变化 (The Rollercoaster)
    ax2.plot(df_dynamic['week'], df_dynamic['optimal_weight'], marker='o', markersize=8, 
             linestyle='-', color='#008080', linewidth=2.5)
    ax2.set_title('Dynamic Strategy: Optimal Weight per Week', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Week Number', fontsize=12)
    ax2.set_ylabel('Optimal Judge Weight ($w$)', fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(0.5, color='gray', linestyle=':', label='Standard (0.5)')
    
    # 添加注释
    ax2.text(2, 0.95, 'High Judge Power\n(Filter weak contestants)', ha='center', fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax2.text(7, 0.1, 'High Fan Power\n(Create Controversy)', ha='center', fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2.legend()
    plt.tight_layout()
    plt.savefig('Figure1_Optimization_Strategy.png', dpi=300)
    print("Figure 1 saved.")

    # --- 图表 2: 争议的“甜蜜点”散点图 ---
    # 生成所有可能结果的数据
    all_scenarios = []
    for week in sorted(df_season['week'].unique())[:-1]:
        week_data = df_season[df_season['week'] == week].copy()
        for idx, person_row in week_data.iterrows():
            val, buzz, risk = calculate_commercial_value_tuned(person_row, week_data)
            all_scenarios.append({
                'contestant': person_row['contestant'],
                'buzz': buzz,
                'risk': risk,
                'net_value': val
            })
    df_tradeoff = pd.DataFrame(all_scenarios)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tradeoff['risk'], df_tradeoff['buzz'], 
                          c=df_tradeoff['net_value'], cmap='RdYlGn', 
                          s=120, alpha=0.8, edgecolors='grey')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Net Commercial Value', fontsize=12)
    
    plt.xlabel('Churn Risk (Audience Anger)', fontsize=12)
    plt.ylabel('Buzz (Social Media Attention)', fontsize=12)
    plt.title('The "Sweet Spot" of Controversy: Buzz vs. Risk', fontsize=16, fontweight='bold')
    
    # 标注区域
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.3)
    plt.text(200, 150, 'Danger Zone\n(High Risk)', color='red', fontsize=14, ha='center')
    plt.text(20, 150, 'Sweet Spot\n(Max Profit)', color='green', fontsize=14, ha='center')
    plt.text(20, 20, 'Boring Zone\n(No Impact)', color='gray', fontsize=14, ha='center')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure2_Tradeoff_Space.png', dpi=300)
    print("Figure 2 saved.")

# ==========================================
# 主程序执行
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    df_season = load_and_prep_data()
    print(f"Data loaded for Season 32. Weeks: {df_season['week'].unique()}")
    
    # 2. 运行模拟
    df_static, df_dynamic = simulate_strategies(df_season)
    print("Simulation complete.")
    print(f"Max Static Value: {df_static['total_value'].max():.2f}")
    print(f"Max Dynamic Value: {df_dynamic['max_value'].sum():.2f}")
    
    # 3. 绘图
    plot_results(df_static, df_dynamic, df_season)