import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 数据加载与预处理 (Data Loading)
# ==========================================
def load_and_prep_social_data():
    """
    读取 Excel 文件并计算社交影响力分数。
    如果找不到文件，则生成模拟数据。
    """
    file_name = '2026美赛C题补充数据集！.xlsx'
    # 尝试在当前目录或 dataset 目录下寻找文件
    possible_paths = [file_name, os.path.join('dataset', file_name)]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"正在读取文件: {path} ...")
                df = pd.read_excel(path)
                break
            except Exception as e:
                print(f"读取文件出错: {e}")
    
    # 如果没找到文件，生成模拟数据 (Fallback)
    if df is None:
        print("未找到 Excel 文件，正在生成模拟数据以演示代码功能...")
        return generate_synthetic_contestants()

    # 数据清洗与筛选
    if 'season' in df.columns:
        df = df[df['season'] == 32].copy()
    
    # 计算总粉丝数
    platforms = [
        'celebrity_instagram_followers', 'celebrity_twitter_followers',
        'celebrity_tiktok_followers', 'celebrity_youtube_subscribers'
    ]
    
    # 确保列存在，不存在则补0
    for p in platforms:
        if p not in df.columns:
            df[p] = 0
            
    df['total_followers'] = df[platforms].sum(axis=1)
    
    # 填充 0 值 (避免对数计算错误)
    median_val = df[df['total_followers'] > 0]['total_followers'].median()
    if pd.isna(median_val): median_val = 10000
    df['total_followers'] = df['total_followers'].replace(0, median_val).fillna(median_val)
    
    # 提取选手名单
    contestants = df[['celebrity_name', 'total_followers']].drop_duplicates()
    
    # 计算社交影响力分数 (1-10分)
    contestants['log_followers'] = np.log10(contestants['total_followers'] + 1)
    min_log = contestants['log_followers'].min()
    max_log = contestants['log_followers'].max()
    
    if max_log != min_log:
        contestants['social_power'] = 1 + 9 * (contestants['log_followers'] - min_log) / (max_log - min_log)
    else:
        contestants['social_power'] = 5.0
        
    return contestants

def generate_synthetic_contestants():
    """生成模拟选手数据 (仅在找不到文件时使用)"""
    names = [f'Dancer_{i}' for i in range(1, 15)]
    # 模拟几个大V (Lele Pons 级别) 和 一些普通人
    followers = [19000000, 8000000, 500000] + [np.random.randint(10000, 200000) for _ in range(11)]
    df = pd.DataFrame({'celebrity_name': names, 'total_followers': followers})
    
    # 计算分数
    df['log_followers'] = np.log10(df['total_followers'] + 1)
    df['social_power'] = 1 + 9 * (df['log_followers'] - df['log_followers'].min()) / (df['log_followers'].max() - df['log_followers'].min())
    return df

# ==========================================
# 2. 赛季数据模拟 (Season Simulation)
# ==========================================
def generate_season_simulation(contestants_df):
    """
    基于选手名单，生成 10 周的比赛基础数据（评分、得票潜力等）
    """
    np.random.seed(2026)
    weeks = range(1, 11)
    data_full = []
    
    names = contestants_df['celebrity_name'].tolist()
    social_map = dict(zip(contestants_df['celebrity_name'], contestants_df['social_power']))
    follower_map = dict(zip(contestants_df['celebrity_name'], contestants_df['total_followers']))
    
    for w in weeks:
        for name in names:
            # 1. 模拟评委分 (Judge): 正态分布，与名气无关
            judge_score = np.random.normal(0.5, 0.15)
            judge_score = np.clip(judge_score, 0.1, 0.9)
            
            # 2. 模拟粉丝基础票 (Fan): 技术 + 社交加成
            # 社交影响力(1-10) 贡献一部分基础票仓
            skill = np.random.normal(0.5, 0.15)
            social_bonus = social_map[name] / 10.0 * 0.5  # 社交分最高贡献 0.5 的底仓
            raw_fan = max(0.01, skill + social_bonus)
            
            data_full.append({
                'season': 32,
                'week': w,
                'contestant': name,
                'judge_raw': judge_score,
                'fan_raw': raw_fan,
                'social_power': social_map[name],
                'real_followers': follower_map[name]
            })
            
    df = pd.DataFrame(data_full)
    return df

# ==========================================
# 3. 核心优化模型 (Optimization Model)
# ==========================================
def calculate_social_value(eliminated, fair_elim, survivors_power, weeks_rem):
    """
    计算商业价值：瞬间热度(Buzz) vs 未来损失(Future Loss)
    """
    # 1. 排名数据
    rank_act = eliminated['estimated_fan_rank']
    rank_exp = fair_elim['estimated_fan_rank']
    
    # 2. 惊讶指数 (Surprise)
    surprise = max(0, rank_exp - rank_act)
    
    # 3. 瞬间热度 (Immediate Buzz)
    # 基础热度取决于选手的人气 (Social Power) 和排名
    # 淘汰大V (High Social Power) = 巨大新闻
    base_buzz = 100 * np.exp(-0.2 * (rank_act - 1)) + 60 * eliminated['social_power']
    buzz = base_buzz * (1 + 0.5 * surprise) # 惊讶是放大器
    
    # 4. 未来流量损失 (Future Traffic Loss - CLV)
    # 损失 = 该选手的社交分 * 剩余周数 * 权重
    # 越早淘汰大V，损失越大
    future_loss = eliminated['social_power'] * weeks_rem * 25 
    
    # 5. 风险惩罚 (Risk)
    risk = 0
    # 规则：动了超级大V (Power > 8) 且排名不是垫底
    if eliminated['social_power'] > 8 and rank_act < 8:
        risk += 500 * (eliminated['social_power']/10)
    # 规则：惊讶度过高
    if surprise >= 5:
        risk += 200
        
    # 6. 净价值
    net_val = buzz - risk - future_loss
    return net_val, buzz, future_loss, risk

def run_optimization(df_season, contestants_list):
    print("正在运行优化模型...")
    weeks = sorted(df_season['week'].unique())
    current_survivors = contestants_list.copy()
    history = []
    
    for w in weeks:
        # 如果只剩1人，结束
        if len(current_survivors) < 2: break
        
        # 获取当周幸存者的数据
        week_data = df_season[(df_season['week'] == w) & (df_season['contestant'].isin(current_survivors))].copy()
        
        # 归一化当周分数 (Shares)
        week_data['judge_share'] = week_data['judge_raw'] / week_data['judge_raw'].sum()
        week_data['estimated_fan_share'] = week_data['fan_raw'] / week_data['fan_raw'].sum()
        week_data['estimated_fan_rank'] = week_data['estimated_fan_share'].rank(ascending=False)
        
        # 基准：公平结果 (Fair Baseline)
        week_data['fair_score'] = 0.5 * week_data['judge_share'] + 0.5 * week_data['estimated_fan_share']
        fair_elim = week_data.sort_values('fair_score').iloc[0]
        
        # 寻找最优策略
        best_val = -float('inf')
        best_res = None
        weeks_rem = 10 - w
        
        # 遍历权重 w (0.0 到 1.0)
        for weight in np.linspace(0, 1, 51):
            week_data['score'] = weight * week_data['judge_share'] + (1 - weight) * week_data['estimated_fan_share']
            cand_elim = week_data.sort_values('score').iloc[0]
            
            # 计算剩余选手的总影响力 (用于上下文，虽未直接参与计算)
            survivor_power = week_data['social_power'].sum() - cand_elim['social_power']
            
            # 计算价值
            val, buzz, loss, risk = calculate_social_value(cand_elim, fair_elim, survivor_power, weeks_rem)
            
            if val > best_val:
                best_val = val
                best_res = {
                    'week': w,
                    'optimal_weight': weight,
                    'eliminated': cand_elim['contestant'],
                    'social_power': cand_elim['social_power'],
                    'followers': cand_elim['real_followers'],
                    'buzz': buzz,
                    'future_loss': loss,
                    'risk': risk,
                    'net_value': val,
                    'survivor_total_followers': week_data['real_followers'].sum() - cand_elim['real_followers']
                }
        
        # 执行淘汰
        elim_name = best_res['eliminated']
        current_survivors.remove(elim_name)
        history.append(best_res)
        
    return pd.DataFrame(history)

# ==========================================
# 4. 绘图与可视化 (Visualization)
# ==========================================
def plot_results(results_df):
    print("正在生成图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Figure 7: Harvest vs Invest (Trade-off) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = results_df['week']
    width = 0.35
    
    # 绘制双向柱状图
    ax.bar(x - width/2, results_df['buzz'], width, label='Immediate Buzz (Gain)', color='#FF6B6B')
    ax.bar(x + width/2, -results_df['future_loss'], width, label='Future Loss (Cost)', color='#4ECDC4')
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('The "Harvest vs. Invest" Trade-off', fontsize=15, fontweight='bold')
    ax.set_ylabel('Value Impact')
    ax.set_xlabel('Week')
    ax.legend()
    
    # 标注 Top Influencers
    for idx, row in results_df.iterrows():
        if row['social_power'] > 7.5: # 标注大V
            ax.text(row['week'], row['buzz'] + 10, 
                    f"{row['eliminated']}\n(Top Infl.)", 
                    ha='center', fontsize=9, color='darkred', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig('Figure7_Harvest_vs_Invest.pdf', dpi=300)
    print("- Figure 7 Saved.")
    
    # --- Figure 8: Audience Retention (Curve) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # 将粉丝数转换为百万单位
    y_vals = results_df['survivor_total_followers'] / 1e6
    
    ax.plot(results_df['week'], y_vals, marker='o', linestyle='-', color='#6A0DAD', linewidth=3)
    ax.fill_between(results_df['week'], y_vals, alpha=0.1, color='#6A0DAD')
    
    ax.set_title('Retention of Audience Reach (Total Follower Base)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Total Followers of Survivors (Millions)')
    ax.set_xlabel('Week')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加注释
    ax.text(results_df['week'].iloc[1], y_vals.iloc[1], 'Protective Phase\n(High Retention)', 
            fontsize=10, color='#6A0DAD', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Figure8_Audience_Retention.pdf', dpi=300)
    print("- Figure 8 Saved.")
    
    # --- Figure 9: Strategy Dynamics (Scatter) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散点大小和颜色代表被淘汰者的影响力
    scatter = ax.scatter(results_df['week'], results_df['optimal_weight'], 
                         c=results_df['social_power'], cmap='viridis', 
                         s=100 + results_df['social_power']*30, 
                         edgecolors='k', alpha=0.9)
    
    # 连接线
    ax.plot(results_df['week'], results_df['optimal_weight'], linestyle='--', color='gray', alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Social Power of Eliminated Contestant')
    
    ax.set_title('Adaptive Weight Strategy vs. Social Power', fontsize=15, fontweight='bold')
    ax.set_ylabel('Judge Weight (w)')
    ax.set_xlabel('Week')
    ax.set_ylim(-0.1, 1.1)
    
    # 标注基准线
    ax.axhline(0.5, color='red', linestyle=':', label='Fairness (0.5)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Figure9_Strategy_Dynamics.pdf', dpi=300)
    print("- Figure 9 Saved.")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    contestants_df = load_and_prep_social_data()
    
    if contestants_df is not None:
        print(f"成功加载 {len(contestants_df)} 名选手数据。")
        contestants_list = contestants_df['celebrity_name'].tolist()
        
        # 2. 生成模拟赛季数据
        df_season = generate_season_simulation(contestants_df)
        
        # 3. 运行优化模型
        results_df = run_optimization(df_season, contestants_list)
        
        # 4. 保存结果到 CSV
        csv_filename = 'social_optimization_results.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"数据结果已保存至: {csv_filename}")
        
        # 5. 绘图
        plot_results(results_df)
        
        print("\n=== 程序运行结束 ===")
    else:
        print("数据加载失败。")