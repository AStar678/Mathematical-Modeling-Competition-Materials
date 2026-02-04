import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import rankdata, zscore
import warnings

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
# 设置字体以兼容英文显示 (如果需要中文，可尝试 'SimHei' 等)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 文件路径配置
DATA_PATH = 'dataset/2026_MCM_Problem_C_Data.csv'
INPUT_Q1_FILE = 'dataset/estimated_fan_votes_final.csv'
MAX_JUDGES = 4

# ==============================================================================
# 模块 1: 基础数据加载与 Q1 快速估算 (确保 Q2 有数据可用)
# ==============================================================================
def load_and_process_data(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        # 尝试在当前目录查找
        if os.path.exists(os.path.basename(filepath)):
            filepath = os.path.basename(filepath)
        else:
            raise FileNotFoundError(f"Data file {filepath} not found.")

    df = pd.read_csv(filepath)
    
    def parse_elimination(res):
        res = str(res).lower()
        if 'eliminated week' in res:
            try: return int(re.search(r'week (\d+)', res).group(1))
            except: return 99
        elif any(x in res for x in ['place', 'winner', 'runner', 'finalist']):
            return 99 
        return 99

    df['eliminated_week'] = df['results'].apply(parse_elimination)
    
    long_data = []
    for idx, row in df.iterrows():
        season = row['season']
        contestant = row['celebrity_name']
        elim_week = row['eliminated_week']
        
        for week in range(1, 16): 
            if f'week{week}_judge1_score' not in df.columns: break
            scores = []
            for j in range(1, MAX_JUDGES + 1):
                col = f'week{week}_judge{j}_score'
                if col in row and pd.notna(row[col]):
                    try: 
                        val = float(row[col])
                        if val > 0: scores.append(val)
                    except: continue
            
            if scores:
                week_total = sum(scores)
                long_data.append({
                    'season': season,
                    'contestant': contestant,
                    'week': week,
                    'total_judge_score': week_total,
                    'is_eliminated': (week == elim_week)
                })

    df_long = pd.DataFrame(long_data)
    stats = df_long.groupby(['season', 'week'])['total_judge_score'].agg(['sum', 'count'])
    df_long = df_long.merge(stats, on=['season', 'week'], suffixes=('', '_week_total'))
    df_long['judge_share'] = df_long['total_judge_score'] / df_long['sum']
    df_long['judge_rank'] = df_long.groupby(['season', 'week'])['total_judge_score'].rank(ascending=False, method='min')
    
    return df_long

class FanVoteEstimator:
    """Q1 简易估算器，用于在缺失中间文件时生成数据"""
    def __init__(self, df):
        self.df = df
        self.results = []
    
    def solve_all(self):
        print("正在运行 Q1 估算模型以生成基础数据...")
        seasons = sorted(self.df['season'].unique())
        for s in seasons:
            self.solve_season(s)
        return pd.DataFrame(self.results)
    
    def solve_season(self, season):
        season_data = self.df[self.df['season'] == season]
        weeks = sorted(season_data['week'].unique())
        contestants = season_data['contestant'].unique()
        current_priors = {c: 1.0/len(contestants) for c in contestants}
        method = 'rank' if (season <= 2 or season >= 28) else 'percent'
        
        for w in weeks:
            week_df = season_data[season_data['week'] == w]
            if week_df.empty: continue
            active = week_df['contestant'].tolist()
            eliminated = week_df[week_df['is_eliminated']]['contestant'].tolist()
            j_shares = dict(zip(week_df['contestant'], week_df['judge_share']))
            j_ranks = dict(zip(week_df['contestant'], week_df['judge_rank']))
            
            # 简化的求解逻辑 (Percent模式)
            # 这里为了代码紧凑，统一使用带约束的最小二乘法，Rank模式通过逻辑转换兼容
            active_prior_sum = sum(current_priors[c] for c in active)
            week_priors = {c: current_priors[c]/active_prior_sum for c in active}
            
            n = len(active)
            x0 = np.array([week_priors[c] for c in active])
            j_vals = np.array([j_shares[c] for c in active])
            
            def objective(x): return 0.7*np.sum((x - x0)**2) + 0.3*np.sum((x - j_vals)**2)
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            
            safe_indices = [i for i, c in enumerate(active) if c not in eliminated]
            elim_indices = [i for i, c in enumerate(active) if c in eliminated]
            
            # 通用约束：淘汰者表现 <= 幸存者
            # Percent: (J+F)_elim <= (J+F)_safe
            # Rank: (J_r+F_r)_elim >= (J_r+F_r)_safe (Rank数值大=差)
            # 我们统一估算 F_share，后续分析再区分
            for ei in elim_indices:
                for si in safe_indices:
                    constraints.append({
                        'type': 'ineq', 
                        'fun': lambda x, e=ei, s=si, Je=j_vals[ei], Js=j_vals[si]: 
                               (Js + x[s]) - (Je + x[e]) + 0.001
                    })
            
            res = minimize(objective, x0, method='SLSQP', bounds=[(0.01, 1.0)]*n, constraints=constraints)
            est_shares = dict(zip(active, res.x / np.sum(res.x)))
            
            # 更新先验
            for c, share in est_shares.items():
                current_priors[c] = 0.7*current_priors[c] + 0.3*share
                self.results.append({
                    'season': season, 'week': w, 'contestant': c,
                    'judge_share': j_shares[c], 'judge_rank': j_ranks[c],
                    'estimated_fan_share': share, 'is_eliminated': c in eliminated,
                    'method': method
                })

def ensure_data_exists():
    if os.path.exists(INPUT_Q1_FILE):
        print(f"检测到现有数据文件: {INPUT_Q1_FILE}")
        return pd.read_csv(INPUT_Q1_FILE)
    else:
        print("未检测到 Q1 结果文件，开始自动生成...")
        try:
            raw_df = load_and_process_data()
            estimator = FanVoteEstimator(raw_df)
            est_df = estimator.solve_all()
            est_df.to_csv(INPUT_Q1_FILE, index=False)
            return est_df
        except Exception as e:
            print(f"错误: 无法生成数据。请检查 {DATA_PATH} 是否存在。")
            raise e

# ==============================================================================
# 模块 2: Q2 核心仿真引擎 (反事实分析)
# ==============================================================================
class VotingSystemSimulator:
    def __init__(self, df):
        self.df = df
        
    def run_simulation(self):
        print("正在运行多赛制反事实仿真...")
        results = []
        for (season, week), group in self.df.groupby(['season', 'week']):
            eliminated_now = group[group['is_eliminated']]
            if len(eliminated_now) != 1: continue 
            
            actual_elim = eliminated_now['contestant'].iloc[0]
            contestants = group['contestant'].values
            j_shares = group['judge_share'].values
            f_shares = group['estimated_fan_share'].values
            
            # 1. Percent System
            p_scores = j_shares + f_shares
            p_bottom2 = np.argsort(p_scores)[:2]
            p_elim = contestants[p_bottom2[0]]
            
            # 2. Rank System
            j_ranks = rankdata(-j_shares, method='min')
            f_ranks = rankdata(-f_shares, method='min')
            r_scores = j_ranks + f_ranks
            r_bottom2 = np.argsort(-r_scores)[:2]
            r_elim = contestants[r_bottom2[0]]
            
            # 3. Judges' Save
            # Percent + Save
            idx1, idx2 = p_bottom2
            p_save_elim = contestants[idx2] if j_shares[idx1] > j_shares[idx2] else contestants[idx1]
            # Rank + Save
            idx1r, idx2r = r_bottom2
            r_save_elim = contestants[idx2r] if j_shares[idx1r] > j_shares[idx2r] else contestants[idx1r]
            
            results.append({
                'season': season, 'week': week, 'actual_eliminated': actual_elim,
                'percent_elim': p_elim, 'rank_elim': r_elim,
                'percent_save_elim': p_save_elim, 'rank_save_elim': r_save_elim
            })
        return pd.DataFrame(results)

# ==============================================================================
# 模块 3: 独立绘图与保存 (Separate PDF Generation)
# ==============================================================================
def plot_flip_rate(sim_df):
    """图1: 赛制敏感度分析 (翻转率)"""
    sim_df['is_flip'] = sim_df['percent_elim'] != sim_df['rank_elim']
    season_flip = sim_df.groupby('season')['is_flip'].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(season_flip.index, season_flip.values, color='#E76F51', alpha=0.8)
    plt.axhline(season_flip.mean(), color='gray', linestyle='--', label=f'Avg Disagreement: {season_flip.mean():.1%}')
    
    plt.title('Method Sensitivity (Disagreement Rate: Rank vs Percent)', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Probability of Different Outcome', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    filename = 'Q2_Plot1_Method_Sensitivity.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_judge_save_impact(sim_df):
    """图2: 评委拯救权的影响力对比"""
    impact_p = (sim_df['percent_elim'] != sim_df['percent_save_elim']).mean()
    impact_r = (sim_df['rank_elim'] != sim_df['rank_save_elim']).mean()
    
    plt.figure(figsize=(8, 6))
    x = ['Percent System', 'Rank System']
    y = [impact_p, impact_r]
    bars = plt.bar(x, y, color=['#2A9D8F', '#264653'], width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title("Impact of Introducing Judges' Save", fontsize=14)
    plt.ylabel('Frequency of Result Modification', fontsize=12)
    plt.ylim(0, 0.4)
    
    filename = 'Q2_Plot2_Judge_Save_Impact.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_micro_analysis(est_df):
    """图3: 四大争议案例微观画像 (双轴图)"""
    cases = [
        {'name': 'Jerry Rice', 'season': 2, 'color': '#2A9D8F', 'title': 'S2: The Rank Shield'},
        {'name': 'Billy Ray Cyrus', 'season': 4, 'color': '#E9C46A', 'title': 'S4: Country Music Force'},
        {'name': 'Bristol Palin', 'season': 11, 'color': '#F4A261', 'title': 'S11: Political Survivor'},
        {'name': 'Bobby Bones', 'season': 27, 'color': '#E76F51', 'title': 'S27: The Populism Peak'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, case in enumerate(cases):
        ax1 = axes[i]
        data = est_df[(est_df['season'] == case['season']) & (est_df['contestant'] == case['name'])].sort_values('week')
        if data.empty: continue
        
        # 左轴：评委排名 (倒序)
        l1 = ax1.plot(data['week'], data['judge_rank'], color='#264653', marker='o', lw=2, label='Judge Rank')
        ax1.set_ylabel('Judge Rank (Lower is Better)', color='#264653', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#264653')
        ax1.set_ylim(data['judge_rank'].max()+1, 0.5) # 倒序
        ax1.set_xlabel('Week')
        
        # 右轴：粉丝份额
        ax2 = ax1.twinx()
        l2 = ax2.plot(data['week'], data['estimated_fan_share'], color=case['color'], marker='s', ls='--', lw=2, label='Fan Share')
        ax2.set_ylabel('Fan Vote Share', color=case['color'], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=case['color'])
        
        ax1.set_title(case['title'], fontsize=12, fontweight='bold')
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)

    plt.suptitle('Micro-Analysis of Controversial Survivors', fontsize=16)
    plt.tight_layout()
    
    filename = 'Q2_Plot3_Micro_Survival_Analysis.pdf'
    plt.savefig(filename) # 不使用bbox_inches='tight'以防剪裁suptitle
    plt.close()
    print(f"Saved: {filename}")

def plot_populism_quadrant(est_df):
    """图4: 民粹主义象限图"""
    # 计算 Z-Scores
    df_z = est_df.copy()
    df_z['j_z'] = df_z.groupby(['season', 'week'])['judge_share'].transform(lambda x: zscore(x, nan_policy='omit'))
    df_z['f_z'] = df_z.groupby(['season', 'week'])['estimated_fan_share'].transform(lambda x: zscore(x, nan_policy='omit'))
    
    plt.figure(figsize=(10, 10))
    
    # 背景点
    plt.scatter(df_z['j_z'], df_z['f_z'], c='lightgray', alpha=0.3, s=15, label='Regular Contestants')
    
    # 坐标轴
    plt.axhline(0, c='k', lw=1)
    plt.axvline(0, c='k', lw=1)
    
    # 象限标注
    plt.text(2.2, 2.2, 'Elite Icons\n(High Skill / High Fans)', color='#2A9D8F', ha='center', fontweight='bold', fontsize=11)
    plt.text(-2.2, 2.2, 'Populist Heroes\n(Low Skill / High Fans)', color='#E76F51', ha='center', fontweight='bold', fontsize=11)
    plt.text(2.2, -2.2, 'Technical Experts\n(High Skill / Low Fans)', color='#264653', ha='center', fontsize=10)
    plt.text(-2.2, -2.2, 'Elimination Zone', color='gray', ha='center', fontsize=10)
    
    # 争议人物轨迹
    targets = {'Jerry Rice': '#2A9D8F', 'Bobby Bones': '#E76F51', 'Bristol Palin': '#F4A261'}
    for name, col in targets.items():
        track = df_z[df_z['contestant'] == name].sort_values('week')
        if not track.empty:
            plt.plot(track['j_z'], track['f_z'], c=col, lw=3, label=name, alpha=0.9)
            # 终点标记
            plt.scatter(track['j_z'].iloc[-1], track['f_z'].iloc[-1], c=col, marker='X', s=150, edgecolors='white', zorder=5)
            # 起点标记
            plt.scatter(track['j_z'].iloc[0], track['f_z'].iloc[0], c=col, marker='o', s=80, edgecolors='white', zorder=5)

    plt.title('The "Populism Quadrant" - Trajectories of Controversy', fontsize=15)
    plt.xlabel('Technical Merit (Z-Score)', fontsize=12)
    plt.ylabel('Popularity (Z-Score)', fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    filename = 'Q2_Plot4_Populism_Quadrant.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == "__main__":
    print("=== 开始执行 MCM Problem C 第二问完整分析 ===")
    
    # 1. 准备数据 (包含 Q1 结果)
    est_df = ensure_data_exists()
    
    # 2. 运行 Q2 仿真
    simulator = VotingSystemSimulator(est_df)
    sim_results = simulator.run_simulation()
    
    # 保存仿真数据表，供论文引用
    sim_results.to_csv('Q2_Simulation_Metrics.csv', index=False)
    print("仿真结果已保存至: Q2_Simulation_Metrics.csv")
    
    # 3. 生成独立 PDF 图表
    print("\n正在生成分析图表 (独立PDF)...")
    plot_flip_rate(sim_results)        # 图1: 赛制敏感度
    plot_judge_save_impact(sim_results) # 图2: 评委拯救影响
    plot_micro_analysis(est_df)         # 图3: 微观案例 (2x2)
    plot_populism_quadrant(est_df)      # 图4: 民粹象限
    
    print("\n所有操作完成！请查看生成的 .pdf 文件。")