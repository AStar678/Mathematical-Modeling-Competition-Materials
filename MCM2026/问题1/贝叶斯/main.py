import pandas as pd
import numpy as np
import re
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 全局绘图设置 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 如需显示中文请取消注释

DATA_PATH = 'dataset/2026_MCM_Problem_C_Data.csv'
MAX_JUDGES = 4

# --- 模块1: 数据加载与清洗 ---
def load_and_process_data(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        if os.path.exists(os.path.basename(filepath)):
            filepath = os.path.basename(filepath)
        else:
            raise FileNotFoundError(f"Data file {filepath} not found.")

    df = pd.read_csv(filepath)
    
    # 解析淘汰周次
    def parse_elimination(res):
        res = str(res).lower()
        if 'eliminated week' in res:
            try: return int(re.search(r'week (\d+)', res).group(1))
            except: return 99
        elif any(x in res for x in ['place', 'winner', 'runner', 'finalist']):
            return 99 
        return 99

    df['eliminated_week'] = df['results'].apply(parse_elimination)
    
    # 转换为长表 (Long Format)
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
                is_eliminated = (week == elim_week)
                
                long_data.append({
                    'season': season,
                    'contestant': contestant,
                    'week': week,
                    'total_judge_score': week_total,
                    'is_eliminated': is_eliminated
                })

    df_long = pd.DataFrame(long_data)
    
    # 计算辅助指标
    stats = df_long.groupby(['season', 'week'])['total_judge_score'].agg(['sum', 'count'])
    df_long = df_long.merge(stats, on=['season', 'week'], suffixes=('', '_week_total'))
    
    df_long['judge_share'] = df_long['total_judge_score'] / df_long['sum']
    # 排名: 1为最好
    df_long['judge_rank'] = df_long.groupby(['season', 'week'])['total_judge_score'].rank(ascending=False, method='min')
    
    return df_long

# --- 模块2: 逆向优化模型 ---
class FanVoteEstimator:
    def __init__(self, df):
        self.df = df
        self.results = []
        
    def solve_all(self):
        seasons = sorted(self.df['season'].unique())
        for s in seasons:
            self.solve_season(s)
        return pd.DataFrame(self.results)
    
    def solve_season(self, season):
        season_data = self.df[self.df['season'] == season]
        weeks = sorted(season_data['week'].unique())
        contestants = season_data['contestant'].unique()
        
        # 初始先验：均等
        current_priors = {c: 1.0/len(contestants) for c in contestants}
        
        # 规则判断
        method = 'percent'
        if season <= 2 or season >= 28:
            method = 'rank'
            
        for w in weeks:
            week_df = season_data[season_data['week'] == w]
            if week_df.empty: continue
            
            active = week_df['contestant'].tolist()
            eliminated = week_df[week_df['is_eliminated']]['contestant'].tolist()
            
            j_shares = dict(zip(week_df['contestant'], week_df['judge_share']))
            j_ranks = dict(zip(week_df['contestant'], week_df['judge_rank']))
            
            active_prior_sum = sum(current_priors[c] for c in active)
            week_priors = {c: current_priors[c]/active_prior_sum for c in active}
            
            # 求解
            if method == 'percent':
                est = self._solve_percent(active, j_shares, eliminated, week_priors)
            else:
                est = self._solve_rank(active, j_ranks, eliminated, week_priors)
            
            # 贝叶斯更新先验
            alpha = 0.3 
            for c, share in est.items():
                current_priors[c] = (1-alpha)*current_priors[c] + alpha*share
                
                self.results.append({
                    'season': season,
                    'week': w,
                    'contestant': c,
                    'judge_share': j_shares.get(c),
                    'judge_rank': j_ranks.get(c),
                    'estimated_fan_share': share,
                    'is_eliminated': c in eliminated,
                    'method': method
                })

    def _solve_percent(self, contestants, j_shares, eliminated, priors):
        n = len(contestants)
        x0 = np.array([priors[c] for c in contestants])
        j_vals = np.array([j_shares[c] for c in contestants])
        
        def objective(x):
            return 0.7*np.sum((x - x0)**2) + 0.3*np.sum((x - j_vals)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.01, 1.0) for _ in range(n)]
        
        safe_indices = [i for i, c in enumerate(contestants) if c not in eliminated]
        elim_indices = [i for i, c in enumerate(contestants) if c in eliminated]
        
        for ei in elim_indices:
            for si in safe_indices:
                # 约束: (Safe_Judge + Safe_Fan) >= (Elim_Judge + Elim_Fan)
                # 即 Safe - Elim >= 0
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, e=ei, s=si, Je=j_vals[ei], Js=j_vals[si]: 
                           (Js + x[s]) - (Je + x[e]) + 0.0001
                })
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        final_x = res.x if res.success else x0
        return dict(zip(contestants, final_x / np.sum(final_x)))

    def _solve_rank(self, contestants, j_ranks, eliminated, priors):
        sorted_priors = sorted(priors.items(), key=lambda x: x[1], reverse=True)
        expected_order = [x[0] for x in sorted_priors]
        
        candidates = [expected_order]
        # 随机扰动搜索200次
        for _ in range(200): 
            p = expected_order[:]
            if len(p) > 1:
                i, j = np.random.choice(len(p), 2, replace=False)
                p[i], p[j] = p[j], p[i]
                candidates.append(p)
            
        valid_candidates = []
        for perm in candidates:
            f_ranks = {c: i+1 for i, c in enumerate(perm)}
            scores = {c: j_ranks[c] + f_ranks[c] for c in contestants}
            
            if not eliminated: 
                valid_candidates.append(perm)
                continue
                
            is_valid = True
            for e in eliminated:
                score_e = scores[e]
                # Rank制: 排名总和越大越差。淘汰者总分 >= 所有存活者总分
                for s in contestants:
                    if s not in eliminated and score_e < scores[s]: 
                        is_valid = False; break
                if not is_valid: break
            
            if is_valid: valid_candidates.append(perm)
            
        best_p = valid_candidates[0] if valid_candidates else expected_order
        
        # 排名转份额 (线性分布)
        n = len(contestants)
        denom = n * (n + 1) / 2
        est_shares = {}
        for i, c in enumerate(best_p):
            est_shares[c] = (n - (i + 1) + 1) / denom 
            
        return est_shares

# --- 模块3: 评估与绘图 (含准确率计算) ---
def evaluate_and_plot(df):
    # 1. 计算分赛季准确率
    season_stats = []
    for season, s_data in df.groupby('season'):
        total = 0
        correct = 0
        
        for week, w_data in s_data.groupby('week'):
            elim = w_data[w_data['is_eliminated']]
            safe = w_data[~w_data['is_eliminated']]
            
            if elim.empty or safe.empty: continue
            total += 1
            method = w_data['method'].iloc[0]
            
            consistent = False
            if method == 'percent':
                # Elim Score <= Min Safe Score
                s_e = (elim['judge_share'] + elim['estimated_fan_share']).iloc[0]
                s_s = (safe['judge_share'] + safe['estimated_fan_share']).min()
                if s_s - s_e >= -0.001: consistent = True
            else:
                # Rank Score >= Max Safe Score (Big sum is bad)
                w_data = w_data.copy()
                w_data['f_rank'] = w_data['estimated_fan_share'].rank(ascending=False)
                elim_sub = w_data[w_data['is_eliminated']]
                safe_sub = w_data[~w_data['is_eliminated']]
                
                s_e = (elim_sub['judge_rank'] + elim_sub['f_rank']).iloc[0]
                s_s = (safe_sub['judge_rank'] + safe_sub['f_rank']).max()
                if s_e - s_s >= -0.001: consistent = True
            
            if consistent: correct += 1
            
        season_stats.append({
            'Season': season,
            'Total Eliminations': total,
            'Correct': correct,
            'Accuracy': correct/total if total>0 else 0
        })
    
    acc_df = pd.DataFrame(season_stats)
    print("\n=== Season Accuracy Table ===")
    print(acc_df)
    acc_df.to_csv('season_accuracy_table.csv', index=False)
    
    # 2. 绘图
    # Plot 1: Correlation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='judge_share', y='estimated_fan_share', hue='method', alpha=0.5)
    plt.title('Judge Share vs. Fan Share')
    plt.savefig('plot1_correlation.png')
    
    # Plot 2: Fan Trends (Example S23)
    s23 = df[df['season'] == 23]
    if not s23.empty:
        top = s23.groupby('contestant')['estimated_fan_share'].mean().nlargest(5).index
        plt.figure(figsize=(10, 5))
        for c in top:
            d = s23[s23['contestant'] == c]
            plt.plot(d['week'], d['estimated_fan_share'], marker='o', label=c)
        plt.title('Fan Vote Trends (Season 23)')
        plt.legend()
        plt.savefig('plot2_trends.png')
        
    # Plot 3: Accuracy Bar Chart (NEW)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=acc_df, x='Season', y='Accuracy', color='skyblue')
    plt.axhline(acc_df['Accuracy'].mean(), color='r', linestyle='--', label='Mean Accuracy')
    plt.title('Model Prediction Accuracy by Season')
    plt.savefig('plot3_accuracy.png')
    
    # Plot 4: Safety Margins
    margins = []
    for _, g in df.groupby(['season', 'week']):
        if g['method'].iloc[0] == 'percent':
            e = g[g['is_eliminated']]
            s = g[~g['is_eliminated']]
            if not e.empty and not s.empty:
                margins.append((s['judge_share']+s['estimated_fan_share']).min() - 
                               (e['judge_share']+e['estimated_fan_share']).iloc[0])
    plt.figure(figsize=(8, 5))
    sns.histplot(margins, bins=20, kde=True, color='green')
    plt.title('Safety Margin Distribution')
    plt.savefig('plot4_margins.png')
    
    # Plot 5: Rank Disagreement Heatmap
    df['f_rank'] = df.groupby(['season', 'week'])['estimated_fan_share'].rank(ascending=False)
    pivot = pd.crosstab(df['judge_rank'], df['f_rank']).iloc[:10, :10]
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='Blues')
    plt.title('Judge vs Fan Rank Disagreement')
    plt.savefig('plot5_heatmap.png')
    
    return acc_df

# --- 执行 ---
df_processed = load_and_process_data()
model = FanVoteEstimator(df_processed)
results_df = model.solve_all()
acc_table = evaluate_and_plot(results_df)
results_df.to_csv('estimated_fan_votes_final.csv', index=False)
print("Done.")