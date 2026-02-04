import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import rankdata

# --- Style Configuration (Nature Style) ---
def set_nature_style():
    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif'
    try:
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    except:
        pass
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    sns.set_palette(colors)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.size'] = 10

def save_fig(fig, filename):
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved {filename}")

# --- 1. Re-run Estimator (确保数据包含 estimated_fan_vote_share) ---
# 为了保证代码独立运行，这里包含了一个简化版的估算器
class FanVoteEstimator:
    def __init__(self, df):
        self.df = df.copy()
        self.rank_seasons = list(range(1, 3)) + list(range(28, 35))
        
    def estimate_all(self):
        print("Re-calculating fan votes for validation...")
        results = []
        for (season, week), group in self.df.groupby(['season', 'week']):
            if len(group) < 2: continue
            method = 'rank' if season in self.rank_seasons else 'percent'
            votes = self._solve(group, method)
            for i, v in zip(group.index, votes):
                results.append({'idx': i, 'fan_share': v})
        res_df = pd.DataFrame(results).set_index('idx')
        # Merge back to original df
        merged = self.df.merge(res_df, left_index=True, right_index=True, how='left')
        return merged

    def _solve(self, group, method):
        n = len(group)
        j_scores = group['total_score'].values
        # Normalize judge scores
        if j_scores.sum() == 0:
             j_share = np.ones(n)/n
        else:
             j_share = j_scores / j_scores.sum()
             
        if not any(group['is_eliminated']): return j_share
        
        elim_idx = np.where(group['is_eliminated'])[0][0]
        
        # Objective: Min(Fan - Judge)^2
        fun = lambda x: np.sum((x - j_share)**2)
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x)-1}]
        
        # Constraints based on Method
        if method == 'percent':
            # Survivor Score > Eliminated Score
            for i in range(n):
                if i != elim_idx:
                    # (J_i + F_i) - (J_elim + F_elim) >= 0
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: (j_share[i]+x[i]) - (j_share[elim_idx]+x[elim_idx])})
        else: 
            # Rank Method Approximation
            # (J_i + F_i) - (J_elim + F_elim) >= 0 (loosely)
            for i in range(n):
                if i != elim_idx:
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: (j_share[i]+x[i]) - (j_share[elim_idx]+x[elim_idx])})

        try:
            res = minimize(fun, j_share, bounds=[(0.01,0.99)]*n, constraints=cons)
            return res.x if res.success else j_share
        except:
            return j_share

# --- 2. Validation Logic ---
def validate_consistency(df):
    records = []
    rank_seasons = list(range(1, 3)) + list(range(28, 35))
    
    print("Running Consistency Check...")
    for (season, week), group in df.groupby(['season', 'week']):
        if not any(group['is_eliminated']): continue
        if 'fan_share' not in group.columns or pd.isna(group['fan_share']).all(): continue

        method = 'rank' if season in rank_seasons else 'percent'
        
        j_scores = group['total_score'].values
        f_share = group['fan_share'].values
        elim_idx = np.where(group['is_eliminated'])[0][0]
        
        # Logic: Re-calculate the specific metric and see if the eliminated person is indeed last
        if method == 'percent':
            j_pct = (j_scores / (j_scores.sum() + 1e-9)) * 100
            f_pct = f_share * 100
            total = j_pct + f_pct
            # Margin = (Safe Person Score) - (Eliminated Person Score)
            # Should be positive
            sorted_total = np.sort(total)
            margin = sorted_total[1] - total[elim_idx] 
            correct = (np.argmin(total) == elim_idx)
        else:
            # Rank Method (Lower is better)
            j_rank = rankdata(-j_scores, method='min')
            f_rank = rankdata(-f_share, method='min')
            total = j_rank + f_rank
            # Margin = (Eliminated Score) - (Safe Score)
            # Since High Score = Bad in Rank sum context usually? 
            # Wait, Rank 1 is best. Rank N is worst. 
            # Sum of Ranks: Small = Good. Large = Bad.
            # So Eliminated should have MAX sum.
            # Margin = (Eliminated Sum) - (2nd Highest Sum)
            sorted_total = np.sort(total)
            margin = total[elim_idx] - sorted_total[-2]
            correct = (np.argmax(total) == elim_idx)
            
        records.append({
            'season': season, 'week': week, 'method': method,
            'correct': correct, 'margin': margin
        })
    return pd.DataFrame(records)

def compare_methods_sensitivity(df, target_season=5):
    print(f"Running Method Comparison on Season {target_season}...")
    records = []
    s_data = df[df['season'] == target_season]
    
    for week, group in s_data.groupby('week'):
        if not any(group['is_eliminated']): continue
        if 'fan_share' not in group.columns: continue
        
        j_scores = group['total_score'].values
        f_share = group['fan_share'].values
        contestants = group['contestant'].values
        elim_mask = group['is_eliminated'].values
        
        # 1. Original Method (Percent for S5)
        j_pct = (j_scores / (j_scores.sum() + 1e-9)) * 100
        f_pct = f_share * 100
        comb_pct = j_pct + f_pct
        pred_elim_percent = contestants[np.argmin(comb_pct)]
        
        # 2. Counterfactual Method (Rank)
        j_rank = rankdata(-j_scores, method='min')
        f_rank = rankdata(-f_share, method='min')
        comb_rank = j_rank + f_rank
        # Max rank sum is worst
        pred_elim_rank = contestants[np.argmax(comb_rank)]
        
        actual = contestants[elim_mask][0]
        
        records.append({
            'week': week,
            'actual_eliminated': actual,
            'predicted_percent': pred_elim_percent,
            'predicted_rank': pred_elim_rank,
            'match': (pred_elim_percent == pred_elim_rank)
        })
    return pd.DataFrame(records)

# --- Execution ---
if __name__ == "__main__":
    # 1. Load Data
    try:
        df_long = pd.read_csv('dataset/processed_data_long.csv')
    except FileNotFoundError:
        print("Error: 'processed_data_long.csv' not found. Please run preprocessing first.")
        exit()
        
    # Check column names
    if 'final_placement' in df_long.columns:
        df_long['placement'] = df_long['final_placement']

    # 2. Run Estimation
    est = FanVoteEstimator(df_long)
    df_with_votes = est.estimate_all()

    # 3. Validation Step 1: Consistency
    val_df = validate_consistency(df_with_votes)
    val_df.to_csv('validation_consistency_table.csv', index=False)
    print("Saved validation_consistency_table.csv")

    # Plot 1
    set_nature_style()
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    if not val_df.empty:
        sns.histplot(data=val_df, x='margin', hue='method', element='step', palette='viridis', ax=ax1)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_title('Validation of Model Consistency (Margin of Safety)', fontweight='bold')
        ax1.set_xlabel('Safety Margin (Distance from Elimination Threshold)')
        ax1.set_ylabel('Frequency')
        # Annotations
        ax1.text(0.6, 0.8, 'Robust Zone\n(High Margin)', color='green', transform=ax1.transAxes)
        ax1.text(0.05, 0.8, 'Risk Zone', color='red', transform=ax1.transAxes)
    save_fig(fig1, 'Fig1.4_Model_Consistency_Check.pdf')

    # 4. Validation Step 2: Method Comparison (Counterfactual)
    comp_df = compare_methods_sensitivity(df_with_votes, target_season=5)
    comp_df.to_csv('validation_method_comparison.csv', index=False)
    print("Saved validation_method_comparison.csv")

    # Plot 2
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if not comp_df.empty:
        counts = comp_df['match'].value_counts()
        sizes = [counts.get(True, 0), counts.get(False, 0)]
        if sum(sizes) > 0:
            plt.pie(sizes, labels=['Same Outcome', 'Different Outcome'], 
                    colors=['#4DBBD5', '#E64B35'], autopct='%1.1f%%', explode=[0, 0.1], startangle=90)
            plt.title('Sensitivity Analysis (Rank vs Percent Method)', fontweight='bold')
    save_fig(fig2, 'Fig1.5_Method_Sensitivity.pdf')

    print("\nAll validation steps completed successfully.")