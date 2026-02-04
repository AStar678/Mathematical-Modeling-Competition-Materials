import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import warnings

# --- Global Settings ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# File Paths
RAW_DATA_FILE = 'dataset/2026_MCM_Problem_C_Data.csv'
EXTRA_DATA_FILE = 'dataset/dancing_with_the_stars_dataset.csv'

# ==============================================================================
# 1. Data Loading & Feature Engineering (Shared)
# ==============================================================================
def load_data():
    print(">>> Loading Data...")
    if not os.path.exists(RAW_DATA_FILE):
        print("Error: Data file not found.")
        return pd.DataFrame()

    df = pd.read_csv(RAW_DATA_FILE)
    
    # Parse Elimination
    def parse_elimination(res):
        res = str(res).lower()
        if 'eliminated week' in res:
            try: return int(res.split('week ')[1])
            except: return 99
        elif any(x in res for x in ['place', 'winner', 'runner', 'finalist']): return 99 
        return 99

    df['eliminated_week'] = df['results'].apply(parse_elimination)
    
    # Wide to Long
    long_data = []
    for idx, row in df.iterrows():
        season = row['season']
        contestant = row['celebrity_name']
        elim_week = row['eliminated_week']
        # Metadata
        age = row.get('celebrity_age_during_season', 35)
        ind = row.get('celebrity_industry', 'Other')
        partner = row.get('ballroom_partner', 'Unknown')
        
        for week in range(1, 16): 
            if f'week{week}_judge1_score' not in df.columns: break
            scores = []
            for j in range(1, 5):
                col = f'week{week}_judge{j}_score'
                if col in row and pd.notna(row[col]):
                    try: 
                        val = float(row[col])
                        if val > 0: scores.append(val)
                    except: continue
            
            if scores:
                week_total = sum(scores)
                long_data.append({
                    'season': season, 'contestant': contestant, 'week': week,
                    'total_judge_score': week_total,
                    'is_eliminated': (week == elim_week),
                    'age': age, 'industry': ind, 'partner': partner
                })

    df_long = pd.DataFrame(long_data)
    
    # Contextual Metrics
    stats = df_long.groupby(['season', 'week'])['total_judge_score'].agg(['sum', 'count'])
    df_long = df_long.merge(stats, on=['season', 'week'], suffixes=('', '_week_total'))
    df_long['judge_share'] = df_long['total_judge_score'] / df_long['sum']
    df_long['judge_rank'] = df_long.groupby(['season', 'week'])['total_judge_score'].rank(ascending=False, method='min')
    
    # Feature Engineering (Industry Simple)
    def simplify_industry(s):
        s = str(s).lower()
        if any(x in s for x in ['athlete', 'nfl', 'nba']): return 'Athlete'
        if any(x in s for x in ['actor', 'actress', 'movie']): return 'Actor'
        if any(x in s for x in ['singer', 'music', 'pop']): return 'Musician'
        if any(x in s for x in ['reality', 'bachelor']): return 'Reality TV'
        return 'Other'
    df_long['industry_simple'] = df_long['industry'].apply(simplify_industry)
    df_long['age'] = pd.to_numeric(df_long['age'], errors='coerce').fillna(35)
    
    return df_long

# ==============================================================================
# 2. Q3 Model: Factor Predictor (Learns from estimates)
# ==============================================================================
def train_factor_model(df, target_col='estimated_fan_share'):
    """Trains a model to predict fan share based on features (Age, Industry, etc.)"""
    # Features
    features = ['age', 'industry_simple', 'judge_share', 'partner']
    model_df = df.dropna(subset=[target_col]).copy()
    
    X = model_df[features].copy()
    y = model_df[target_col]
    
    # Encode
    encoders = {}
    for col in ['industry_simple', 'partner']:
        le = LabelEncoder()
        # Handle unknown labels in future by using 'unknown' class or similar
        # Here we just fit on current data
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, encoders

def predict_priors(df, model, encoders):
    """Predicts 'theoretical' fan share for all rows"""
    features = ['age', 'industry_simple', 'judge_share', 'partner']
    X = df[features].copy()
    
    for col in ['industry_simple', 'partner']:
        le = encoders[col]
        # Robust transform: map unseen to mode or similar (simplified here)
        X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        
    preds = model.predict(X)
    return preds

# ==============================================================================
# 3. Q1 Model: Estimator (Updated to accept custom priors)
# ==============================================================================
class FanVoteEstimator_v2:
    def __init__(self, df):
        self.df = df
        self.results = []
        
    def solve_with_priors(self, priors_col='prior_fan_share'):
        seasons = sorted(self.df['season'].unique())
        results = []
        
        for s in seasons:
            season_data = self.df[self.df['season'] == s]
            method = 'rank' if (s <= 2 or s >= 28) else 'percent'
            
            for w in sorted(season_data['week'].unique()):
                week_df = season_data[season_data['week'] == w]
                if week_df.empty: continue
                
                active = week_df['contestant'].tolist()
                eliminated = week_df[week_df['is_eliminated']]['contestant'].tolist()
                
                # Get specific priors for this week's contestants
                # Normalize them to sum to 1
                raw_priors = dict(zip(week_df['contestant'], week_df[priors_col]))
                total_p = sum(raw_priors.values())
                week_priors = {k: v/total_p for k,v in raw_priors.items()}
                
                j_shares = dict(zip(week_df['contestant'], week_df['judge_share']))
                j_ranks = dict(zip(week_df['contestant'], week_df['judge_rank']))
                
                # Solve
                if method == 'percent':
                    est = self._solve_percent(active, j_shares, eliminated, week_priors)
                else:
                    # For Rank method, prior share maps to rank expectation
                    # Simplified: use prior directly as share estimate
                    # Refinement: Monte Carlo with prior-biased sampling
                    est = week_priors # Placeholder for speed in iteration
                
                for c, share in est.items():
                    results.append({
                        'season': s, 'week': w, 'contestant': c,
                        'estimated_fan_share': share,
                        'judge_share': j_shares[c],
                        'is_eliminated': c in eliminated
                    })
        return pd.DataFrame(results)

    def _solve_percent(self, contestants, j_shares, eliminated, priors):
        n = len(contestants)
        # Priors are now INFORMED by the Factor Model
        x0 = np.array([priors[c] for c in contestants])
        j_vals = np.array([j_shares[c] for c in contestants])
        
        # Objective: Trust the Informed Prior more (0.8) than Judge (0.2)
        # Because the Prior now contains info about Age, Partner, etc.
        def objective(x):
            return 0.8*np.sum((x - x0)**2) + 0.2*np.sum((x - j_vals)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.01, 1.0) for _ in range(n)]
        
        # Constraints
        safe = [c for c in contestants if c not in eliminated]
        elim = [c for c in contestants if c in eliminated]
        for e in elim:
            for s in safe:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, ei=contestants.index(e), si=contestants.index(s), Je=j_shares[e], Js=j_shares[s]: 
                           (Js + x[si]) - (Je + x[ei]) + 0.001
                })
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(contestants, res.x / np.sum(res.x)))

# ==============================================================================
# 4. Iterative Feedback Loop
# ==============================================================================
def run_feedback_loop():
    print(">>> Starting Feedback Loop Optimization...")
    
    # 1. Init Data
    df = load_data()
    if df.empty: return
    
    # Iteration 0: Uniform Prior
    print("\n--- Iteration 0: Uniform Priors ---")
    df['prior_fan_share'] = 1.0 # Will be normalized in solver
    
    estimator = FanVoteEstimator_v2(df)
    current_estimates = estimator.solve_with_priors('prior_fan_share')
    
    # Merge estimates back to main df
    df = df.merge(current_estimates[['season', 'week', 'contestant', 'estimated_fan_share']], 
                  on=['season', 'week', 'contestant'], how='left')
    
    history = []
    
    # Loop
    for i in range(1, 4): # Run 3 iterations
        print(f"\n--- Iteration {i}: Learning from Q3 Model ---")
        
        # Step A: Train Q3 Model on current estimates
        # We learn: "Oh, young athletes usually get 25% votes, not 10%."
        model, encoders = train_factor_model(df, target_col='estimated_fan_share')
        
        # Step B: Predict new priors (Theoretical Share)
        # "Based on being a 24yo Athlete, Star X *should* get 25%."
        new_priors = predict_priors(df, model, encoders)
        df['prior_fan_share'] = new_priors
        
        # Step C: Re-run Q1 Estimation with new priors
        # "Find fan votes close to 25% that also satisfy elimination rules."
        new_estimates_df = estimator.solve_with_priors('prior_fan_share')
        
        # Calculate Change (RMSE)
        # Align data
        old_vals = df['estimated_fan_share'].fillna(0)
        # Update df with new estimates
        df = df.drop(columns=['estimated_fan_share'])
        df = df.merge(new_estimates_df[['season', 'week', 'contestant', 'estimated_fan_share']], 
                      on=['season', 'week', 'contestant'], how='left')
        
        new_vals = df['estimated_fan_share'].fillna(0)
        rmse = np.sqrt(np.mean((new_vals - old_vals)**2))
        print(f"Convergence Metric (RMSE): {rmse:.6f}")
        history.append(rmse)
        
        # Save iteration result
        df.to_csv(f'feedback_loop_iter_{i}.csv', index=False)

    # ==============================================================================
    # 5. Evaluation & Plotting
    # ==============================================================================
    print("\n>>> Generating Evaluation Report...")
    
    # Plot 1: Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history)+1), history, marker='o', linestyle='-', color='#E76F51')
    plt.title('Feedback Loop Convergence (Change in Estimates)')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE (vs Previous Iteration)')
    plt.xticks(range(1, len(history)+1))
    plt.savefig('Loop_Convergence.pdf')
    
    # Plot 2: Prior vs Posterior (Did the model learn?)
    # Compare Iter 0 (Uniform-ish) vs Iter 3 (Informed)
    # We visualize how "Informed Prior" aligns with "Final Estimate"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='prior_fan_share', y='estimated_fan_share', alpha=0.3, color='#2A9D8F')
    plt.plot([0, 0.5], [0, 0.5], 'r--')
    plt.title('Final Iteration: Theory (Prior) vs Reality (Posterior)')
    plt.xlabel('Theoretical Fan Share (Predicted by Q3 Model)')
    plt.ylabel('Actual Estimated Fan Share (Constraint-Satisfying)')
    plt.savefig('Loop_Prior_vs_Posterior.pdf')
    
    print("Done. Saved 'Loop_Convergence.pdf' and 'Loop_Prior_vs_Posterior.pdf'.")

if __name__ == "__main__":
    run_feedback_loop()