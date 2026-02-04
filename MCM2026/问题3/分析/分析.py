import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 文件路径 ---
Q1_OUTPUT_FILE = 'dataset/estimated_fan_votes_final.csv'
RAW_DATA_FILE = 'dataset/2026_MCM_Problem_C_Data.csv'
EXTRA_DATA_FILE = 'dataset/dancing_with_the_stars_dataset.csv'

# ==============================================================================
# 模块 1: 多源数据融合 (Data Integration)
# ==============================================================================
def load_and_enrich_data():
    print(">>> Phase 1: Data Integration & Fusion...")
    
    # 1. 加载 Q1 结果
    if not os.path.exists(Q1_OUTPUT_FILE):
        print(f"Error: {Q1_OUTPUT_FILE} not found. Using dummy data.")
        n = 1000
        df_votes = pd.DataFrame({
            'season': np.random.randint(1, 35, n),
            'contestant': [f'Star_{i}' for i in range(n)],
            'week': np.random.randint(1, 11, n),
            'judge_share': np.random.beta(5, 5, n),
            'estimated_fan_share': np.random.beta(2, 5, n),
            'judge_rank': np.random.randint(1, 10, n)
        })
    else:
        df_votes = pd.read_csv(Q1_OUTPUT_FILE)

    # 2. 加载元数据
    df_raw = pd.read_csv(RAW_DATA_FILE)
    cols_raw = [c for c in ['season', 'celebrity_name', 'celebrity_age_during_season', 
                            'celebrity_industry', 'ballroom_partner'] if c in df_raw.columns]
    df_raw_meta = df_raw[cols_raw].drop_duplicates()
    df_raw_meta.rename(columns={
        'celebrity_name': 'contestant', 
        'celebrity_age_during_season': 'age',
        'celebrity_industry': 'industry',
        'ballroom_partner': 'partner'
    }, inplace=True)
    
    # 队友补充数据
    df_new_meta = pd.DataFrame()
    if os.path.exists(EXTRA_DATA_FILE):
        df_new = pd.read_csv(EXTRA_DATA_FILE)
        cols_new = [c for c in ['season', 'celebrity_name', 'celebrity_age_during_season', 
                                'celebrity_industry', 'ballroom_partner'] if c in df_new.columns]
        df_new_meta = df_new[cols_new].drop_duplicates()
        df_new_meta.rename(columns={
            'celebrity_name': 'contestant', 
            'celebrity_age_during_season': 'age_new',
            'celebrity_industry': 'industry_new',
            'ballroom_partner': 'partner_new'
        }, inplace=True)

    # 合并
    df_merged = pd.merge(df_votes, df_raw_meta, on=['season', 'contestant'], how='left')
    
    if not df_new_meta.empty:
        df_merged = pd.merge(df_merged, df_new_meta, on=['season', 'contestant'], how='left')
        for col in ['age', 'industry', 'partner']:
            if f'{col}_new' in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(df_merged[f'{col}_new'])
                df_merged.drop(columns=[f'{col}_new'], inplace=True)

    # 3. 清洗
    def simplify_industry(s):
        s = str(s).lower()
        if any(x in s for x in ['athlete', 'nfl', 'nba', 'olympic', 'football']): return 'Athlete'
        if any(x in s for x in ['actor', 'actress', 'movie', 'film']): return 'Actor'
        if any(x in s for x in ['singer', 'music', 'pop']): return 'Musician'
        if any(x in s for x in ['reality', 'bachelor', 'survivor']): return 'Reality Star'
        if any(x in s for x in ['host', 'presenter', 'news']): return 'TV Host'
        return 'Other'
    
    df_merged['industry_simple'] = df_merged['industry'].fillna('Other').apply(simplify_industry)
    df_merged['age'] = pd.to_numeric(df_merged['age'], errors='coerce').fillna(35)
    df_merged['partner'] = df_merged['partner'].fillna('Unknown')
    
    # 构造 Underdog
    df_merged.sort_values(['season', 'contestant', 'week'], inplace=True)
    if 'judge_rank' in df_merged.columns:
        df_merged['prev_rank'] = df_merged.groupby(['season', 'contestant'])['judge_rank'].shift(1).fillna(1)
        df_merged['is_underdog'] = (df_merged['prev_rank'] >= 4).astype(int)
    else:
        df_merged['is_underdog'] = 0
        
    print(f"Data Fusion Complete. Rows: {len(df_merged)}")
    return df_merged

# ==============================================================================
# 模块 2: 双轨模型训练
# ==============================================================================
def perform_dual_modeling(df):
    print(">>> Phase 2: Training Dual Models...")
    
    model_df = df[['estimated_fan_share', 'judge_share', 'age', 'industry_simple', 'is_underdog', 'partner']].dropna()
    
    # --- Model A: LMM ---
    scaler = StandardScaler()
    model_df['age_std'] = scaler.fit_transform(model_df[['age']])
    model_df['judge_std'] = scaler.fit_transform(model_df[['judge_share']])
    
    if model_df['industry_simple'].nunique() > 1:
        formula = "estimated_fan_share ~ age_std + C(industry_simple) + judge_std + is_underdog"
    else:
        formula = "estimated_fan_share ~ age_std + judge_std + is_underdog"
        
    try:
        if model_df['partner'].nunique() > 1:
            lmm = smf.mixedlm(formula, model_df, groups=model_df['partner'])
            res_lmm = lmm.fit()
            print("LMM Converged.")
        else:
            raise ValueError
    except:
        print("LMM failed. Fallback to OLS.")
        res_lmm = smf.ols(formula, model_df).fit()
        
    # --- Model B: Random Forest ---
    features = ['age', 'industry_simple', 'judge_share', 'is_underdog', 'partner']
    X = model_df[features].copy()
    y = model_df['estimated_fan_share']
    
    for c in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, y)
    
    return res_lmm, rf, model_df, features

# ==============================================================================
# 模块 3: 生成结果 (*** 重点修复部分 ***)
# ==============================================================================
def save_results(df, res_lmm, rf, model_df, features):
    print(">>> Phase 3: Generating Outputs...")
    
    # --- 1. 保存统计表格 (CSV) - 修复版 ---
    try:
        print("Extracting statistical table...")
        # 直接从模型结果对象中提取，而不是解析 summary 文本
        # 这对于 OLS 和 MixedLM 都通用且稳健
        
        # 提取系数、P值、标准误
        params = res_lmm.params
        pvalues = res_lmm.pvalues
        bse = res_lmm.bse # 标准误
        
        # 组合成 DataFrame
        stats_df = pd.DataFrame({
            'Coefficient': params,
            'Std_Error': bse,
            'P_Value': pvalues
        })
        
        # 添加显著性标记
        stats_df['Significance'] = stats_df['P_Value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        )
        
        # 保存
        stats_df.to_csv('Q3_Table_Statistical_Significance.csv')
        print("Success: Saved 'Q3_Table_Statistical_Significance.csv'")
        
        # 提取随机效应 (职业舞伴价值) - 仅针对 LMM
        if hasattr(res_lmm, 'random_effects'):
            print("Extracting random effects...")
            # random_effects 是一个字典，键是组名，值是Series
            re_data = []
            for group_name, effect in res_lmm.random_effects.items():
                # effect 通常是一个包含 'Group' 截距的 Series
                val = effect[0] if isinstance(effect, (list, np.ndarray)) else effect.iloc[0]
                re_data.append({'Partner': group_name, 'Value_Added': val})
            
            re_df = pd.DataFrame(re_data)
            re_df.sort_values('Value_Added', ascending=False, inplace=True)
            re_df.to_csv('Q3_Table_Pro_Partner_Value.csv', index=False)
            print("Success: Saved 'Q3_Table_Pro_Partner_Value.csv'")
            
    except Exception as e:
        print(f"Error saving tables: {str(e)}")
        # 打印详细错误以便调试
        import traceback
        traceback.print_exc()

    # --- 2. 绘制独立PDF图表 ---
    
    # Plot A
    plt.figure(figsize=(10, 6))
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='#264653', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Non-Linear Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('Q3_Plot1_Feature_Importance.pdf')
    plt.close()
    
    # Plot B
    plt.figure(figsize=(10, 6))
    params = res_lmm.params.drop(['Intercept', 'Group Var'], errors='ignore')
    plot_params = params[~params.index.str.contains('partner')]
    plot_params.plot(kind='barh', color='#E76F51')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Directional Factors (Statistical Model Coefficients)')
    plt.tight_layout()
    plt.savefig('Q3_Plot2_Statistical_Effects.pdf')
    plt.close()
    
    # Plot C
    plt.figure(figsize=(10, 6))
    model_df['judge_bin'] = pd.qcut(model_df['judge_share'], q=5, labels=['Very Low', 'Low', 'Med', 'High', 'Very High'])
    sns.boxplot(data=model_df, x='judge_bin', y='estimated_fan_share', palette='viridis')
    plt.title('The "Halo Effect" (Do Fans Follow Judges?)')
    plt.tight_layout()
    plt.savefig('Q3_Plot3_Halo_Effect.pdf')
    plt.close()
    
    # Plot D
    plt.figure(figsize=(10, 6))
    sns.regplot(data=model_df, x='age', y='estimated_fan_share', scatter_kws={'alpha':0.1}, 
                line_kws={'color':'red'}, lowess=True)
    plt.title('The "Age Penalty" Curve')
    plt.axvline(30, ls='--', color='gray')
    plt.tight_layout()
    plt.savefig('Q3_Plot4_Age_Curve.pdf')
    plt.close()
    
    # Plot E
    if hasattr(res_lmm, 'random_effects'):
        try:
            # 重新利用上面生成的 re_df 进行绘图，或者重新提取
            # 为安全起见重新简易提取用于绘图
            re_vals = [eff.iloc[0] for eff in res_lmm.random_effects.values()]
            re_names = list(res_lmm.random_effects.keys())
            temp_df = pd.DataFrame({'Name': re_names, 'Val': re_vals}).sort_values('Val', ascending=False).head(12)
            
            plt.figure(figsize=(12, 7))
            sns.barplot(x=temp_df['Val'], y=temp_df['Name'], palette='magma')
            plt.title('The "Kingmaker" Index (Pro Partner Value Added)')
            plt.tight_layout()
            plt.savefig('Q3_Plot5_Kingmaker_Index.pdf')
            plt.close()
        except:
            pass

    print("All plots saved as separate PDFs.")

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    df_fused = load_and_enrich_data()
    
    if len(df_fused) > 10:
        res, rf_model, model_data, feature_names = perform_dual_modeling(df_fused)
        save_results(df_fused, res, rf_model, model_data, feature_names)
        print("\n=== Q3 Analysis Complete ===")
        print("Check the folder for 5 PDF plots and 2 CSV tables.")
    else:
        print("Error: Insufficient data for analysis.")