import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ks_2samp
import os
import warnings
from matplotlib.backends.backend_pdf import PdfPages

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 文件路径 ---
BASELINE_FILE = 'estimated_fan_votes_final.csv' # 第一问结果 (Baseline)
ENHANCED_FILE = 'feedback_loop_iter_3.csv'      # 第三问反馈优化后的结果 (Enhanced)
RAW_DATA_FILE = 'dataset/2026_MCM_Problem_C_Data.csv'

# ==============================================================================
# 1. 数据加载与对齐
# ==============================================================================
def load_and_align_data():
    print(">>> Loading Data for Comparison...")
    
    if not os.path.exists(BASELINE_FILE) or not os.path.exists(ENHANCED_FILE):
        print("Error: Result files not found. Please run Q1 and Q3 codes first.")
        # 生成假数据演示
        n = 1000
        df_base = pd.DataFrame({
            'season': np.random.randint(1, 35, n),
            'contestant': [f'Star_{i}' for i in range(n)],
            'week': np.random.randint(1, 11, n),
            'estimated_fan_share': np.random.uniform(0.01, 0.2, n) # Q1: Uniform-ish
        })
        df_enh = df_base.copy()
        # Q3: More peaked (Gamma/Beta distribution)
        df_enh['estimated_fan_share'] = np.random.beta(2, 5, n)
        return df_base, df_enh
    
    df_base = pd.read_csv(BASELINE_FILE)
    df_enh = pd.read_csv(ENHANCED_FILE)
    
    # 确保列名一致
    common_cols = ['season', 'contestant', 'week', 'estimated_fan_share']
    df_base = df_base[common_cols].rename(columns={'estimated_fan_share': 'share_baseline'})
    df_enh = df_enh[common_cols].rename(columns={'estimated_fan_share': 'share_enhanced'})
    
    # 合并
    df_comp = pd.merge(df_base, df_enh, on=['season', 'contestant', 'week'], how='inner')
    
    print(f"Aligned Data Rows: {len(df_comp)}")
    return df_base, df_enh, df_comp

# ==============================================================================
# 2. 多维度评估指标计算
# ==============================================================================
def calculate_metrics(df_comp):
    print(">>> Calculating Metrics...")
    
    metrics = []
    
    # 1. 分布差异 (KS Test)
    # 检验新旧分布是否显著不同
    ks_stat, p_val = ks_2samp(df_comp['share_baseline'], df_comp['share_enhanced'])
    metrics.append({'Metric': 'Distribution Shift (KS Stat)', 'Value': ks_stat, 'Note': 'Higher = More Change'})
    
    # 2. 信息熵 (Entropy) - 衡量分布的“尖锐度”
    # Q1假设均匀，熵较高；Q3引入先验，应该更集中（熵降低），或者区分度更高
    # 我们将概率离散化后计算熵
    def calc_entropy(series):
        counts, _ = np.histogram(series, bins=20, density=True)
        # Add epsilon to avoid log(0)
        counts += 1e-10
        return entropy(counts)
    
    ent_base = calc_entropy(df_comp['share_baseline'])
    ent_enh = calc_entropy(df_comp['share_enhanced'])
    metrics.append({'Metric': 'Entropy (Baseline)', 'Value': ent_base, 'Note': 'Uniformity'})
    metrics.append({'Metric': 'Entropy (Enhanced)', 'Value': ent_enh, 'Note': 'Differentiation'})
    metrics.append({'Metric': 'Entropy Reduction', 'Value': ent_base - ent_enh, 'Note': 'Positive = More Informative'})
    
    # 3. 极端值捕捉能力 (Top 10% Share Sum)
    # 现实中，头部明星往往占据大量票数（马太效应）。Q1可能低估了头部。
    top10_base = df_comp.nlargest(int(len(df_comp)*0.1), 'share_baseline')['share_baseline'].sum() / df_comp['share_baseline'].sum()
    top10_enh = df_comp.nlargest(int(len(df_comp)*0.1), 'share_enhanced')['share_enhanced'].sum() / df_comp['share_enhanced'].sum()
    metrics.append({'Metric': 'Top 10% Concentration (Baseline)', 'Value': top10_base, 'Note': 'Pareto Principle'})
    metrics.append({'Metric': 'Top 10% Concentration (Enhanced)', 'Value': top10_enh, 'Note': 'Should be higher'})
    
    return pd.DataFrame(metrics)

# ==============================================================================
# 3. 可视化对比报告
# ==============================================================================
def generate_comparison_report(df_comp, metrics_df):
    print(">>> Generating Comparison Plots...")
    
    with PdfPages('Model_Comparison_Report.pdf') as pdf:
        
        # Plot 1: 分布形态对比 (KDE Plot)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df_comp['share_baseline'], fill=True, label='Baseline (Q1)', color='gray', alpha=0.3)
        sns.kdeplot(df_comp['share_enhanced'], fill=True, label='Enhanced (Q3 Feedback)', color='#E76F51', alpha=0.3)
        plt.title('Fig 1: Distribution Shift - From Uniform to Informed', fontsize=14)
        plt.xlabel('Estimated Fan Vote Share')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot 2: 散点图对比 (Scatter)
        # 展示哪些点的预测值发生了剧烈变化
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_comp, x='share_baseline', y='share_enhanced', alpha=0.3, color='#264653')
        plt.plot([0, 0.5], [0, 0.5], 'r--', label='No Change')
        
        # 标记偏离最大的点（Model Correction）
        df_comp['diff'] = np.abs(df_comp['share_enhanced'] - df_comp['share_baseline'])
        top_diff = df_comp.nlargest(5, 'diff')
        for _, row in top_diff.iterrows():
            plt.text(row['share_baseline'], row['share_enhanced'], 
                     f"{row['contestant']}\n(S{row['season']})", fontsize=8)
            
        plt.title('Fig 2: Model Corrections - Who did the model "re-evaluate"?', fontsize=14)
        plt.xlabel('Baseline Estimate (Q1)')
        plt.ylabel('Enhanced Estimate (Q3)')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot 3: 核心指标对比 (Bar Chart)
        # 对比 Top 10% Concentration
        plt.figure(figsize=(8, 6))
        m_filter = metrics_df[metrics_df['Metric'].str.contains('Concentration')]
        sns.barplot(data=m_filter, x='Metric', y='Value', palette=['gray', '#E76F51'])
        plt.title('Fig 3: Pareto Principle Check (Top 10% Voter Share)', fontsize=14)
        plt.ylabel('Share of Total Votes')
        plt.ylim(0, max(m_filter['Value'])*1.2)
        for i, v in enumerate(m_filter['Value']):
            plt.text(i, v, f"{v:.1%}", ha='center', va='bottom')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Save Metrics Table
    metrics_df.to_csv('Model_Comparison_Metrics.csv', index=False)
    print("Report Saved: Model_Comparison_Report.pdf & Model_Comparison_Metrics.csv")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    # 1. Load
    _, _, df_comp = load_and_align_data()
    
    if not df_comp.empty:
        # 2. Calculate
        metrics = calculate_metrics(df_comp)
        print("\n=== Metrics Summary ===")
        print(metrics)
        
        # 3. Visualize
        generate_comparison_report(df_comp, metrics)
    else:
        print("Error: No overlapping data found for comparison.")