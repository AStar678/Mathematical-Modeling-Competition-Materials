import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, norm
import datetime



# ==========================================
# 2. 核心模型：贝叶斯推理引擎
# ==========================================
class BayesianVoteInference:
    def __init__(self, contestants, judge_scores, eliminated_name, rule_type='Percent'):
        self.contestants = contestants
        self.judge_scores = np.array(judge_scores)
        self.n = len(contestants)
        self.eliminated_idx = contestants.index(eliminated_name)
        self.rule_type = rule_type
        
        # 预计算裁判部分的数据，加速似然函数计算
        if self.rule_type == 'Percent':
            self.j_part = self.judge_scores / self.judge_scores.sum() * 100
        else:
            # Rank制：分数越高(好)，Rank数值越小(1)。
            # 使用 dense rank (1, 2, 2, 4...) 或 min rank
            self.j_part = rankdata(-self.judge_scores, method='min')

    def softmax(self, x):
        """将潜在人气值转化为投票比例 (0-1)"""
        e_x = np.exp(x - np.max(x)) # 减去max防止溢出
        return e_x / e_x.sum()

    def prior_log_prob(self, theta):
        """
        [可扩展部分] 先验分布
        目前假设：所有选手的人气服从标准正态分布 N(0, 1)
        扩展思路：可以引入选手特征 X，使得 theta ~ N(X*beta, sigma)
        """
        # 简单的 L2 正则化 / 高斯先验
        return -0.5 * np.sum(theta**2)

    def check_consistency(self, fan_shares):
        """
        检查生成的观众投票是否导致了正确的人被淘汰
        """
        if self.rule_type == 'Percent':
            # Percent制：总分 = 裁判% + 观众%
            # 分数最低者淘汰
            f_part = fan_shares * 100
            total_score = self.j_part + f_part
            predicted_eliminated = np.argmin(total_score)
            
        else: # Rank
            # Rank制：总分 = 裁判Rank + 观众Rank
            # Rank数值最大者(表现最差)淘汰
            # 观众票数越高 -> Rank数值越小
            f_part = rankdata(-fan_shares, method='min')
            total_score = self.j_part + f_part
            # 寻找Rank Sum最大的人
            predicted_eliminated = np.argmax(total_score)
            
        return predicted_eliminated == self.eliminated_idx

    def likelihood_log_prob(self, theta):
        """
        似然函数
        这里使用硬约束：如果不符合历史淘汰结果，概率为0 (log概率为 -inf)
        """
        fan_shares = self.softmax(theta)
        is_consistent = self.check_consistency(fan_shares)
        
        if is_consistent:
            return 0.0 # log(1)
        else:
            return -np.inf # log(0)

    def run_metropolis_hastings(self, n_samples=10000, step_size=0.5):
        """
        执行 MCMC 采样
        """
        samples = []
        # 初始化：随机生成直到找到一个可行解
        current_theta = np.random.randn(self.n)
        while self.likelihood_log_prob(current_theta) == -np.inf:
            current_theta = np.random.randn(self.n) * 2 # 扩大搜索范围
            
        current_log_prob = self.prior_log_prob(current_theta) + self.likelihood_log_prob(current_theta)
        
        accepted_count = 0
        
        for _ in range(n_samples):
            # 1. 提议 (Proposal)
            proposal = current_theta + np.random.normal(0, step_size, self.n)
            
            # 2. 计算后验概率 (Posterior)
            # 由于 Likelihood 是 0/-inf，这其实是在 Prior 分布上进行受限采样
            prior_lp = self.prior_log_prob(proposal)
            like_lp = self.likelihood_log_prob(proposal)
            
            if like_lp == -np.inf:
                # 违反约束，直接拒绝
                accept = False
            else:
                proposal_log_prob = prior_lp + like_lp
                # Metropolis 接受率
                alpha = np.exp(proposal_log_prob - current_log_prob)
                accept = np.random.rand() < alpha
            
            # 3. 更新状态
            if accept:
                current_theta = proposal
                current_log_prob = proposal_log_prob
                accepted_count += 1
            
            # 保存转换后的 Fan Shares (我们关心的是这个，不是 theta)
            samples.append(self.softmax(current_theta))
            
        acceptance_rate = accepted_count / n_samples
        return np.array(samples), acceptance_rate

# ==========================================
# 3. 主流程控制 (Main Execution)
# ==========================================
def main():
    print(">>> 正在加载数据...")
    df = df = pd.read_csv('DWTS_Preprocessed.csv')
    
    results_summary = []
    all_samples_storage = {} # 用于绘图
    
    # 按赛季和周分组处理
    groups = df.groupby(['Season', 'Week'])
    
    total_tasks = len(groups)
    print(f">>> 开始处理 {total_tasks} 个比赛周任务...\n")
    
    for (season, week), group in groups:
        print(f"正在分析 Season {season} Week {week} ...")
        
        contestants = group['Contestant'].tolist()
        scores = group['Judge_Score'].tolist()
        rule = group['Rule_Type'].iloc[0]
        
        # 找到被淘汰的人
        eliminated = group[group['Is_Eliminated'] == 1]['Contestant'].values
        if len(eliminated) == 0:
            print(f"  [跳过] S{season}W{week}: 无人淘汰")
            continue
        eliminated_name = eliminated[0]
        
        # --- 实例化模型并运行 ---
        model = BayesianVoteInference(contestants, scores, eliminated_name, rule_type=rule)
        
        # 运行 MCMC (预热 2000 次，采样 10000 次)
        # O奖Trick: 使用 Burn-in 去掉初始不稳定的采样
        raw_samples, acc_rate = model.run_metropolis_hastings(n_samples=12000, step_size=0.3)
        valid_samples = raw_samples[2000:] # 去掉前2000个 Burn-in
        
        print(f"  MCMC 接受率: {acc_rate:.2%}")
        
        # --- 整理结果 ---
        # 存储样本用于画图
        key = f"S{season}_W{week}"
        all_samples_storage[key] = pd.DataFrame(valid_samples, columns=contestants)
        
        # 计算统计量
        means = valid_samples.mean(axis=0)
        stds = valid_samples.std(axis=0)
        lower_ci = np.percentile(valid_samples, 2.5, axis=0)
        upper_ci = np.percentile(valid_samples, 97.5, axis=0)
        
        for i, name in enumerate(contestants):
            results_summary.append({
                "Season": season,
                "Week": week,
                "Contestant": name,
                "Rule": rule,
                "Is_Eliminated": 1 if name == eliminated_name else 0,
                "Est_Fan_Vote_Mean": means[i],
                "Est_Fan_Vote_Std": stds[i], # 这就是 Certainty 的度量
                "CI_Lower_95": lower_ci[i],
                "CI_Upper_95": upper_ci[i]
            })
            
    # ==========================================
    # 4. 结果保存
    # ==========================================
    print("\n>>> 正在保存结果 CSV ...")
    res_df = pd.DataFrame(results_summary)
    res_df.to_csv("DWTS_Estimated_Fan_Votes.csv", index=False)
    print("文件已保存: DWTS_Estimated_Fan_Votes.csv")
    
    # ==========================================
    # 5. 可视化 (PDF Output)
    # ==========================================
    print(">>> 正在生成可视化 PDF ...")
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('DWTS_Posterior_Analysis.pdf') as pdf:
        # 封面页
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'MCM Problem C: Bayesian Analysis Report\n\nPosterior Distributions of Fan Votes', 
                 ha='center', va='center', fontsize=20)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 遍历每个分析过的周，画图
        for key, sample_df in all_samples_storage.items():
            # 排序：按照估算的平均得票率排序，图表更好看
            sorted_indices = sample_df.mean().argsort()[::-1]
            sorted_df = sample_df.iloc[:, sorted_indices]
            
            plt.figure(figsize=(12, 6))
            
            # 使用小提琴图展示分布 (Bayesian 的精髓：展示不确定性)
            sns.violinplot(data=sorted_df, inner="quartile", palette="muted")
            
            s_str, w_str = key.split('_')
            plt.title(f"Posterior Fan Vote Distribution: {s_str} {w_str}")
            plt.ylabel("Estimated Vote Share (0-1)")
            plt.xticks(rotation=45)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            pdf.savefig() # 保存当前页
            plt.close()
            
    print("文件已保存: DWTS_Posterior_Analysis.pdf")
    print("\n全部任务完成！")

if __name__ == "__main__":
    main()