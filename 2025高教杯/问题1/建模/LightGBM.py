import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lightgbm as lgb  # 导入LightGBM库
from sklearn.metrics import r2_score  # 用于置换检验性能对比（不输出）

# --------------------------
# 1. 基础设置（与历史代码完全一致）
# --------------------------
# 解决中文显示与负号问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 2. 数据加载与预处理（复用您的特征选择）
# --------------------------
# 加载数据源（路径与原始代码一致）
data_path = "数据预处理/男胎特征工程后数据.csv"
df = pd.read_csv(data_path)

# 目标变量与特征变量（完全复用您选择的7个单特征）
target = "Y染色体浓度"
features = [
    "检测抽血次数",
    "Y染色体的Z值",
    "孕妇BMI",
    "18号染色体的Z值",
    "年龄",
    "在参考基因组上比对的比例",
    "检测孕周_天数"
]

# 数据完整性检查（逻辑与原始代码一致）
print(f"数据集原始形状: {df.shape}")
print(f"缺失值情况:\n{df[features + [target]].isnull().sum()}")

# 处理缺失值（删除缺失值，保持与历史代码一致）
df_clean = df[features + [target]].dropna().copy()
print(f"删除缺失值后数据集形状: {df_clean.shape}")

# --------------------------
# 3. 置换检验与LightGBM参数配置
# --------------------------
# 置换检验参数（保证p值可靠性）
n_permutations = 1000  # 置换次数（≥1000确保精度，可按需调整）
random_seed = 42       # 固定随机种子，结果可重复
np.random.seed(random_seed)

# LightGBM核心参数（单特征场景优化，防过拟合）
lgb_params = {
    "objective": "regression",  # 回归任务
    "metric": "l2",            # 损失函数（均方误差，适配回归）
    "boosting_type": "gbdt",    # 梯度提升决策树（默认）
    "num_leaves": 15,          # 叶子节点数（≤2^max_depth，避免过拟合）
    "max_depth": 4,            # 树最大深度（单特征无需过深，4层足够）
    "learning_rate": 0.05,     # 学习率（小学习率+足够树数量，保证稳定）
    "n_estimators": 100,       # 总树数量（100棵树平衡性能与耗时）
    "verbose": -1,             # 关闭训练日志输出（仅保留关键信息）
    "random_state": random_seed# 固定随机种子，模型可重复
}

# --------------------------
# 4. 初始化结果存储（适配LightGBM专属参数）
# --------------------------
summary_results = []

# --------------------------
# 5. 循环为每个特征构建LightGBM+置换检验
# --------------------------
for feature in features:
    print(f"\n===== 分析特征: {feature} =====")
    
    # 提取特征与目标变量（LightGBM无需标准化，直接用原始特征）
    X_original = df_clean[[feature]]  # 单特征输入（与您原始单特征分析逻辑一致）
    y_true = df_clean[target].values  # 转为numpy数组，方便置换操作
    
    # --------------------------
    # 步骤1：训练原始LightGBM模型
    # --------------------------
    lgb_original = lgb.LGBMRegressor(**lgb_params)  # 加载参数
    lgb_original.fit(X_original, y_true)  # 直接用原始特征拟合（无需标准化）
    
    # --------------------------
    # 步骤2：计算原始模型性能（R²，仅用于置换检验对比，不输出）
    # --------------------------
    y_pred_original = lgb_original.predict(X_original)
    original_r2 = r2_score(y_true, y_pred_original)  # 仅用于p值计算，不对外展示
    
    # --------------------------
    # 步骤3：置换检验（核心！生成随机性能分布）
    # --------------------------
    permutation_r2 = []  # 存储每次置换的随机R²
    print(f"正在进行{ n_permutations }次置换检验...")
    
    for i in range(n_permutations):
        # 打乱目标变量Y的顺序（破坏特征与Y的真实关联，构建零假设）
        y_permuted = np.random.permutation(y_true)
        
        # 用打乱的Y训练随机LightGBM（参数与原始模型完全一致）
        lgb_perm = lgb.LGBMRegressor(**lgb_params)
        lgb_perm.fit(X_original, y_permuted)
        y_pred_perm = lgb_perm.predict(X_original)
        perm_r2 = r2_score(y_permuted, y_pred_perm)
        permutation_r2.append(perm_r2)
    
    # --------------------------
    # 步骤4：计算显著性p值（零假设下的极端概率）
    # --------------------------
    # p值 = 随机模型性能 ≥ 原始模型性能的次数 / 总置换次数
    count_extreme = sum(perm_r2 >= original_r2 for perm_r2 in permutation_r2)
    p_value = count_extreme / n_permutations
    
    # 输出显著性结果（按α=0.05行业标准判断）
    significance = "显著" if p_value < 0.05 else "不显著"
    print(f"特征 {feature} 的LightGBM模型显著性检验结果:")
    print(f"- 置换检验p值: {p_value:.4f}")
    print(f"- 关联显著性: {significance} (α=0.05)")
    
    # --------------------------
    # 步骤5：存储LightGBM专属结果（含集成模型参数）
    # --------------------------
    summary_results.append({
        "特征名称": feature,
        "置换检验p值": p_value,
        "置换次数": n_permutations,
        "LightGBM树数量": lgb_original.n_estimators,
        "最大树深度": lgb_original.max_depth,
        "叶子节点数": lgb_original.num_leaves,
        "学习率": lgb_original.learning_rate,
        "原始模型R²（用于检验）": round(original_r2, 4),  # 隐藏参数，方便调试
        "目标变量均值": round(y_true.mean(), 6),
        "目标变量标准差": round(y_true.std(), 6)
    })
    
    # --------------------------
    # 步骤6：可视化LightGBM拟合曲线（集成模型平滑特性）
    # --------------------------
    plt.figure(figsize=(10, 6))
    
    # 1. 绘制原始数据散点
    plt.scatter(X_original, y_true, alpha=0.5, label="原始数据", color="#1f77b4")
    
    # 2. 绘制LightGBM拟合曲线（集成模型拟合线平滑，需按特征排序）
    X_sorted = X_original.sort_values(by=feature)  # 按原始特征值排序（保证曲线连续）
    y_pred_sorted = lgb_original.predict(X_sorted)  # 对排序特征预测
    
    plt.plot(
        X_sorted,
        y_pred_sorted,
        'r-', linewidth=2.5,  # 平滑拟合线（LightGBM集成特性）
        label=f"LightGBM拟合曲线 (p={p_value:.4f})"  # 标注p值
    )
    
    # 图表配置（与历史代码风格一致）
    plt.title(f"Y染色体浓度与{feature}的LightGBM关系（p值={p_value:.4f}）")
    plt.xlabel(feature)
    plt.ylabel("Y染色体浓度")
    plt.legend(loc="best")
    plt.tight_layout()  # 避免标签被截断
    
    # --------------------------
    # 保存可视化结果（路径区分其他模型）
    # --------------------------
    plot_dir = "可视化结果/问题1"
    os.makedirs(plot_dir, exist_ok=True)  # 自动创建文件夹（不存在时）
    plot_path = os.path.join(plot_dir, f"{feature}_LightGBM_p值={p_value:.4f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # 保留完整标签
    plt.close()
    print(f"特征 {feature} 的LightGBM拟合图已保存至: {plot_path}")

# --------------------------
# 6. 保存汇总结果到CSV（含p值与LightGBM参数）
# --------------------------
summary_df = pd.DataFrame(summary_results)
result_dir = "表格结果/问题1"
os.makedirs(result_dir, exist_ok=True)
summary_csv_path = os.path.join(result_dir, "LightGBM显著性检验结果汇总.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf_8_sig")  # 支持中文显示

print(f"\n所有LightGBM模型显著性检验完成! 汇总结果已保存至: {summary_csv_path}")