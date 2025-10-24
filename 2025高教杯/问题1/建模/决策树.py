import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归模型
from sklearn.metrics import r2_score  # 用于置换检验的性能对比（不输出）

# --------------------------
# 1. 基础设置（与原始代码/SVR代码一致）
# --------------------------
# 解决中文显示与负号问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 2. 数据加载与预处理（完全复用您的特征选择）
# --------------------------
# 加载数据源（与原始代码路径一致）
data_path = "数据预处理/男胎特征工程后数据.csv"
df = pd.read_csv(data_path)

# 目标变量与特征变量（完全复用您选择的7个特征）
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

# 数据完整性检查（与原始代码逻辑一致）
print(f"数据集原始形状: {df.shape}")
print(f"缺失值情况:\n{df[features + [target]].isnull().sum()}")

# 处理缺失值（删除缺失值，与原始代码一致）
df_clean = df[features + [target]].dropna().copy()
print(f"删除缺失值后数据集形状: {df_clean.shape}")

# --------------------------
# 3. 置换检验参数配置（保证p值可靠）
# --------------------------
n_permutations = 1000  # 置换次数（≥1000确保p值精度，可根据耗时调整）
random_seed = 42       # 固定随机种子，结果可重复
np.random.seed(random_seed)

# --------------------------
# 4. 初始化结果存储（适配决策树参数）
# --------------------------
summary_results = []

# --------------------------
# 5. 循环为每个特征构建决策树+置换检验
# --------------------------
for feature in features:
    print(f"\n===== 分析特征: {feature} =====")
    
    # 提取特征与目标变量（决策树无需标准化，直接用原始特征）
    X_original = df_clean[[feature]]  # 单特征（决策树支持多特征，但此处与您原始逻辑一致）
    y_true = df_clean[target].values  # 转为numpy数组，方便置换操作
    
    # --------------------------
    # 步骤1：训练原始决策树模型（控制过拟合是关键）
    # --------------------------
    # 决策树参数说明（针对单特征场景优化，避免过拟合）：
    # - max_depth=5：限制树深度（单特征无需过深，5层足够拟合非线性且防过拟合）
    # - min_samples_split=10：至少10个样本才分裂节点（减少噪声影响）
    # - random_state：固定随机种子，保证每次训练结果一致
    dt_model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        random_state=random_seed
    )
    dt_model.fit(X_original, y_true)  # 决策树直接用原始特征拟合（无需标准化）
    
    # --------------------------
    # 步骤2：计算原始模型性能（R²，仅用于置换检验对比，不输出）
    # --------------------------
    y_pred_original = dt_model.predict(X_original)
    original_r2 = r2_score(y_true, y_pred_original)  # 仅用于p值计算，不对外展示
    
    # --------------------------
    # 步骤3：置换检验（核心！生成随机性能分布，计算p值）
    # --------------------------
    permutation_r2 = []  # 存储每次置换的随机R²
    print(f"正在进行{ n_permutations }次置换检验...")
    
    for i in range(n_permutations):
        # 打乱目标变量Y的顺序（破坏特征与Y的真实关联，构建零假设场景）
        y_permuted = np.random.permutation(y_true)
        
        # 用打乱的Y训练随机决策树，计算随机性能
        dt_perm = DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=10,
            random_state=random_seed  # 保持参数与原始模型一致
        )
        dt_perm.fit(X_original, y_permuted)
        y_pred_perm = dt_perm.predict(X_original)
        perm_r2 = r2_score(y_permuted, y_pred_perm)
        permutation_r2.append(perm_r2)
    
    # --------------------------
    # 步骤4：计算显著性p值（零假设下的极端概率）
    # --------------------------
    # p值定义：随机模型性能 ≥ 原始模型性能的次数 / 总置换次数
    count_extreme = sum(perm_r2 >= original_r2 for perm_r2 in permutation_r2)
    p_value = count_extreme / n_permutations
    
    # 输出显著性结果（按α=0.05判断，行业通用标准）
    significance = "显著" if p_value < 0.05 else "不显著"
    print(f"特征 {feature} 的决策树模型显著性检验结果:")
    print(f"- 置换检验p值: {p_value:.4f}")
    print(f"- 关联显著性: {significance} (α=0.05)")
    
    # --------------------------
    # 步骤5：存储决策树专属结果（含模型结构参数）
    # --------------------------
    summary_results.append({
        "特征名称": feature,
        "置换检验p值": p_value,
        "置换次数": n_permutations,
        "决策树最大深度": dt_model.max_depth,
        "最小分裂样本数": dt_model.min_samples_split,
        "总节点数": dt_model.tree_.node_count,  # 决策树特有：总节点数量
        "叶子节点数": sum(1 for i in range(dt_model.tree_.node_count) if dt_model.tree_.children_left[i] == -1),
        "目标变量均值": y_true.mean(),
        "目标变量标准差": y_true.std()
    })
    
    # --------------------------
    # 步骤6：可视化决策树拟合曲线（分段特性，需排序特征）
    # --------------------------
    plt.figure(figsize=(10, 6))
    
    # 1. 绘制原始数据散点
    plt.scatter(X_original, y_true, alpha=0.5, label="原始数据", color="#1f77b4")
    
    # 2. 绘制决策树拟合曲线（决策树是分段函数，需按特征排序确保曲线连续）
    X_sorted = X_original.sort_values(by=feature)  # 按原始特征值排序
    y_pred_sorted = dt_model.predict(X_sorted)      # 对排序后的特征预测
    
    plt.plot(
        X_sorted,
        y_pred_sorted,
        'r-', linewidth=2.5,  # 决策树拟合线（分段）
        label=f"决策树拟合曲线 (p={p_value:.4f})"  # 标注p值
    )
    
    # 图表配置（与原始代码风格一致）
    plt.title(f"Y染色体浓度与{feature}的决策树关系（p值={p_value:.4f}）")
    plt.xlabel(feature)
    plt.ylabel("Y染色体浓度")
    plt.legend(loc="best")
    plt.tight_layout()  # 避免标签被截断
    
    # --------------------------
    # 保存可视化结果（路径区分其他模型）
    # --------------------------
    plot_dir = "可视化结果/问题1"
    os.makedirs(plot_dir, exist_ok=True)  # 自动创建文件夹（不存在时）
    plot_path = os.path.join(plot_dir, f"{feature}_决策树_p值={p_value:.4f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # 保留完整标签
    plt.close()
    print(f"特征 {feature} 的决策树拟合图已保存至: {plot_path}")

# --------------------------
# 6. 保存汇总结果到CSV（含p值与决策树参数）
# --------------------------
summary_df = pd.DataFrame(summary_results)
result_dir = "表格结果/问题1"
os.makedirs(result_dir, exist_ok=True)
summary_csv_path = os.path.join(result_dir, "决策树显著性检验结果汇总.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf_8_sig")  # 支持中文显示

print(f"\n所有决策树模型显著性检验完成! 汇总结果已保存至: {summary_csv_path}")