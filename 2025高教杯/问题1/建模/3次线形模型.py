import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 加载数据
data_path = "数据预处理/男胎特征工程后数据.csv"
df = pd.read_csv(data_path)

# 定义目标变量和特征变量
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

# 检查数据完整性
print(f"数据集形状: {df.shape}")
print(f"缺失值情况:\n{df[features + [target]].isnull().sum()}")

# 处理缺失值 (删除或填充)
df_clean = df[features + [target]].dropna().copy()
print(f"处理后数据集形状: {df_clean.shape}")

# 存储每个特征的模型结果
summary_results = []

# 为每个特征建立三次多项式回归模型
for feature in features:
    print(f"\n===== 分析特征: {feature} =====")
    
    # 提取特征和目标变量
    X = df_clean[[feature]]
    y = df_clean[target]
    
    # 生成三次多项式特征
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # 添加截距项
    X_poly = sm.add_constant(X_poly)
    
    # 拟合模型
    model = sm.OLS(y, X_poly).fit()
    
    # 输出模型摘要
    print(model.summary())
    
    # 预测值
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    
    # 存储结果
    summary_results.append({
        "特征名称": feature,
        "R平方值": r2,
        "F统计量": model.fvalue,
        "F_pvalue": model.f_pvalue,
        "系数": model.params.to_dict(),
        "p值": model.pvalues.to_dict()
    })
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label="原始数据")
    plt.plot(X.sort_values(by=feature), y_pred[X.sort_values(by=feature).index], 'r-', linewidth=2, label="三次多项式拟合")
    plt.title(f"Y染色体含量与{feature}的三次多项式关系 (R²={r2:.4f})")
    plt.xlabel(feature)
    plt.ylabel("Y染色体含量")
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join("可视化结果/问题1", f"{feature}_三次多项式拟合.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

# 保存汇总结果到CSV
summary_df = pd.DataFrame(summary_results)
summary_csv_path = os.path.join("表格结果/问题1", "三次多项式回归结果汇总.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf_8_sig")

print(f"\n所有分析完成! 结果已保存")