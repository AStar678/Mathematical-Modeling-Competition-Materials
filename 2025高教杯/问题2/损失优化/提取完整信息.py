import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # 添加joblib导入

# 加载数据源
data_path = "数据预处理/男胎特征工程后数据.csv"
df = pd.read_csv(data_path)

# 提取目标字段
selected_columns = ["Y染色体浓度", "孕妇BMI", "检测孕周_天数"]
result_df = df[selected_columns].copy()

# 数据归一化处理（通道独立的归一化）
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(result_df)
normalized_df = pd.DataFrame(normalized_data, columns=result_df.columns)

# 保存归一化后的数据
output_path = "问题2/损失优化/关键信息提取结果（完整信息）.csv"
normalized_df.to_csv(output_path, index=False)
print(f"归一化数据已保存至: {output_path}")

# 创建需要计算归一化值的特定数据点
# 其他特征使用原始数据的平均值作为默认值
default_values = result_df.mean().to_dict()

# 要计算的特定数据点
specific_points = [
    {**default_values, "Y染色体浓度": 0.04, "检测孕周_天数": 70},
    {** default_values, "Y染色体浓度": 0.04, "检测孕周_天数": 84},
    {**default_values, "Y染色体浓度": 0.04, "检测孕周_天数": 175}
]

# 转换为DataFrame
specific_df = pd.DataFrame(specific_points)[selected_columns]

# 应用归一化
specific_normalized = scaler.transform(specific_df)
specific_normalized_df = pd.DataFrame(specific_normalized, columns=selected_columns)

# 显示结果
print("\n特定数据点的原始值:")
print(specific_df[["Y染色体浓度", "检测孕周_天数"]])  # 只显示我们关注的字段

print("\n对应的归一化值:")
print(specific_normalized_df[["Y染色体浓度", "检测孕周_天数"]])

# 反归一化验证
specific_original = scaler.inverse_transform(specific_normalized)
specific_original_df = pd.DataFrame(specific_original, columns=selected_columns)
print("\n反归一化验证（应与原始特定值一致）:")
print(specific_original_df[["Y染色体浓度", "检测孕周_天数"]])

# 保存归一化模型（替换原scaler.save()行）
joblib.dump(scaler, "问题2/损失优化/归一化模型.pkl")
print(specific_original_df[["Y染色体浓度", "检测孕周_天数"]])
