import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 加载数据源
data_path = "数据预处理/男胎特征工程后数据.csv"
df = pd.read_csv(data_path)

# 提取目标字段
result_df = df[["Y染色体浓度", "孕妇BMI","18号染色体的Z值","年龄","体重","在参考基因组上比对的比例", "检测孕周_天数"]].copy()


# 归一化处理：对"18号染色体的Z值"、"年龄"、"在参考基因组上比对的比例"进行归一化
# 指定需要归一化的列
columns_to_normalize = ["18号染色体的Z值", "年龄", "在参考基因组上比对的比例"]

# 初始化MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# 对指定列进行归一化
result_df[columns_to_normalize] = scaler.fit_transform(result_df[columns_to_normalize]) * df["孕妇BMI"].max()
# 主成分分析：对指定列进行PCA并保留第一主成分
columns_to_pca = ["18号染色体的Z值", "年龄","体重", "在参考基因组上比对的比例"]

# 数据标准化（PCA前必须进行标准化）
scaler = StandardScaler()
scaled_data = scaler.fit_transform(result_df[columns_to_pca])

# 执行主成分分析，保留1个主成分
pca = PCA(n_components=1)
result_df["主成分1"] = pca.fit_transform(scaled_data)

# 删除原始特征列
result_df.drop(columns=columns_to_pca, inplace=True)
# 保存为新表格
output_path = "问题3/贪心算法/关键信息提取结果(PCA).csv"
result_df.to_csv(output_path, index=False, encoding="utf_8_sig")

print(f"提取完成！共{len(result_df)}条记录，已保存至：{output_path}")

