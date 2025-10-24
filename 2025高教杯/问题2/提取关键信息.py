import pandas as pd

# 加载数据源
data_path = "数据预处理/男胎特征工程后数据.csv"
df = pd.read_csv(data_path)

# 提取目标字段
result_df = df[["Y染色体浓度", "孕妇BMI", "检测孕周_天数"]].copy()

# 保存为新表格
output_path = "问题2/关键信息提取结果.csv"
result_df.to_csv(output_path, index=False, encoding="utf_8_sig")

print(f"提取完成！共{len(result_df)}条记录，已保存至：{output_path}")