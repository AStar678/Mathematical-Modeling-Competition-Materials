from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd
import os

# 加载数据
df = pd.read_csv('数据预处理/男胎特征工程后数据.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

# 计算指定列与 Y 染色体浓度的皮尔逊相关系数和 p 值
columns = ['年龄', '身高', '体重', '检测抽血次数', '孕妇BMI', '原始读段数', '在参考基因组上比对的比例', '重复读段的比例',
           '唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
           'Y染色体的Z值', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例',
           '怀孕次数', '生产次数', '受孕方式_自然受孕', '受孕方式_IUI', '受孕方式_IVF', '检测孕周_天数']
result = []
for col in columns:
    corr, p_value = pearsonr(df[col], df['Y染色体浓度'])
    result.append([col, corr, p_value])

# 将结果转换为 DataFrame 并输出
result_df = pd.DataFrame(result, columns=['Column', 'Pearson_Correlation', 'P_Value'])
if not os.path.exists('表格结果/问题1'):
    os.makedirs('表格结果/问题1')
result_df = result_df.sort_values(by='P_Value')
result_df.to_csv('表格结果/问题1/皮尔逊检验结果.csv', index=False)