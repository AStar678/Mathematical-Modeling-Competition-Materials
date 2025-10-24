import pandas as pd

# 加载数据
data = pd.read_csv('问题3/损失优化/关键信息提取结果（完整信息）.csv')


# 查看数据集行数和列数
rows, columns = data.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(data.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(data.head().to_csv(sep='\t', na_rep='nan'))

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 查看Y染色体浓度的分布情况，并保留两位小数
distribution = data['Y染色体浓度'].describe().round(2)

# 提取特征和目标变量
X = data.drop('Y染色体浓度', axis=1)
y = data['Y染色体浓度']

# 标准化特征变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 由于Y染色体浓度是连续型数值，我们可以将其分箱转化为类别型变量，这里简单分10箱
# 获取分箱边界并保存
y_binned, bin_edges = pd.cut(y, bins=10, labels=False, retbins=True)

# 保存分箱边界用于后续转换
import pickle
with open('问题3/损失优化/Y染色体分箱边界.pkl', 'wb') as f:
    pickle.dump(bin_edges, f)

# 使用SMOTE进行过采样，设置较小的k_neighbors值
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_binned)

# 查看过采样后的数据分布
resampled_distribution = Counter(y_resampled)

# 创建一个包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制原始Y染色体浓度分布直方图
axes[0].hist(y, bins=10, edgecolor='black')
axes[0].set_title('原始Y染色体浓度分布')
axes[0].set_xlabel('Y染色体浓度')
axes[0].set_ylabel('频次')

# 绘制过采样后的Y染色体浓度分布直方图（分箱后）
axes[1].hist(y_resampled, bins=10, edgecolor='black')
axes[1].set_title('过采样后的Y染色体浓度分布（分箱后）')
axes[1].set_xlabel('Y染色体浓度分箱')
axes[1].set_ylabel('频次')

plt.tight_layout()

print('Y染色体浓度的分布情况：', distribution)
print('过采样后的数据分布：', resampled_distribution)

# 保存过采样后的数据
columns = data.columns.drop('Y染色体浓度')  # 获取特征列名
resampled_df = pd.DataFrame(X_resampled, columns=columns)
resampled_df['Y染色体浓度分箱'] = y_resampled  # 添加分箱后的目标变量

# 计算分箱中间值作为连续值近似
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
resampled_df['Y染色体浓度'] = bin_centers[y_resampled]
resampled_df.drop('Y染色体浓度分箱', axis=1, inplace=True)

resampled_df.to_csv('问题3/损失优化/SMOTE过采样数据.csv', index=False)
print('过采样数据已保存至问题3/损失优化/SMOTE过采样数据.csv')
