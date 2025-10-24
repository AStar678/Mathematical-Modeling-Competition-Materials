import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier  # 替换为LightGBM分类器
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, fbeta_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv('问题4/女胎检测数据_处理后.csv')

# 提取特征和目标变量
X = df.drop('染色体的非整倍体', axis=1)
y = df['染色体的非整倍体']

# 对缺失值进行均值填充
X = X.apply(lambda x: x.fillna(x.mean()))

# 使用 SMOTE 进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

if not os.path.exists('可视化结果/问题4'):
    os.makedirs('可视化结果/问题4')

# SMOTE结果可视化 - 类别分布对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title('原始数据类别分布')
plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled)
plt.title('SMOTE过采样后类别分布')
plt.tight_layout()
plt.savefig('可视化结果/问题4/SMOTE过采样类别分布.png')
plt.close()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 使用LightGBM模型（替换随机森林）
model = LGBMClassifier(random_state=42)  # LightGBM分类器
model.fit(X_train, y_train)

# 获取预测概率（用于寻找最优阈值）
y_proba = model.predict_proba(X_test)[:, 1]  # 只取正例的概率

# 寻找F2最优阈值（更重视召回率）
thresholds = np.arange(0.01, 1.0, 0.01)  # 生成0.01到0.99的阈值序列
f2_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)  # 根据阈值生成预测结果
    f2 = fbeta_score(y_test, y_pred_threshold, beta=2)  # 计算F2分数（beta=2更重视召回率）
    f2_scores.append(f2)

# 找到最优阈值（F2分数最高的阈值）
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds[optimal_idx]
max_f2_score = f2_scores[optimal_idx]

print(f"最优阈值: {optimal_threshold:.4f}")
print(f"最优阈值下的F2分数: {max_f2_score:.4f}")


# 使用最优阈值进行预测
y_pred = (y_proba >= optimal_threshold).astype(int)

# 评估模型
print('模型准确率：', accuracy_score(y_test, y_pred))
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred))
print('分类报告：\n', classification_report(y_test, y_pred))

# 可视化阈值与F2分数关系
plt.figure(figsize=(10, 6))
font = FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'], size=12)
plt.plot(thresholds, f2_scores, marker='o', markersize=4, color='royalblue', linewidth=2)
plt.axvline(x=optimal_threshold, color='firebrick', linestyle='--', linewidth=2,
            label=f'最优阈值: {optimal_threshold:.4f}\nF2分数: {max_f2_score:.4f}')
plt.xlabel('阈值', fontproperties=font, fontsize=14)
plt.ylabel('F2分数', fontproperties=font, fontsize=14)
plt.title('不同阈值下的F2分数变化', fontproperties=font, fontsize=16, pad=20)
plt.legend(prop=font)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('可视化结果/问题4/阈值与F2分数关系.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存模型评估结果到CSV
result_dir = '表格结果/问题4'
os.makedirs(result_dir, exist_ok=True)

# 1. 保存准确率
accuracy = accuracy_score(y_test, y_pred)
accuracy_df = pd.DataFrame({'评估指标': ['准确率'], '数值': [accuracy]})
accuracy_df.to_csv(f'{result_dir}/LightGBM模型准确率.csv', index=False)  # 修改文件名

# 2. 保存混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
cm_df.to_csv(f'{result_dir}/LightGBM混淆矩阵.csv')  # 修改文件名

# 3. 保存分类报告
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(f'{result_dir}/LightGBM分类报告.csv')  # 修改文件名

# 4. 保存综合评估指标（包含最优阈值）
evaluation_summary = {
    '准确率': accuracy,
    '精确率（负例）': class_report['0']['precision'],
    '精确率（正例）': class_report['1']['precision'],
    '召回率（负例）': class_report['0']['recall'],
    '召回率（正例）': class_report['1']['recall'],
    'F1分数（负例）': class_report['0']['f1-score'],
    'F1分数（正例）': class_report['1']['f1-score'],
    '宏平均F1': class_report['macro avg']['f1-score'],
    '加权平均F1': class_report['weighted avg']['f1-score'],
    '最优阈值': optimal_threshold,
    '最优F2分数': max_f2_score
}

summary_df = pd.DataFrame(list(evaluation_summary.items()), columns=['评估指标', '数值'])
summary_df.to_csv(f'{result_dir}/LightGBM模型评估综合指标.csv', index=False)  # 修改文件名

print(f'模型评估结果已保存至: {result_dir}')

# LightGBM特征重要性可视化
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

feature_importance = pd.Series(model.feature_importances_, index=X.columns)  # LightGBM使用feature_importances_属性
feature_importance = feature_importance.sort_values(ascending=False)

colors = sns.color_palette('Blues_d', n_colors=len(feature_importance))
bars = plt.barh(feature_importance.index, feature_importance.values, color=colors)

font = FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'], size=10)
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}',
             ha='left', va='center',
             fontproperties=font,
             fontweight='bold',
             color='darkslategray')

plt.title('LightGBM特征重要性', fontproperties=font, fontsize=16, pad=20, fontweight='bold')  # 修改标题
plt.xlabel('特征重要性分数', fontproperties=font, fontsize=14, labelpad=10)
plt.ylabel('特征名称', fontproperties=font, fontsize=14, labelpad=10)
plt.xticks(fontproperties=font, fontsize=12)
plt.yticks(fontproperties=font, fontsize=12)
sns.despine(top=True, right=True)

output_dir = '可视化结果/问题4'
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.85, box.height])
plt.tight_layout()
plt.savefig(output_dir + '/LightGBM特征重要性.png', dpi=300, bbox_inches='tight')  # 修改文件名
plt.close()

# 混淆矩阵热图可视化
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)

ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 annot_kws={'fontproperties': font, 'fontsize': 12})

plt.xlabel('预测标签', fontproperties=font, fontsize=14, labelpad=10)
plt.ylabel('真实标签', fontproperties=font, fontsize=14, labelpad=10)
plt.title('LightGBM混淆矩阵', fontproperties=font, fontsize=16, pad=20, fontweight='bold')  # 修改标题
plt.xticks(fontproperties=font, fontsize=12)
plt.yticks(fontproperties=font, fontsize=12)

plt.tight_layout()
plt.savefig(output_dir + '/LightGBM混淆矩阵.png', dpi=300, bbox_inches='tight')  # 修改文件名
plt.close()