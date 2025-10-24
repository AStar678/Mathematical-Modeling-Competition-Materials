import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             fbeta_score)
from imblearn.over_sampling import SMOTE
import os

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 构建逻辑回归模型
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 获取预测概率（用于寻找最优阈值）
y_proba = model.predict_proba(X_test)[:, 1]  # 正例的预测概率

# 寻找最优F2阈值（β=2，更重视召回率）
thresholds = np.arange(0.1, 0.91, 0.01)  # 阈值范围：0.1到0.9，步长0.01
f2_scores = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)  # 根据阈值生成预测标签
    f2 = fbeta_score(y_test, y_pred, beta=2)  # 计算F2分数
    f2_scores.append(f2)

# 找到最优阈值（F2最大的阈值）
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds[optimal_idx]
best_f2 = f2_scores[optimal_idx]

print(f'最优阈值: {optimal_threshold:.2f}')
print(f'最优阈值下的F2分数: {best_f2:.4f}')

# 使用最优阈值生成最终预测结果
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# 评估模型（基于最优阈值）
print('\n基于最优阈值的模型评估:')
print('模型准确率：', accuracy_score(y_test, y_pred_optimal))
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred_optimal))
print('分类报告：\n', classification_report(y_test, y_pred_optimal))

# 创建结果保存目录
result_dir = '表格结果/问题4'
os.makedirs(result_dir, exist_ok=True)

# 保存准确率
accuracy = accuracy_score(y_test, y_pred_optimal)
accuracy_df = pd.DataFrame({'评估指标': ['准确率'], '数值': [accuracy]})
accuracy_df.to_csv(f'{result_dir}/逻辑回归模型准确率.csv', index=False)

# 保存混淆矩阵
cm = confusion_matrix(y_test, y_pred_optimal)
cm_df = pd.DataFrame(cm, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
cm_df.to_csv(f'{result_dir}/逻辑回归混淆矩阵.csv')

# 保存分类报告
class_report = classification_report(y_test, y_pred_optimal, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(f'{result_dir}/逻辑回归分类报告.csv')

# 提取关键指标生成综合评估表（包含最优阈值）
evaluation_summary = {
    '准确率': accuracy,
    '精确率（宏观平均）': class_report['macro avg']['precision'],
    '召回率（宏观平均）': class_report['macro avg']['recall'],
    'F1分数（宏观平均）': class_report['macro avg']['f1-score'],
    'F2分数（最优阈值）': best_f2,
    '最优阈值': optimal_threshold
}

summary_df = pd.DataFrame(list(evaluation_summary.items()), columns=['评估指标', '数值'])
summary_df.to_csv(f'{result_dir}/逻辑回归模型评估综合指标.csv', index=False)

print(f'模型评估结果已保存至: {result_dir}')