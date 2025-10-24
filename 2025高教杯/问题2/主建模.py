import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv('问题2/关键信息提取结果.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

import statsmodels.api as sm
from statsmodels.formula.api import ols

df['检测孕周_天数_2'] = df['检测孕周_天数'] ** 2
df['孕妇BMI_2'] = df['孕妇BMI'] ** 2
# 构建三次项
df['检测孕周_天数_3'] = df['检测孕周_天数'] ** 3
df['孕妇BMI_3'] = df['孕妇BMI'] ** 3

# 构建模型
model = ols('Y染色体浓度 ~ 检测孕周_天数 + 检测孕周_天数_2 + 检测孕周_天数_3 + 孕妇BMI + 孕妇BMI_2 + 孕妇BMI_3', data=df).fit()

# 进行方差分析
anova_table = sm.stats.anova_lm(model, typ=2)

# 输出模型的摘要信息，包含参数的p值
# print('模型摘要信息：')
# print(model.summary())

# 输出方差分析表
print('方差分析表：')
print(anova_table)

results = pd.DataFrame({
    '变量': model.params.index,
    '系数(权重)': model.params.values,
    '标准误': model.bse.values,
    't值': model.tvalues.values,
    'p值(显著性)': model.pvalues.values,
    '95%置信区间下限': model.conf_int().values[:, 0],
    '95%置信区间上限': model.conf_int().values[:, 1]
})

results.to_csv('表格结果/问题2/模型权重.csv', index=False)

# 打印模型权重（系数）
print("\n=== 模型权重系数 ===")
weights = model.params
print(f"截距项: {weights['Intercept']:.6f}")
print(f"检测孕周_天数: {weights['检测孕周_天数']:.6f}")
print(f"检测孕周_天数_2: {weights['检测孕周_天数_2']:.6f}")
print(f"检测孕周_天数_3: {weights['检测孕周_天数_3']:.6f}")
print(f"孕妇BMI: {weights['孕妇BMI']:.6f}")
print(f"孕妇BMI_2: {weights['孕妇BMI_2']:.6f}")
print(f"孕妇BMI_3: {weights['孕妇BMI_3']:.6f}")


# 新增：计算模型预测误差的均值和方差
# 生成预测值
y_pred = model.predict(df)
# 获取真实值
y_true = df['Y染色体浓度']
# 计算误差
errors = y_pred - y_true

# 计算误差统计量
mean_error = errors.mean()
var_error = errors.var()
std_error = errors.std()

# 打印误差分析结果
print("\n=== 模型预测误差分析 ===")
print(f"误差均值 (ME): {mean_error:.6f}")
print(f"误差方差 (Variance): {var_error:.6f}")
print(f"误差标准差 (SD): {std_error:.6f}")

# 保存误差分析结果到CSV文件
error_results = pd.DataFrame({
    '统计量': ['误差均值', '误差方差', '误差标准差'],
    '数值': [mean_error, var_error, std_error]
})
output_dir = '表格结果/问题2'
error_results.to_csv(f'{output_dir}/预测误差分析.csv', index=False)
print(f"误差分析结果已保存至: {output_dir}/预测误差分析.csv")


def main(target):
    # 添加数据收集列表
    bmi_values = []
    min_weeks_values = []
    
    def f(x, y):
        return weights['检测孕周_天数_3'] * x**3 + weights['检测孕周_天数_2'] * x**2 + weights['检测孕周_天数'] * x + weights['孕妇BMI_3'] * y**3 + weights['孕妇BMI_2'] * y**2 + weights['孕妇BMI'] * y + weights['Intercept'] - target

    def find_x_min(y):
        try:
            # 寻找一个起始点
            x0 = root_scalar(f, args=(y,), bracket=[0, 300], method='brentq').root
            x_vals = np.linspace(0, 300, 1000)
            f_vals = f(x_vals, y)
            above_zero = f_vals > 0
            if np.any(above_zero):
                start = x_vals[np.argmax(above_zero)]
                return start
            else:
                return None
        except ValueError:
            return None

    y_values = np.arange(df['孕妇BMI'].min(), df['孕妇BMI'].max(), 0.1)
    for y in y_values:
        x_min = find_x_min(y)
        if x_min is not None:
            print(f"当孕妇BMI y = {y:.2f} 时，检测孕周数 x 的最小值是 {x_min / 7:.2f}")
            # 收集有效数据
            bmi_values.append(y)
            min_weeks_values.append(x_min / 7)
        else:
            print(f"当孕妇BMI y = {y:.2f} 时，没有找到满足条件的检测孕周数 x 的最小值")

    # 数据排序与分段处理
    sorted_indices = np.argsort(bmi_values)
    sorted_bmi = np.array(bmi_values)[sorted_indices]
    sorted_weeks = np.array(min_weeks_values)[sorted_indices]
    
    # 计算总极差和每段目标极差
    global_min_weeks = min(min_weeks_values)
    global_max_weeks = max(min_weeks_values)
    total_range = global_max_weeks - global_min_weeks
    num_segments = 6
    target_segment_range = total_range / num_segments
    
    segments = []
    current_start = 0
    current_min = sorted_weeks[0]
    current_max = sorted_weeks[0]
    
    for i in range(1, len(sorted_weeks)):
        # 更新当前段的最大最小值
        current_min = min(current_min, sorted_weeks[i])
        current_max = max(current_max, sorted_weeks[i])
        current_range = current_max - current_min
        
        # 当达到目标极差或最后一个点时分割
        if current_range >= target_segment_range or i == len(sorted_weeks) - 1:
            segments.append({
                'bmi': sorted_bmi[current_start:i+1],
                'weeks': sorted_weeks[current_start:i+1],
                'color': plt.cm.viridis(len(segments) / num_segments)
            })
            current_start = i + 1
            # 重置当前段的最小最大值
            if current_start < len(sorted_weeks):
                current_min = sorted_weeks[current_start]
                current_max = sorted_weeks[current_start]
            else:
                break
    
    # 确保最终分为6个段（处理数据不足情况）
    while len(segments) < num_segments and len(segments) > 0:
        last = segments.pop()
        mid = len(last['bmi']) // 2
        segments.append({
            'bmi': last['bmi'][:mid], 'weeks': last['weeks'][:mid],
            'color': plt.cm.viridis((len(segments)) / num_segments)
        })
        segments.append({
            'bmi': last['bmi'][mid:], 'weeks': last['weeks'][mid:],
            'color': plt.cm.viridis((len(segments)) / num_segments)
        })
    
    # 创建可视化结果目录
    output_dir = '可视化结果/问题2'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文显示
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 绘制分段散点图
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,          # 全局字体大小
        'axes.labelsize': 16,     # 坐标轴标签
        'axes.titlesize': 18,     # 标题
        'legend.fontsize': 13,    # 图例
        'xtick.labelsize': 12,    # x轴刻度
        'ytick.labelsize': 12     # y轴刻度
    })
    
    # 绘制分段散点图 - 更换为cool色系（青蓝到紫色，无黄色）
    plt.figure(figsize=(12, 6))
    for i, seg in enumerate(segments):
        # 使用cool色系替代viridis，避免黄色
        seg_color = plt.cm.cool(i / num_segments)
        plt.scatter(seg['bmi'], seg['weeks'], color=seg_color, alpha=0.7, s=30, label=f'段 {i+1}')
        
        # 添加段平均值水平线
        seg_mean = np.mean(seg['weeks'])
        seg_mid = (seg['bmi'].min() + seg['bmi'].max()) / 2
        plt.hlines(y=seg_mean, xmin=seg['bmi'].min(), xmax=seg['bmi'].max(), 
                  color=seg_color, linestyle='-', linewidth=2, alpha=0.8)
        plt.text(seg_mid, seg_mean + 0.1, f'均值: {seg_mean:.2f}', 
                 color='black', ha='center', fontweight='bold', fontsize=14)
    
    # 添加段分隔竖线和划分点标记（增大文本）
    for i, seg in enumerate(segments[:-1]):
        div_point = seg['bmi'][-1]
        plt.axvline(x=div_point, color='gray', linestyle='--', alpha=0.5)
        plt.scatter([div_point], [global_min_weeks - 0.2], color='red', s=50, zorder=5)
        plt.text(div_point, global_min_weeks - 0.4, f'划分点: {div_point:.2f}', 
                 color='red', ha='center', rotation=45, fontsize=13)
    
    # 更新标题和标签字体大小
    plt.xlabel('孕妇BMI', fontsize=16)
    plt.ylabel('最小检测孕周数', fontsize=16)
    plt.title('BMI与最小检测孕周数关系', fontsize=18)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13)
    
    # 设置统一的纵坐标范围
    plt.ylim(global_min_weeks - 0.05*total_range, global_max_weeks + 0.05*total_range)
    
    # 添加参考线与标签
    plt.axhline(y=global_min_weeks, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=global_max_weeks, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('孕妇BMI')
    plt.ylabel('最小检测孕周数')
    plt.title('BMI与最小检测孕周数关系')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'BMI与最小孕周关系_6段划分_{target}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"6段划分可视化结果已保存至: {output_path}")
    
    # 提取划分点和段平均值
    division_points = [seg['bmi'][-1] for seg in segments[:-1]]  # 每个段的结束BMI值作为划分点
    segment_means = [np.mean(seg['weeks']) for seg in segments]    # 每段纵坐标平均值
    
    return {
        'target': target,
        'division_points': division_points,
        'segment_means': segment_means
    }

targets = [0.04,0.05,0.06,0.07]
results = []
for target in targets:
    result = main(target)
    results.append(result)

# 保存汇总结果到CSV
output_csv = os.path.join('表格结果/问题2', '段划分结果汇总.csv')
pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"所有段划分结果已保存至: {output_csv}")