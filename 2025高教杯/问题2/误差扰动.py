import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df_original = pd.read_csv('问题2/关键信息提取结果.csv')

print('原始数据基本信息：')
df_original.info()

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 数据处理和模型构建函数
def process_data(data, data_type="原始"):
    # 创建高次项
    data['检测孕周_天数_2'] = data['检测孕周_天数'] **2
    data['孕妇BMI_2'] = data['孕妇BMI']** 2
    data['检测孕周_天数_3'] = data['检测孕周_天数'] **3
    data['孕妇BMI_3'] = data['孕妇BMI']** 3

    # 构建模型
    model = ols('Y染色体浓度 ~ 检测孕周_天数 + 检测孕周_天数_2 + 检测孕周_天数_3 + 孕妇BMI + 孕妇BMI_2 + 孕妇BMI_3', data=data).fit()

    # 进行方差分析
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # 保存模型权重
    results = pd.DataFrame({
        '变量': model.params.index,
        '系数(权重)': model.params.values,
        '标准误': model.bse.values,
        't值': model.tvalues.values,
        'p值(显著性)': model.pvalues.values,
        '95%置信区间下限': model.conf_int().values[:, 0],
        '95%置信区间上限': model.conf_int().values[:, 1]
    })
    
    output_dir = f'表格结果/问题2/{data_type}'
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(f'{output_dir}/模型权重.csv', index=False)

    # 打印模型权重（系数）
    print(f"\n=== {data_type}数据 - 模型权重系数 ===")
    weights = model.params
    print(f"截距项: {weights['Intercept']:.6f}")
    print(f"检测孕周_天数: {weights['检测孕周_天数']:.6f}")
    print(f"检测孕周_天数_2: {weights['检测孕周_天数_2']:.6f}")
    print(f"检测孕周_天数_3: {weights['检测孕周_天数_3']:.6f}")
    print(f"孕妇BMI: {weights['孕妇BMI']:.6f}")
    print(f"孕妇BMI_2: {weights['孕妇BMI_2']:.6f}")
    print(f"孕妇BMI_3: {weights['孕妇BMI_3']:.6f}")

    # 计算模型预测误差
    y_pred = model.predict(data)
    y_true = data['Y染色体浓度']
    errors = y_pred - y_true

    mean_error = errors.mean()
    var_error = errors.var()
    std_error = errors.std()

    # 打印误差分析结果
    print(f"\n=== {data_type}数据 - 模型预测误差分析 ===")
    print(f"误差均值 (ME): {mean_error:.6f}")
    print(f"误差方差 (Variance): {var_error:.6f}")
    print(f"误差标准差 (SD): {std_error:.6f}")

    # 保存误差分析结果
    error_results = pd.DataFrame({
        '统计量': ['误差均值', '误差方差', '误差标准差'],
        '数值': [mean_error, var_error, std_error]
    })
    error_results.to_csv(f'{output_dir}/预测误差分析.csv', index=False)
    print(f"{data_type}数据误差分析结果已保存至: {output_dir}/预测误差分析.csv")
    
    return model, data, output_dir

# 计算原始Y值的标准差，用于确定扰动幅度
y_std = df_original['Y染色体浓度'].std()

# 定义需要测试的扰动幅度列表
noise_levels = [0.1,0.5,1,2,10]

# 处理原始数据
model_original, df_original, dir_original = process_data(df_original, "原始")


targets = [0.04, 0.05, 0.06, 0.07]


# 定义主分析函数
def analyze_model(model, data, target, data_type, output_root):
    weights = model.params
    bmi_values = []
    min_weeks_values = []
    
    def f(x, y):
        return (weights['检测孕周_天数_3'] * x**3 + 
                weights['检测孕周_天数_2'] * x**2 + 
                weights['检测孕周_天数'] * x + 
                weights['孕妇BMI_3'] * y**3 + 
                weights['孕妇BMI_2'] * y**2 + 
                weights['孕妇BMI'] * y + 
                weights['Intercept'] - target)

    def find_x_min(y):
        try:
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

    y_values = np.arange(data['孕妇BMI'].min(), data['孕妇BMI'].max(), 0.1)
    for y in y_values:
        x_min = find_x_min(y)
        if x_min is not None:
            bmi_values.append(y)
            min_weeks_values.append(x_min / 7)

    # 数据排序与分段处理
    sorted_indices = np.argsort(bmi_values)
    sorted_bmi = np.array(bmi_values)[sorted_indices]
    sorted_weeks = np.array(min_weeks_values)[sorted_indices]
    
    if len(sorted_weeks) == 0:
        return None
    
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
        current_min = min(current_min, sorted_weeks[i])
        current_max = max(current_max, sorted_weeks[i])
        current_range = current_max - current_min
        
        if current_range >= target_segment_range or i == len(sorted_weeks) - 1:
            segments.append({
                'bmi': sorted_bmi[current_start:i+1],
                'weeks': sorted_weeks[current_start:i+1],
            })
            current_start = i + 1
            if current_start < len(sorted_weeks):
                current_min = sorted_weeks[current_start]
                current_max = sorted_weeks[current_start]
            else:
                break
    
    # 确保最终分为6个段
    while len(segments) < num_segments and len(segments) > 0:
        last = segments.pop()
        mid = len(last['bmi']) // 2
        segments.append({'bmi': last['bmi'][:mid], 'weeks': last['weeks'][:mid]})
        segments.append({'bmi': last['bmi'][mid:], 'weeks': last['weeks'][mid:]})
    
    # 创建可视化结果目录
    output_dir = f'{output_root}/可视化结果'
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制分段散点图
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    plt.figure(figsize=(12, 6))
    for i, seg in enumerate(segments):
        seg_color = plt.cm.cool(i / num_segments)
        plt.scatter(seg['bmi'], seg['weeks'], color=seg_color, alpha=0.7, s=30, label=f'段 {i+1}')
        
        # 添加段平均值水平线
        seg_mean = np.mean(seg['weeks'])
        seg_mid = (seg['bmi'].min() + seg['bmi'].max()) / 2
        plt.hlines(y=seg_mean, xmin=seg['bmi'].min(), xmax=seg['bmi'].max(), 
                  color=seg_color, linestyle='-', linewidth=2, alpha=0.8)
        plt.text(seg_mid, seg_mean + 0.1, f'均值: {seg_mean:.2f}', 
                 color='black', ha='center', fontweight='bold', fontsize=14)
    
    # 添加段分隔竖线和划分点标记
    for i, seg in enumerate(segments[:-1]):
        div_point = seg['bmi'][-1]
        plt.axvline(x=div_point, color='gray', linestyle='--', alpha=0.5)
        plt.scatter([div_point], [global_min_weeks - 0.2], color='red', s=50, zorder=5)
        plt.text(div_point, global_min_weeks - 0.4, f'划分点: {div_point:.2f}', 
                 color='red', ha='center', rotation=45, fontsize=13)
    
    plt.xlabel('孕妇BMI', fontsize=16)
    plt.ylabel('最小检测孕周数', fontsize=16)
    plt.title(f'{data_type}数据 - BMI与最小检测孕周数关系 (目标值: {target})', fontsize=18)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13)
    plt.ylim(global_min_weeks - 0.05*total_range, global_max_weeks + 0.05*total_range)
    plt.axhline(y=global_min_weeks, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=global_max_weeks, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'BMI与最小孕周关系_6段划分_{target}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{data_type}数据可视化结果已保存至: {output_path}")
    
    # 提取划分点和段平均值
    division_points = [seg['bmi'][-1] for seg in segments[:-1]]
    segment_means = [np.mean(seg['weeks']) for seg in segments]
    
    return {
        'target': target,
        'data_type': data_type,
        'division_points': division_points,
        'segment_means': segment_means,
        'global_min_weeks': global_min_weeks,
        'global_max_weeks': global_max_weeks
    }

all_results = []
# 循环测试不同扰动幅度
for level in noise_levels:
    # 生成当前扰动幅度的噪声
    noise = np.random.normal(0, y_std * level, size=len(df_original))
    
    # 创建带扰动的数据集
    df_noisy = df_original.copy()
    df_noisy['Y染色体浓度'] = df_original['Y染色体浓度'] + noise
    
    # 处理扰动数据，使用带幅度的类型名称
    data_type = f"扰动_{level}"
    model_noisy, df_noisy, dir_noisy = process_data(df_noisy, data_type)
    
    # 分析当前扰动幅度下的所有targets（新增）
    for target in targets:
        result = analyze_model(model_noisy, df_noisy, target, data_type, dir_noisy)
        if result:
            all_results.append(result)
    
    # 打印当前扰动幅度完成信息
    print(f"扰动幅度 {level} 处理完成，结果保存在: {dir_noisy}")
# 分析目标值



# # 分析原始数据
# for target in targets:
#     result = analyze_model(model_original, df_original, target, "原始", dir_original)
#     if result:
#         all_results.append(result)



# 保存汇总结果
output_csv = os.path.join('表格结果/问题2', '原始与扰动结果对比.csv')
pd.DataFrame(all_results).to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"原始与扰动结果对比已保存至: {output_csv}")

# 新增：计算不同扰动幅度下的误差
def calculate_errors(all_results, original_data_type="原始"):
    errors = []
    original_results = [r for r in all_results if r['data_type'] == original_data_type]
    
    for result in all_results:
        if result['data_type'] == original_data_type:
            continue
        
        # 找到对应target的原始结果
        orig_result = next((r for r in original_results if r['target'] == result['target']), None)
        if not orig_result:
            continue
        
        # 计算每个段的误差
        for seg_idx in range(len(result['segment_means'])):
            orig_mean = orig_result['segment_means'][seg_idx]
            perturbed_mean = result['segment_means'][seg_idx]
            abs_error = abs(perturbed_mean - orig_mean)
            rel_error = (abs_error / orig_mean) * 100 if orig_mean != 0 else 0
            
            errors.append({
                'target': result['target'],
                'noise_level': float(result['data_type'].split('_')[1]),
                'segment': seg_idx + 1,
                'original_mean': orig_mean,
                'perturbed_mean': perturbed_mean,
                'absolute_error': abs_error,
                'relative_error(%)': rel_error
            })
    
    return pd.DataFrame(errors)

# 计算并保存误差结果
error_df = calculate_errors(all_results)
error_output_csv = os.path.join('表格结果/问题2', '不同扰动幅度误差分析.csv')
error_df.to_csv(error_output_csv, index=False, encoding='utf-8-sig')
print(f"不同扰动幅度误差分析已保存至: {error_output_csv}")

# 生成对比图表
def plot_comparison(original_results, noisy_results, target):
    plt.figure(figsize=(14, 7))
    
    # 原始数据
    orig_segments = original_results[original_results['target'] == target]
    if not orig_segments.empty:
        orig_means = orig_segments.iloc[0]['segment_means']
        orig_divs = orig_segments.iloc[0]['division_points']
        plt.plot(range(1, len(orig_means)+1), orig_means, 'o-', color='blue', label='原始数据', linewidth=2)
    
    # 扰动数据
    noisy_segments = noisy_results[noisy_results['target'] == target]
    if not noisy_segments.empty:
        noisy_means = noisy_segments.iloc[0]['segment_means']
        noisy_divs = noisy_segments.iloc[0]['division_points']
        plt.plot(range(1, len(noisy_means)+1), noisy_means, 'o-', color='red', label='扰动数据', linewidth=2)
    
    plt.xlabel('段编号', fontsize=14)
    plt.ylabel('最小检测孕周数均值', fontsize=14)
    plt.title(f'目标值 {target} 时原始与扰动数据各段均值对比', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_dir = '可视化结果/问题2/对比'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/目标值_{target}_对比.png', dpi=300)
    plt.close()

# 生成各目标值的对比图
results_df = pd.DataFrame(all_results)
for target in targets:
    plot_comparison(
        results_df[results_df['data_type'] == '原始'],
        results_df[results_df['data_type'] == '扰动'],
        target
    )

print("所有对比图表已生成")