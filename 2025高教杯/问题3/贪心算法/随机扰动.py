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

# 数据预处理与模型训练函数
def prepare_data_and_train_model(df_original):
    """预处理数据并训练模型"""
    df = df_original.copy()
    
    # 修复列名问题
    df.rename(columns={
        '18号染色体的Z值': '染色体18号Z值',
        '在参考基因组上比对的比例': '参考基因组比对比例'
    }, inplace=True)
    
    # 创建二次项和三次项
    df['检测孕周_天数_2'] = df['检测孕周_天数'] **2
    df['孕妇BMI_2'] = df['孕妇BMI']** 2
    df['检测孕周_天数_3'] = df['检测孕周_天数'] **3
    df['孕妇BMI_3'] = df['孕妇BMI']** 3
    df['主成分1_2'] = df['主成分1'] **2
    df['主成分1_3'] = df['主成分1']** 3
    
    # 构建模型
    model = ols('Y染色体浓度 ~ 检测孕周_天数 + 检测孕周_天数_2 + 检测孕周_天数_3 + 孕妇BMI + 孕妇BMI_2 + 孕妇BMI_3 + 主成分1 + 主成分1_2 + 主成分1_3', data=df).fit()
    
    return df, model

# 添加随机扰动的函数
def add_random_perturbation(df, features, epsilon=0.05):
    """为指定特征添加随机扰动"""
    df_perturbed = df.copy()
    for feature in features:
        # 基于特征标准差的扰动
        std = df_perturbed[feature].std()
        perturbation = np.random.normal(0, epsilon * std, size=len(df_perturbed))
        df_perturbed[feature] += perturbation
    return df_perturbed

# 修改后的主计算函数，接受weights作为参数
def main(target, weights, df):
    # 添加数据收集列表
    bmi_values = []
    min_weeks_values = []
    
    def f(x, y):
        return weights['检测孕周_天数_3'] * x**3 + weights['检测孕周_天数_2'] * x**2 + weights['检测孕周_天数'] * x + \
               weights['孕妇BMI_3'] * y**3 + weights['孕妇BMI_2'] * y**2 + weights['孕妇BMI'] * y + \
               weights['Intercept'] - target

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

    # 使用原始数据的BMI范围
    y_values = np.arange(df['孕妇BMI'].min(), df['孕妇BMI'].max(), 0.1)
    for y in y_values:
        x_min = find_x_min(y)
        if x_min is not None:
            bmi_values.append(y)
            min_weeks_values.append(x_min / 7)

    # 数据排序与分段处理
    if not bmi_values:  # 处理空数据情况
        return {'target': target, 'division_points': [], 'segment_means': []}
        
    sorted_indices = np.argsort(bmi_values)
    sorted_bmi = np.array(bmi_values)[sorted_indices]
    sorted_weeks = np.array(min_weeks_values)[sorted_indices]
    
    # 计算总极差和每段目标极差
    global_min_weeks = min(min_weeks_values)
    global_max_weeks = max(min_weeks_values)
    total_range = global_max_weeks - global_min_weeks
    num_segments = 6
    target_segment_range = total_range / num_segments if total_range != 0 else 0
    
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
                'weeks': sorted_weeks[current_start:i+1]
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
    
    # 提取划分点和段平均值
    division_points = [seg['bmi'][-1] for seg in segments[:-1]] if len(segments) > 1 else []
    segment_means = [np.mean(seg['weeks']) for seg in segments] if segments else []
    
    return {
        'target': target,
        'division_points': division_points,
        'segment_means': segment_means
    }

# 计算结果的函数
def compute_results(model, df, targets):
    results = []
    for target in targets:
        result = main(target, model.params, df)
        results.append(result)
    return results

# 误差计算函数
def calculate_errors(original, perturbed):
    errors = []
    for orig, pert in zip(original, perturbed):
        # 确保目标值匹配
        if orig['target'] != pert['target']:
            continue
            
        # 计算划分点误差
        div_orig = np.array(orig['division_points'])
        div_pert = np.array(pert['division_points'])
        # 处理长度不一致的情况
        min_len = min(len(div_orig), len(div_pert))
        div_mse = np.mean((div_orig[:min_len] - div_pert[:min_len])**2) if min_len > 0 else np.nan
        div_mae = np.mean(np.abs(div_orig[:min_len] - div_pert[:min_len])) if min_len > 0 else np.nan
        
        # 计算段平均值误差
        seg_orig = np.array(orig['segment_means'])
        seg_pert = np.array(pert['segment_means'])
        min_seg_len = min(len(seg_orig), len(seg_pert))
        seg_mse = np.mean((seg_orig[:min_seg_len] - seg_pert[:min_seg_len])** 2) if min_seg_len > 0 else np.nan
        seg_mae = np.mean(np.abs(seg_orig[:min_seg_len] - seg_pert[:min_seg_len])) if min_seg_len > 0 else np.nan
        
        errors.append({
            'target': orig['target'],
            '划分点_MSE': div_mse,
            '划分点_MAE': div_mae,
            '段平均值_MSE': seg_mse,
            '段平均值_MAE': seg_mae
        })
    return pd.DataFrame(errors)

# 主程序执行
if __name__ == "__main__":
    # 加载原始数据
    df_original = pd.read_csv('问题3/贪心算法/关键信息提取结果(PCA).csv')
    print('原始数据基本信息：')
    df_original.info()
    
    # 训练原始模型并获取结果
    df_processed, model_original = prepare_data_and_train_model(df_original)
    targets = [0.04, 0.05, 0.06, 0.07]
    original_results = compute_results(model_original, df_processed, targets)
    
    # 保存原始结果
    os.makedirs('表格结果/问题3/贪心算法', exist_ok=True)
    pd.DataFrame(original_results).to_csv('表格结果/问题3/贪心算法/原始段划分结果汇总.csv', index=False, encoding='utf-8-sig')
    
    # 定义要添加扰动的特征
    features_to_perturb = ['检测孕周_天数', '孕妇BMI', '主成分1']
    perturbation_strength = 0.2  # 扰动强度：特征标准差的5%
    num_trials = 10  # 扰动实验次数
    
    # 多次扰动实验
    all_errors = []
    for trial in range(num_trials):
        print(f"\n进行第 {trial+1}/{num_trials} 次扰动实验...")
        
        # 添加随机扰动
        df_perturbed = add_random_perturbation(df_original, features_to_perturb, perturbation_strength)
        
        # 用扰动后的数据训练模型并计算结果
        df_perturbed_processed, model_perturbed = prepare_data_and_train_model(df_perturbed)
        perturbed_results = compute_results(model_perturbed, df_perturbed_processed, targets)
        
        # 计算本次实验的误差
        trial_errors = calculate_errors(original_results, perturbed_results)
        trial_errors['实验次数'] = trial + 1
        all_errors.append(trial_errors)
        
        print(f"第 {trial+1} 次实验误差计算完成")
    
    # 整合所有误差结果
    error_df = pd.concat(all_errors, ignore_index=True)
    mean_errors = error_df.groupby('target').mean().reset_index()  # 计算平均误差
    
    # 保存误差结果
    error_df.to_csv('表格结果/问题3/各次扰动误差.csv', index=False, encoding='utf-8-sig')
    mean_errors.to_csv('表格结果/问题3/平均扰动误差.csv', index=False, encoding='utf-8-sig')
    
    print("\n所有实验完成！")
    print(f"各次扰动误差已保存至: 表格结果/问题3/各次扰动误差.csv")
    print(f"平均扰动误差已保存至: 表格结果/问题3/平均扰动误差.csv")