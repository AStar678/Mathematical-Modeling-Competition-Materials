import pandas as pd
import os

# ---------------------- 配置参数 ----------------------
# 输入文件路径 - 新增女胎数据路径
INPUT_FILES = {
    '男': '数据预处理/男胎检测数据.csv',
    '女': '数据预处理/女胎检测数据.csv'
}
# 输出文件路径 - 新增女胎输出路径
OUTPUT_FILES = {
    '男': '数据预处理/男胎检测数据_去离群值.csv',
    '女': '数据预处理/女胎检测数据_去离群值.csv'
}
# 需要处理的指标列表
INDICATORS = [
    '年龄', '身高', '体重', '孕妇BMI', '原始读段数', 
    '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 'GC含量', 
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 
    'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度', 'X染色体浓度', 
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 
    '被过滤掉读段数的比例'
]
# 离群值判断阈值 (IQR倍数)
IQR_THRESHOLD = 1.5

# ---------------------- 核心函数 ----------------------
def remove_outliers_by_iqr(df, indicators, iqr_threshold=1.5):
    """使用IQR方法去除指定指标的离群值"""
    df_clean = df.copy()
    # 记录原始样本量
    original_count = len(df_clean)
    
    for indicator in indicators:
        if indicator not in df_clean.columns:
            print(f"警告：'{indicator}' 列不存在于数据中，已跳过")
            continue
        
        # 计算四分位数和IQR
        Q1 = df_clean[indicator].quantile(0.25)
        Q3 = df_clean[indicator].quantile(0.75)
        IQR = Q3 - Q1
        
        # 确定上下限
        lower_bound = Q1 - iqr_threshold * IQR
        upper_bound = Q3 + iqr_threshold * IQR
        
        # 过滤离群值
        df_clean = df_clean[(df_clean[indicator] >= lower_bound) & (df_clean[indicator] <= upper_bound)]
        
    # 计算过滤后的样本量和保留比例
    cleaned_count = len(df_clean)
    retention_rate = cleaned_count / original_count * 100
    print(f"数据清洗完成：原始样本量={original_count}, 清洗后样本量={cleaned_count}, 保留比例={retention_rate:.2f}%")
    
    return df_clean

# ---------------------- 主流程 ----------------------
if __name__ == '__main__':
    # 存储清洗前后的样本量用于汇总
    summary = {}
    
    # 循环处理男胎和女胎数据
    for gender in ['男', '女']:
        input_path = INPUT_FILES[gender]
        output_path = OUTPUT_FILES[gender]
        
        # 读取原始数据
        if not os.path.exists(input_path):
            print(f"错误：{gender}胎文件 '{input_path}' 不存在")
            continue
        
        df = pd.read_csv(input_path)
        original_count = len(df)
        summary[gender] = {'original': original_count}
        print(f"成功读取{gender}胎检测数据：{original_count} 条记录")
        
        # 去除离群值
        df_cleaned = remove_outliers_by_iqr(df, INDICATORS, IQR_THRESHOLD)
        
        # 保存清洗后的数据
        df_cleaned.to_csv(output_path, index=False)
        cleaned_count = len(df_cleaned)
        summary[gender]['cleaned'] = cleaned_count
        summary[gender]['retention'] = cleaned_count / original_count * 100
        
        # 验证保存结果
        if os.path.exists(output_path):
            print(f"{gender}胎清洗后的数据已保存至：{output_path} (大小：{os.path.getsize(output_path)} 字节)")
        else:
            print(f"错误：{gender}胎文件保存失败")
    
    # 输出汇总统计
    print("\n===== 数据清洗汇总 ====")
    for gender, stats in summary.items():
        print(f"{gender}胎数据：")
        print(f"  原始样本量：{stats['original']}")
        print(f"  清洗后样本量：{stats['cleaned']}")
        print(f"  保留比例：{stats['retention']:.2f}%\n")