import pandas as pd

# 读取原始数据
data_path = '数据预处理/男胎检测数据.csv'
df = pd.read_csv(data_path)

# 显示清洗前数据量
print(f'清洗前数据量: {len(df)} 条')

# 过滤GC含量在0.4-0.6之间的数据
df_cleaned = df.copy()


if '检测孕周' in df_cleaned.columns:
    def convert_gestational_age(age_str):
        # 处理特殊情况，如'1w'直接转换为7天
        if 'w+' in str(age_str):
            a,b = str(age_str).split('w+')
            return 7 * int(a) + int(b)
        else:
            age_str = str(age_str)[:2]
            return int(age_str) * 7

    df_cleaned['检测孕周_天数'] = df_cleaned['检测孕周'].apply(convert_gestational_age)

else:
    raise ValueError("未找到'检测孕周'列，请检查数据文件")

print(f'清洗后数据量: {len(df_cleaned)} 条')



# 保存清洗后的数据（添加_cleaned后缀）
output_path = '/Users/aoxiang/Desktop/数模国赛/C题/问题2/男胎检测数据处理.csv'
df_cleaned.to_csv(output_path, index=False)
print(f'清洗后数据已保存至: {output_path}')

