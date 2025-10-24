import pandas as pd
import re

# 读取清洗后的数据
cleaned_data_path = '数据预处理/男胎检测数据.csv'
df = pd.read_csv(cleaned_data_path)

# 1. IVF妊娠值转换: 自然受孕=1, 其余=0
# 假设原始列名为'妊娠方式', 请根据实际列名调整
# 1. 受孕方式编码: 自然受孕/IUI/IVF 独热编码
if 'IVF妊娠' in df.columns:
    # 先将列转换为Categorical类型并指定类别
    df['IVF妊娠'] = pd.Categorical(df['IVF妊娠'], 
                                 categories=['自然受孕', 'IUI', 'IVF'])
    
    # 使用独热编码处理三种受孕方式
    df_encoded = pd.get_dummies(df['IVF妊娠'], 
                               prefix='受孕方式', 
                               drop_first=False)

    df_encoded = df_encoded.astype(int)
    
    # 将编码结果合并到原数据
    df = pd.concat([df, df_encoded], axis=1)
    
    # 删除原始列
    df.drop(columns=['IVF妊娠'], inplace=True)
    
    # 确保所有类别列都存在
    required_columns = ['受孕方式_自然受孕', '受孕方式_IUI', '受孕方式_IVF']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
else:
    raise ValueError("未找到'IVF妊娠'列，请检查数据文件")

# 2. 检测孕周转换: aw+b格式转为7*a + b
# 假设原始列名为'检测孕周', 请根据实际列名调整
if '检测孕周' in df.columns:
    def convert_gestational_age(age_str):
        # 处理特殊情况，如'1w'直接转换为7天
        if 'w+' in str(age_str):
            a,b = str(age_str).split('w+')
            return 7 * int(a) + int(b)
        else:
            age_str = str(age_str)[:2]
            return int(age_str) * 7

    df['检测孕周_天数'] = df['检测孕周'].apply(convert_gestational_age)

else:
    raise ValueError("未找到'检测孕周'列，请检查数据文件")

df.drop(columns=['检测孕周'], inplace=True)
df.drop(columns=['末次月经'], inplace=True)
df.drop(columns=['检测日期'], inplace=True)

# 3. 怀孕次数处理: >=3的全部取3
# 假设原始列名为'怀孕次数', 请根据实际列名调整
if '怀孕次数' in df.columns:
    df['怀孕次数'] = df['怀孕次数'].apply(lambda x: int(3) if x == "≥3" else int(x))
else:
    raise ValueError("未找到'怀孕次数'列，请检查数据文件")

# 4. 胎儿健康状况编码: 健康=1, 不健康=0
# 假设原始列名为'胎儿健康', 请根据实际列名调整
if '胎儿是否健康' in df.columns:
    # 处理可能的字符串或分类值
    df['胎儿是否健康'] = df['胎儿是否健康'].apply(lambda x: 1 if str(x).strip() in ['是'] else 0)
else:
    raise ValueError("未找到'胎儿是否健康'列，请检查数据文件")

if '染色体的非整倍体' in df.columns:
    def encode_chromosome_status(x):
        if "13" in str(x) or "18" in str(x) or "21" in str(x):
            return 1
        else:
            return 0

    df['染色体的非整倍体'] = df['染色体的非整倍体'].apply(encode_chromosome_status)
else:
    raise ValueError("未找到'染色体的非整倍体'列，请检查数据文件")

# 调整胎儿健康编码列至最后
if '胎儿是否健康' in df.columns:
    # 获取当前列列表并移动目标列到末尾
    cols = df.columns.tolist()
    cols.remove('胎儿是否健康')
    cols.append('胎儿是否健康')
    df = df[cols]
else:
    raise ValueError("未找到'胎儿是否健康'列，请检查健康状况编码步骤")

# 保存特征工程后的数据
output_path = '数据预处理/男胎特征工程后数据.csv'
df.to_csv(output_path, index=False)
print(f'特征工程完成，数据已保存至: {output_path}')

