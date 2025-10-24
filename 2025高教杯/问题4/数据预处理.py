import pandas as pd
import re

#删除
drop = ["序号","孕妇代码","末次月经","检测日期","Unnamed: 20","Unnamed: 21","胎儿是否健康"]
data = pd.read_csv('数据预处理/女胎检测数据.csv')
data.drop(drop,axis=1,inplace=True)

df = data

if 'IVF妊娠' in df.columns:

    df.loc[:, 'IVF妊娠'] = pd.Categorical(df['IVF妊娠'], 
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
    df.drop(columns=['检测孕周'], inplace=True)

else:
    raise ValueError("未找到'检测孕周'列，请检查数据文件")

if '怀孕次数' in df.columns:
    df['怀孕次数'] = df['怀孕次数'].apply(lambda x: int(3) if x == "≥3" else int(x))
else:
    raise ValueError("未找到'怀孕次数'列，请检查数据文件")



if '染色体的非整倍体' in df.columns:
    def encode_chromosome_status(x):
        if "13" in str(x) or "18" in str(x) or "21" in str(x):
            return 1
        else:
            return 0

    df['染色体的非整倍体'] = df['染色体的非整倍体'].apply(encode_chromosome_status)
else:
    raise ValueError("未找到'染色体的非整倍体'列，请检查数据文件")

# 将“染色体的非整倍体”列移动到最后
cols = df.columns.tolist()
cols.remove('染色体的非整倍体')
cols.append('染色体的非整倍体')
df = df[cols]
df.to_csv('问题4/女胎检测数据_处理后.csv', index=False)
