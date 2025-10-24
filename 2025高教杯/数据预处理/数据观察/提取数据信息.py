import pandas as pd
import os

data = pd.read_csv('数据预处理/男胎检测数据.csv')
data.shape

describe = data.describe()

if not os.path.exists('表格结果/数据预处理'):
    os.makedirs('表格结果/数据预处理')

describe.to_csv('表格结果/数据预处理/describe.csv')
