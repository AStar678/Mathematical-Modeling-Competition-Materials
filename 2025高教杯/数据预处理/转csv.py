import pandas as pd
import os

# 读取Excel文件中的所有工作表
excel_file = pd.ExcelFile('附件.xlsx')
sheets = excel_file.sheet_names

output_dir = '数据预处理'
os.makedirs(output_dir, exist_ok=True)

# 遍历每个工作表并保存为CSV
for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    csv_path = os.path.join(output_dir, f'{sheet}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"成功将{len(sheets)}个工作表转换为CSV文件，保存在{output_dir}目录下")