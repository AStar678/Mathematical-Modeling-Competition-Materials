import pandas as pd
from scipy.stats import f
import os


def main(alpha):
    # 加载模型优化结果数据
    model_result = pd.read_csv(f'问题3/损失优化/损失优化结果/损失优化模型结果_{alpha}.csv')

    # 加载男胎检测数据
    male_fetus_data = pd.read_csv('数据预处理/男胎特征工程后数据.csv')

    print('模型优化结果数据基本信息：')
    model_result.info()

    # 查看模型优化结果数据集行数和列数
    model_result_rows, model_result_columns = model_result.shape

    # 获取男胎检测数据中孕妇BMI的最小值和最大值
    min_bmi = male_fetus_data['孕妇BMI'].min()
    max_bmi = male_fetus_data['孕妇BMI'].max()

    min_day = male_fetus_data['检测孕周_天数'].min()
    max_day = male_fetus_data['检测孕周_天数'].max()

    # 对模型优化结果中的孕妇BMI和检测孕周进行反归一化
    model_result['孕妇BMI_反归一化'] = model_result['孕妇BMI'] * (max_bmi - min_bmi) + min_bmi
    model_result['检测孕周_反归一化'] = model_result['检测孕周'] * (max_day - min_day) + min_day


    # 将结果保存为csv文件
    csv_path = f'表格结果/问题3/损失函数优化结果/损失优化模型结果_{alpha}_反归一化_含孕妇BMI.csv'
    model_result.to_csv(csv_path)

if __name__ == '__main__':
    os.makedirs('表格结果/问题3/损失函数优化结果', exist_ok=True)
    alpha = [0.1,1,10]
    for a in alpha:
        main(a)
