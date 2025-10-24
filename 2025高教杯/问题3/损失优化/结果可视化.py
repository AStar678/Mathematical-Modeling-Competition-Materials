import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def main(rate):
    # 读取文件
    df = pd.read_csv(f'表格结果/问题3/损失函数优化结果/损失优化模型结果_{rate}_反归一化_含孕妇BMI.csv')

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

    # 获取孕妇BMI_反归一化的前5个值，并保留一位小数
    bmi_values = df['孕妇BMI_反归一化'][:5].round(1).tolist()

    # 获取检测孕周_反归一化的值除以7，并保留一位小数
    week_values = (df['检测孕周_反归一化'] / 7).round(1).tolist()

    # 延长一点作为第6个部分
    extension_value = 50.0
    bmi_values.append(extension_value)

    # 创建一个新的图形
    plt.figure(figsize=(12, 3), facecolor='#f4f4f4')

    # 绘制数轴
    ax = plt.axes()
    ax.set_facecolor('#f4f4f4')
    # 截断数轴，设置 x 轴范围为 20 到最大值
    plt.axis([20, max(bmi_values), 0, 1])
    plt.yticks([])
    plt.xlim(20, max(bmi_values))

    # 设置坐标轴颜色和样式
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['lightblue', 'darkblue'], N=len(bmi_values))
    # 填充每一段不同颜色并标注数值
    for i in range(len(bmi_values)):
        if i == 0:
            left = 0
            right = bmi_values[i]
        else:
            left = bmi_values[i - 1]
            right = bmi_values[i]

        # 如果当前段在截断范围内，跳过
        if right < 20:
            continue

        color = cmap(i / (len(bmi_values) - 1))
        if left < 20:
            left = 20
        plt.fill_between([left, right], 0, 1, color=color, alpha=0.5)

        if i < len(week_values):
            x_text = (left + right) / 2
            y_text = 0.3
            # 调大字体
            plt.text(x_text, y_text, str(week_values[i]), ha='center', va='center', fontsize=20, color='white')

    # 标注孕妇BMI_反归一化的值
    for i, value in enumerate(bmi_values):
        if value < 20 or value == 50:  # 不显示 50 和 20 以下的值
            continue
        plt.axvline(x=value, color='gray', linestyle='--', linewidth=1)
        # 调大字体 - 将fontsize从30调整为更大的值，如40或50
        plt.text(value, 0.7, str(value), ha='center', va='center', fontsize=20, color='blue')

    # 显示图形
    plt.savefig(f'可视化结果/问题3/损失优化模型结果_{rate}.png')

if __name__ == '__main__':
    main(0.1)
    main(1)
    main(10)

