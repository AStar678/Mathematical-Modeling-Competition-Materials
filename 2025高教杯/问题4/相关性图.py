import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
font = FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'], size=10)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载数据
data_path = "/Users/aoxiang/Desktop/数模国赛/C题/问题4/女胎检测数据_处理后.csv"
df = pd.read_csv(data_path)

# 计算相关系数矩阵
correlation_matrix = df.corr()

# 设置画布大小
plt.figure(figsize=(15, 12))

# 绘制热力图
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,  # 显示相关系数数值
    fmt=".2f",  # 数值保留两位小数
    cmap="coolwarm",  # 使用冷暖色调
    square=True,  # 正方形单元格
    linewidths=.5,  # 单元格边框宽度
    cbar_kws={"shrink": .8},  # 颜色条缩放比例
    annot_kws={"fontproperties": font, "size": 8}  # 数值标签字体设置
)

# 设置标题和坐标轴标签字体
plt.title("特征相关性热力图", fontproperties=FontProperties(family=['SimHei', 'WenQuanYi Micro Hei'], size=14, weight='bold'))
plt.xlabel("特征", fontproperties=font)
plt.ylabel("特征", fontproperties=font)

# 旋转x轴标签避免重叠
plt.xticks(rotation=45, ha='right', fontproperties=font)
plt.yticks(fontproperties=font)

# 调整布局并保存
plt.tight_layout()
save_path = "/Users/aoxiang/Desktop/数模国赛/C题/可视化结果/问题4/特征相关性热力图.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"相关性热力图已保存至: {save_path}")