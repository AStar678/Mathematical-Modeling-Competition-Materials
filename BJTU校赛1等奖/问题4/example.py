from curve_processing import CurveProcessor

data_path = "/Users/aoxiang/Desktop/提交代码/数据/问题2-曲线2.csv"
image_save_path = "/Users/aoxiang/Desktop/提交代码/问题4/测试图像.png"
points_save_path = "/Users/aoxiang/Desktop/提交代码/问题4/测试坐标.csv"

processor = CurveProcessor(data_path, image_save_path, points_save_path)
processor.plot_curve()