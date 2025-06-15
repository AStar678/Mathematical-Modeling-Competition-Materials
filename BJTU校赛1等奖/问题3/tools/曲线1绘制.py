from process_space_curve import CurveProcessor

if __name__ == "__main__":
    data_path = "问题2/数据/问题2-曲线1.csv"
    image_save_path = '问题3/方法2/曲线1/image.jpg'
    points_save_path = "问题3/方法2/曲线1/points.csv"
    processor = CurveProcessor(data_path, image_save_path, points_save_path)
    processor.plot_curve()