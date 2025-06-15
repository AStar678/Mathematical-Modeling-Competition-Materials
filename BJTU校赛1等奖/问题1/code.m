% 输入数据
s = [0, 24.9804, 49.9682, 74.936, 99.9164, 125.1002]; % 弧长
kappa = [0.000723, 0.00208, 1.998801, 0.00208, 0.000723, 0.00039]; % 曲率

% 插值曲率函数（平滑过渡）
s_interp = linspace(min(s), max(s), 1000); % 高精度插值
kappa_interp = pchip(s, kappa, s_interp); % 分段三次插值

% 计算切线角 theta(s) = integral(kappa)
theta = cumtrapz(s_interp, kappa_interp); % 数值积分

% 计算坐标 (x, y)
dx = cos(theta);
dy = sin(theta);
x = cumtrapz(s_interp, dx); % x(s) = integral(cos(theta))
y = cumtrapz(s_interp, dy); % y(s) = integral(sin(theta))

% 绘制曲线形状
figure;
plot(x, y, 'b-', 'LineWidth', 2); % 曲线
hold on;
plot(x(1), y(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
plot(x(end), y(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 终点
grid on;
axis equal; % 保持纵横比一致

% 标注关键点
text(x(1), y(1), ' 起点 (s=0)', 'VerticalAlignment', 'bottom');
text(x(end), y(end), ' 终点 (s=125.1)', 'VerticalAlignment', 'top');
title('重建的曲线形状（基于曲率数据）', 'FontSize', 14);
xlabel('x 坐标', 'FontSize', 12);
ylabel('y 坐标', 'FontSize', 12);
set(gcf, 'Color', 'w');