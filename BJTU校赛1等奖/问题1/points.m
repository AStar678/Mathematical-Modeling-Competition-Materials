clc; clear; close all;

data = xlsread('C:\Users\Lenovo\Desktop\工作簿1.xlsx');
s_original = data(:,2);   % 原始弧长数据
kappa_original = data(:,3); % 原始曲率数据

% 生成更密集的弧长点用于插值
s_interp = linspace(min(s_original), max(s_original), 1000); % 1000个插值点

% 进行三次样条插值
kappa_interp = spline(s_original, kappa_original, s_interp);


figure('Position', [100, 100, 800, 600], 'Name', '曲率-弧长关系图');

% 绘制原始数据点
plot(s_original, kappa_original, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5, ...
    'DisplayName', '原始检测点');
hold on;

% 绘制插值曲线
plot(s_interp, kappa_interp, 'b-', 'LineWidth', 2, ...
    'DisplayName', '三次样条插值');

% 图形美化
grid on;
xlabel('弧长 s (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('曲率 \kappa (1/mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('基于三次样条插值的曲率-弧长函数关系', 'FontSize', 14, 'FontWeight', 'bold');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 11);

% 添加参考线
xlim([min(s_original)-0.1, max(s_original)+0.1]);
ylim([min(kappa_original)-0.1, max(kappa_original)+0.1]);

% 突出显示插值效果
text(0.05, 0.95, ['插值点数: ', num2str(length(s_interp))], ...
    'Units', 'normalized', 'FontSize', 10, 'Color', [0.2 0.2 0.2]);