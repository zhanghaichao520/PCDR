% 数据
model_names = {'MF', 'LightGCN', 'MACR', 'DICE', 'PCDR'};
HR20_conservatives = [29.83, 29.14, 31.31, 31.81, 36.47]; 
HR20_radicals      = [19.95, 19.85, 24.81, 24.79, 33.66]; 

% 定义柔和的颜色
softBlue = [0.3, 0.6, 0.9];
softRed = [0.9, 0.4, 0.4];
softGreen = [0.4, 0.8, 0.4];

% 设置画布大小和字体大小
figure('Units', 'inches', 'Position', [0, 0, 5, 4]);
fontSize = 16;

% 绘制柱状图
bar_data = [HR20_conservatives; HR20_radicals]';
bar_handle = bar(bar_data, 'FaceColor', 'flat');

% 设置颜色（学术风格）
bar_handle(1).CData = [0.3, 0.6, 0.9];
bar_handle(2).CData = [0.9, 0.4, 0.4];

% 添加标签和标题
set(gca, 'XTickLabel', model_names, 'FontSize', fontSize);
ylabel('HR@20 (%)', 'FontSize', fontSize);
%xlabel('Model', 'FontSize', fontSize);
legend({'conservatives', 'radicals'}, 'FontSize', fontSize);

% 展示
grid on;

model_names = {'MF', 'LightGCN', 'MACR', 'DICE', 'PCDR'};

% group1
% HR20_conservatives = [31.43, 22.72, 22.72, 23.30, 23.86]; 
% HR20_radicals      = [24.96, 15.88, 16.27, 16.45, 33.87]; 
% 
% 
% % group2
% HR20_conservatives = [29.83, 29.14, 31.31, 31.81, 36.47]; 
% HR20_radicals      = [19.95, 19.85, 24.81, 24.79, 33.66]; 
