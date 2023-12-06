% 数据
model_names = {'MF', 'BPR', 'MACR', 'DICE', 'PCDR'};
HR20_popular = [31.4, 34.88, 31.4, 25.58, 32.56];
HR20_unpopular = [10.23, 10.48, 22.8, 20.07, 29.1];

% 设置画布大小和字体大小
figure('Units', 'inches', 'Position', [0, 0, 5, 4]);
fontSize = 16;

% 绘制柱状图
bar_data = [HR20_popular; HR20_unpopular]';
bar_handle = bar(bar_data, 'FaceColor', 'flat');

% 设置颜色（学术风格）
bar_handle(1).CData = [0 0.4470 0.7410];
bar_handle(2).CData = [0.8500 0.3250 0.0980];

% 添加标签和标题
set(gca, 'XTickLabel', model_names, 'FontSize', fontSize);
ylabel('HR@20', 'FontSize', fontSize);
%xlabel('Model', 'FontSize', fontSize);
legend({'Popular items', 'Unpopular items'}, 'FontSize', fontSize);

% 展示
grid on;
