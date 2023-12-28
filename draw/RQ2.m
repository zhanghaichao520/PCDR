% Data for ml-1m
ml_1m_range = {'1-20', '21-40', '41-60', '61-100', '101-200'};
ml_1m_percentage = [21.06, 22.08, 14.24, 17.09, 17.29];
ml_1m_HR20M = [19.02, 16.89, 18.35, 16.49, 14.68];
ml_1m_HR20C = [5.36, 2.57, 3.36, 6.53, 5.49];
ml_1m_HR20MC = [14.26, 8.44, 12.53, 16.49, 15.87];

% Data for amazon
amazon_range = {'0-1', '1-2', '2-3', '3-5', '5-10'};
amazon_percentage = [60.69, 23.32, 7.22, 6.30, 1.98];
amazon_HR20M = [1.92, 2.82, 2.98, 3.64, 2.57];
amazon_HR20C = [0.49, 0.91, 1.54, 2.12, 2.57];
amazon_HR20MC = [1.06, 2.20, 2.79, 3.34, 1.80];

% 创建子图
figure('Units', 'inches', 'Position', [0, 0, 8, 8]);

% 设置字体和线条大小
lineWidth = 3;
fontSizeLarge = 18;
fontSizeSmall = 16;

% 存储句柄
handles = zeros(1, 3);

subplot(1,2,1);  % 第一个子图
h1 = plot(1:5, ml_1m_HR20M, '-o', 1:5, ml_1m_HR20C, '-s', 1:5, ml_1m_HR20MC, '-^', 'LineWidth', lineWidth);
handles(1:3) = h1;
title('ml-1m', 'FontSize', fontSizeLarge);
xlabel('Interactions Range', 'FontSize', fontSizeSmall);
ylabel('HR@20', 'FontSize', fontSizeSmall);
set(gca, 'XTick', 1:5, 'XTickLabel', ml_1m_range, 'FontSize', fontSizeSmall);
yyaxis right;
bar(1:5, ml_1m_percentage, 0.2, 'FaceColor', [0.2 0.2 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.5);  % ml-1m 子图
ylabel('Percentage of Data (%)', 'FontSize', fontSizeSmall);

subplot(1,2,2);  % 第二个子图
plot(1:5, amazon_HR20M, '-o', 1:5, amazon_HR20C, '-s', 1:5, amazon_HR20MC, '-^', 'LineWidth', lineWidth);
title('amazon', 'FontSize', fontSizeLarge);
xlabel('Interactions Range', 'FontSize', fontSizeSmall);
ylabel('HR@20', 'FontSize', fontSizeSmall);
set(gca, 'XTick', 1:5, 'XTickLabel', amazon_range, 'FontSize', fontSizeSmall);
yyaxis right;
bar(1:5, amazon_percentage, 0.2, 'FaceColor', [0.2 0.2 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.5);  % amazon 子图
ylabel('Percentage of Data (%)', 'FontSize', fontSizeSmall);

% 创建全局图例
figureLegend = legend(handles(1:3), 'HR@20 Int', 'HR@20 Conf', 'HR@20 Int+Conf', 'FontSize', fontSizeSmall);
set(figureLegend, 'Position', [0.4, 0.01, 0, 0], 'Units', 'normalized');