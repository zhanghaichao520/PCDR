% 数据
interactionsRange = {'1-80', '81-160', '161-240', '241-320', '321-900'};
percentageData = [34.71, 27.79, 16.01, 10.22, 11.01];
HR20_wo_Lsim = [36.89, 65.23, 69.64, 81.76, 94.55];
HR20_wo_Lcausal = [78.67, 66.36, 83.94, 96.36, 95.45];
HR20_wo_Ldomain = [70.30, 87.58, 88.48, 84.93, 95.89];
HR20 = [87.58, 89.09, 88.18, 95.89, 99.09];

% 定义柔和的颜色
softBlue = [0.3, 0.6, 0.9];
softRed = [0.9, 0.4, 0.4];
softGreen = [0.4, 0.8, 0.4];

% 创建图
figure('Units', 'inches', 'Position', [0, 0, 10, 8]);
fontSize = 18;
lineWidth = 2.5;

% 折线图
hold on;
plot(1:5, HR20_wo_Lcausal, '-s', 'LineWidth', lineWidth,'Color',softRed);
plot(1:5, HR20_wo_Lsim, '-o', 'LineWidth', lineWidth,'Color',softGreen);
plot(1:5, HR20_wo_Ldomain, '-^', 'LineWidth', lineWidth,'Color',softBlue);
plot(1:5, HR20, '-d', 'LineWidth', lineWidth);
hold off;

% 标题和轴标签
title('HR@20 and Data Percentage by Interactions Range', 'FontSize', fontSize);
xlabel('Number of interactions Range', 'FontSize', fontSize);
ylabel('HR@20', 'FontSize', fontSize);
set(gca, 'XTick', 1:5, 'XTickLabel', interactionsRange, 'FontSize', fontSize);

% 柱状图
yyaxis right;
bar(1:5, percentageData, 0.2, 'FaceColor', [0.2 0.2 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
ylabel('Percentage of Data (%)', 'FontSize', fontSize);

% 图例
legend('HR@20 \alpha=0', 'HR@20 \beta=0',  'HR@20 \delta=0', 'HR@20', 'FontSize', fontSize, 'Location', 'northwest');
