% 数据
topK = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
DMCB_matching = [0.37, 3.61, 3.38, 2.98, 3.42, 3.93, 3.84, 3.64, 3.52, 3.42];
DMCB_conformity = [7.62, 8.08, 7.87, 7.37, 7.28, 6.97, 6.78, 6.52, 6.24, 6.05];
CausE = [6.64, 6.02, 5.96, 5.68, 5.73, 5.64, 5.74, 5.61, 5.5, 5.37];
DICE = [7.49, 5.67, 6.68, 6.69, 6.54, 6.42, 6.12, 5.99, 5.86, 5.7];
DCCL = [6.26, 5.46, 5.5, 5.65, 5.52, 5.25, 5.29, 5.2, 4.98, 5.16];
MF_IPS = [5.21, 5.67, 5.73, 5.7, 5.46, 5.54, 5.48, 5.43, 5.41, 5.43];
MACR = [4.16, 4.6, 4.79, 4.56, 4.69, 4.89, 4.95, 4.93, 5.08, 4.99];

% 创建图形
figure('Units', 'inches', 'Position', [0, 0, 8, 6]);

% 设置线条宽度和字体大小
lineWidth = 3;
fontSize = 20;
markerSize = 10;

% 绘制折线图
hold on;
plot(topK, DMCB_matching, '-o', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
plot(topK, DMCB_conformity, '-s', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
plot(topK, CausE, '-^', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
plot(topK, DICE, '-d', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
plot(topK, DCCL, '-v', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
plot(topK, MF_IPS, '-p', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
plot(topK, MACR, '-h', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
hold off;

% 设置坐标轴
xlabel('TopK', 'FontSize', fontSize);
ylabel('Value', 'FontSize', fontSize);

% 设置刻度字体大小
set(gca, 'FontSize', fontSize);

% 添加图例
legend('PCDR-interest', 'PCDR-conformity', 'CausE', 'DICE', 'DCCL', 'MF_IPS', 'MACR', 'Location', 'best', 'FontSize', fontSize-4);

% 设置网格
grid on;
