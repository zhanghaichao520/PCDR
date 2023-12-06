% Data for Table 1, 2, 3
alpha = [3, 5, 6, 7, 8.5, 9, 9.5, 10, 10.5, 10.8, 11, 13, 15, 17, 20];
hit20_alpha = [0.1339, 0.1343, 0.1336, 0.1441, 0.1489, 0.1574, 0.17, 0.176, 0.1672, 0.1176, 0.1181, 0.1234, 0.1311, 0.1387, 0.1191];
beta = [0, 0.095, 0.099, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5];
hit20_beta = [0.2995, 0.3141, 0.3098, 0.2841, 0.2609, 0.275, 0.2218, 0.2159, 0.2331, 0.2316, 0.2475, 0.2594, 0.2627, 0.2394];
delta = [0, 10, 18, 25, 35, 38, 41, 48, 55, 61, 65, 72, 78, 87, 100];
hit20_delta = [0.1865, 0.1396, 0.2124, 0.2853, 0.2372, 0.2139, 0.2532, 0.2336, 0.2332, 0.2465, 0.2118, 0.2045, 0.2261, 0.1875, 0.1607];

% Create figure and set attributes
figure('Units', 'inches', 'Position', [0, 0, 36, 8]);  % Increased width for more spacing
lineWidth = 2;
fontSize = 16;
lineColor = 'r';

% Plot the first graph
subplot(1,3,1);
plot(alpha, hit20_alpha, '-o', 'LineWidth', lineWidth, 'Color', lineColor);
alpha = char(945)
title(alpha, 'FontSize', fontSize);
ylabel('Hit@20', 'FontSize', fontSize);
grid on;
set(gca, 'FontSize', fontSize);

% Plot the second graph
subplot(1,3,2);
plot(beta, hit20_beta, '-s', 'LineWidth', lineWidth, 'Color', lineColor);
title('Similar Loss Param', 'FontSize', fontSize);
ylabel('Hit@20', 'FontSize', fontSize);  % Added back
grid on;
set(gca, 'FontSize', fontSize);

% Plot the third graph
subplot(1,3,3);
plot(delta, hit20_delta, '-^', 'LineWidth', lineWidth, 'Color', lineColor);
title('Domain Loss Param', 'FontSize', fontSize);
ylabel('Hit@20', 'FontSize', fontSize);  % Added back
grid on;
set(gca, 'FontSize', fontSize);
