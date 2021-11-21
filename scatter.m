clc;
close all;


y = readtable('./prediction/MLP_logid_hetero_14-4_10.csv');
% y = readtable('prediction/GA_logid_hetero_534_100.csv');
figure('units','centimeter','position',[2, 2, 10, 7.5])
true=2;
pred=3;

plot([-12 0],[-12 0],'-','LineWidth', 3,'color','k')
xlim([-12 0]), ylim([-12 0]),xticks([-12:4:0]), yticks([-12:4:0])
hold on
h2 = plot(y{2:598,true},y{2:598,pred},'o','Markersize', 4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
set(gca,'FontWeight','bold','FontSize', 14, 'LineWidth', 2);
xlabel('\bfTarget', 'FontSize',16), ylabel('\bfPrediction', 'FontSize',16);
legend({'\bfIdeal','\bfPredict'},'FontSize',16, 'Location','northwest','Box','off');
txt = ['\bfR^2: 0.9970']
text(-6.5,-10,txt,'FontSize',16);
% text(0.0018,0.0005,txt,'FontSize',16);
% txt2 = ['\bfRMSE:8.74\times10^{-5}'];
txt2 = ['\bfRMSE: 0.1172'];
text(-6.5,-8.5,txt2,'FontSize',16);
% title('MLP 715');

% text(0.0018,0.0011,txt2,'FontSize',16);