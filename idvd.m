clc;
close all;
% y = readtable('prediction/MLP_logid_hetero_14-4_100.csv');
y = readtable('prediction/GA_logid_hetero_5610_100.csv');
true =4 ;
pred = 5;
figure('units','centimeter','position',[2, 2, 10, 7.5]);
 
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j-7):61:(8*j+541),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j-7):61:(8*j+541),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 2.5]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 30nm-1000nm')


figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+542):61:(8*j+1090),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+542):61:(8*j+1090),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]),ylim([0 4]);

    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');
end
title('L-W : 30nm-1500nm')


figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+1091):61:(8*j+1639),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+1091):61:(8*j+1639),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.5]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');
end
title('L-W : 40nm-400nm')



figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+1640):61:(8*j+2188),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+1640):61:(8*j+2188),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]),ylim([0 2]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 40nm-1500nm')



figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+2189):61:(8*j+2737),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+2189):61:(8*j+2737),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 1]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 50nm-1000nm')



figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+2738):61:(8*j+3286),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+2738):61:(8*j+3286),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 2.5]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 50nm-3000nm')



figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+3287):61:(8*j+3835),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+3287):61:(8*j+3835),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]),  ylim([0 0.4]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 60nm-400nm')



figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+3836):61:(8*j+4384),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+3836):61:(8*j+4384),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]),ylim([0 0.8]);

    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 60nm-1000nm')


figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+4385):61:(8*j+4933),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+4385):61:(8*j+4933),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.6]);

    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 300nm-3000nm')


figure('units','centimeter','position',[2, 2, 10, 7.5])
for j =1:7
    h = plot((0:0.2:1.6),1000*y{(8*j+4934):61:(8*j+5482),4},'b-o','Markersize', 5, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.2:1.6),1000*y{(8*j+4934):61:(8*j+5482),5},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.15]);
    xlabel('\bfDrain voltage(V)', 'FontSize',12), ylabel('\bfDrain current (mA)', 'FontSize',12);
    legend({'\bf Measured ','\bf Predict'},'FontSize',12, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2,'color','w');

end
title('L-W : 500nm-1000nm')

