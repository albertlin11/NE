clc;
close all;
y = readtable('prediction/GA_logid_hetero_534_100.csv');
% y = readtable('prediction/MLP_logid_hetero_9-4_100.csv');
true = 4 ;
pred = 5;

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 30nm-1000nm')
for j = 2:7
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');

    
    xlim([0 1.5]), ylim([0 0.003]); 
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end
figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 30nm-1500nm')
for j = 11:16
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.005]);
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 40nm-400nm')
for j = 20:25
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.0005]);
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end


figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 40nm-1500nm')
for j = 29:34
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.003]);
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 50nm-1000nm')
for j = 38:43
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.0015]);

    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 50nm-3000nm')
for j = 47:52
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.004]);
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 60nm-400nm')
for j = 56:61
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.0006]);
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 60nm-1000nm')
for j =65:70
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.001]);

    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end

figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 300nm-3000nm')
for j = 74:79
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.001]);
    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end
figure('units','centimeter','position',[2, 2, 10, 7.5])
colororder({'k','k'})
title('L-W : 500nm-1000nm')
for j = 83:88
    yyaxis left
    h = plot((0:0.025:1.5),abs(y{(61*j-60):(61*j),true}),'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h2 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    hold on
    set(gca, 'YScale', 'log')
    xlabel('\bfGate voltage(V)', 'FontSize',12), ylabel('\bfDrain current (A)', 'FontSize',12);
    yyaxis right
    h3 = plot((0:0.025:1.5),y{(61*j-60):(61*j),true},'o','Markersize', 4, 'MarkerEdgeColor', 'b','LineWidth', 0.8);
    hold on
    h4 = plot((0:0.025:1.5),y{(61*j-60):(61*j),pred},'-','LineWidth', 2,'color','k');
    xlim([0 1.5]), ylim([0 0.00025]);

    legend({'\bf Measured ','\bf Predict'},'FontSize',8, 'Location','northwest','Box','off')
    set(gca,'FontWeight','bold','FontSize', 12, 'LineWidth', 2);
   
end