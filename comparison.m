clear
close all

folder = "./figures/HierLTI/2022-11-24/";


load('data/csv/2022-10-05_oscil100/variables.mat');
sys_d = c2d(sys, Ts);

% order_max = length(sys.Denominator{1})-1;
order_max = 8;

M = csvread('data/csv/2022-10-05_oscil100/inoutdata_multimodes.csv');
u = M(:, 1);
y = M(:, 2);

data = iddata(y, u, Ts);
data_e = data(1:9000);
data_v = data(9000:end);

date_list = ["2022-11-23-4_4_oscil100_3000_losstime10", "2022-10-11-4_4_oscil100_3000_losstime10_modalstart", "/2022-11-23-4_4_oscil100_3000_losstime10_modalstart"];
num_layer_list = [4, 4, 4];

for i=1:length(date_list)
% for i=1:length(date_list)
    A = csvread("figures/HierLTI/" + date_list(i) + "/A.csv");
    B = csvread("figures/HierLTI/" + date_list(i) + "/B.csv");
    C = csvread("figures/HierLTI/" + date_list(i) + "/C.csv");
    D = csvread("figures/HierLTI/" + date_list(i) + "/D.csv");
    
    for layers=1:num_layer_list(i)
        model{i,layers} = ss(A(1:2*layers, 1:2*layers), B(1:2*layers), C(1:2*layers), D, Ts);
    end
end


figure
hold on
box on
grid on
h = bodeplot(sys_d, '-k');
linestyle = [":g", "-.b", "--r"];
for i=1:length(date_list)
    bodeplot(model{i, num_layer_list(i)}, linestyle(i))
end
p = getoptions(h); 
p.PhaseMatching = 'on'; 
setoptions(h,p);
legend('True value', 'Random', 'Mode decomposition', 'Residual learning', 'Location', 'best')
grid on
% saveas(gcf, folder + "bode.png")

title('')


% model_ss = ssest(data_e, 8, 'Ts', data_e.Ts);
% model_ss_modal = canon(model_ss, 'modal');

% for layers=1:num_layer_list(i)
%         model_modal{i,layers} = ss(model_ss_modal.A(1:2*layers, 1:2*layers), model_ss_modal.B(1:2*layers), model_ss_modal.C(1:2*layers), model_ss_modal.D, Ts);
% end

% figure
% hold on
% box on
% grid on
% for i=1:length(date_list)
%     for layers=1:num_layer_list(i)
%         sys_norm(i, layers) = norm(model{i, layers} - sys_d);
%     end
%     plot(2*(1:num_layer_list(i)), sys_norm(i, :), '-o', 'DisplayName', num2str(num_layer_list(i)) + " layers")
% end

% for i=1:length(date_list)
%     for layers=1:num_layer_list(i)
%         sys_norm(i, layers) = norm(model_modal{i, layers} - sys_d);
%     end
%     plot(2*(1:num_layer_list(i)), sys_norm(i, :), '-.+', 'DisplayName', "modal")
% end

% for order=1:order_max
%     rsys{order} = balred(model_ss, order);
%     rsys_norm(order) = norm(rsys{order} - sys_d);
% end
% plot(1:order_max, rsys_norm, '-.*', 'DisplayName', '平衡化打ち切り')

% legend('提案法', 'モード分解', '平衡化打ち切り')
% xticks(1:order_max)
% xlabel('Order')
% ylabel('$|| \hat{G} - G ||_2 $', 'Interpreter', 'latex')
% set(gca, 'YScale', 'log')
% saveas(gcf, folder + "H2_norm.png")


% figure
% hold on
% box on
% grid on
% for i=1:length(date_list)
%     for layers=1:num_layer_list(i)
%         sys_norm_inf(i, layers) = norm(model{i, layers} - sys_d, Inf);
%     end
%     plot(2*(1:num_layer_list(i)), sys_norm_inf(i, :), '-o', 'DisplayName', num2str(num_layer_list(i)) + " layers")
% end

% for i=1:length(date_list)
%     for layers=1:num_layer_list(i)
%         sys_norm_inf(i, layers) = norm(model_modal{i, layers} - sys_d, Inf);
%     end
%     plot(2*(1:num_layer_list(i)), sys_norm_inf(i, :), '-.+', 'DisplayName', "modal")
% end

% for order=1:order_max
%     rsys{order} = balred(model_ss, order);
%     rsys_norm_inf(order) = norm(rsys{order} - sys_d, Inf);
% end
% plot(1:order_max, rsys_norm_inf, '-.*', 'DisplayName', '平衡化打ち切り')
% legend('提案法', 'モード分解', '平衡化打ち切り')
% xticks(1:order_max)
% xlabel('Order')
% ylabel('$|| \hat{G} - G ||_\infty $', 'Interpreter', 'latex')
% set(gca, 'YScale', 'log')
% saveas(gcf, folder + "Hinf_norm.png")


% figure
% hold on
% box on
% grid on
% for i=1:length(date_list)
%     bodeplot(model{i, 1})
%     bodeplot(model_modal{i, 1})
% end
% h = bodeplot(balred(sys_d, 2), sys_d);
% p = getoptions(h); 
% p.PhaseMatching = 'on'; 
% setoptions(h,p);
% legend('提案法',  'モード分解',  '平衡化打ち切り', '真値')
% grid on
% saveas(gcf, folder + "bode_order2.png")


% figure
% hold on
% box on
% grid on
% for i=1:length(date_list)
%     bodeplot(model{i, 2})
%     bodeplot(model_modal{i, 2})
% end
% h = bodeplot(balred(sys_d, 4), sys_d);
% p = getoptions(h); 
% p.PhaseMatching = 'on'; 
% setoptions(h,p);
% legend('提案法',  'モード分解',  '平衡化打ち切り', '真値')
% grid on
% saveas(gcf, folder + "bode_order4.png")




