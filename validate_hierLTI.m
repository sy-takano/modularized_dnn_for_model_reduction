% 学習した階層的LTIモデルを，他の方法による低次元化モデルと比較

clear
close all

folder = "./figures/HierLTI/2022-11-23-4_4_oscil100_3000_losstime10_modalstart/";

% 同定対象の情報を読み込み
load('data/csv/2022-10-05_oscil100/variables.mat');
sys_d = c2d(sys, Ts);

% order_max = length(sys.Denominator{1})-1;
order_max = 8;

% 入出力データ読み込み
M = csvread('data/csv/2022-10-05_oscil100/inoutdata_multimodes.csv');
u = M(:, 1);
y = M(:, 2);

data = iddata(y, u, Ts);
data_e = data(1:9000);
data_v = data(9000:end);

% 学習済階層的LTIモデル読み込み
date_list = ["/2022-11-23-4_4_oscil100_3000_losstime10_modalstart"];
num_layer_list = [4];
% date_list = ["2022-05-30-5", "2022-05-26-10"];
% num_layer_list = [5, 10];

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


% 従来法
model_ss = ssest(data_e, 8, 'Ts', data_e.Ts);
model_ss_modal = canon(model_ss, 'modal');  % モード分解

for layers=1:num_layer_list(i)
        model_modal{i,layers} = ss(model_ss_modal.A(1:2*layers, 1:2*layers), model_ss_modal.B(1:2*layers), model_ss_modal.C(1:2*layers), model_ss_modal.D, Ts);
end

% 平衡化打ち切りに対してH2ノルムで比較
figure
hold on
box on
grid on
for i=1:length(date_list)
    for layers=1:num_layer_list(i)
        sys_norm(i, layers) = norm(model{i, layers} - sys_d);
    end
    plot(2*(1:num_layer_list(i)), sys_norm(i, :), '-o', 'DisplayName', num2str(num_layer_list(i)) + " layers")
end

% for i=1:length(date_list)
%     for layers=1:num_layer_list(i)
%         sys_norm(i, layers) = norm(model_modal{i, layers} - sys_d);
%     end
%     plot(2*(1:num_layer_list(i)), sys_norm(i, :), '-.+', 'DisplayName', "modal")
% end

for order=1:order_max
    rsys{order} = balred(model_ss, order);
    rsys_norm(order) = norm(rsys{order} - sys_d);
end
plot(1:order_max, rsys_norm, '-.*', 'DisplayName', '平衡化打ち切り')

legend('Proposed method', 'Conventional method')
xticks(1:order_max)
xlabel('Order')
ylabel('$|| \hat{G} - G ||_2 $', 'Interpreter', 'latex')
set(gca, 'YScale', 'log')
saveas(gcf, folder + "H2_norm.png")


% モード分解，平衡化打ち切りに対してH∞ノルムで比較
figure
hold on
box on
grid on
for i=1:length(date_list)
    for layers=1:num_layer_list(i)
        sys_norm_inf(i, layers) = norm(model{i, layers} - sys_d, Inf);
    end
    plot(2*(1:num_layer_list(i)), sys_norm_inf(i, :), '-o', 'DisplayName', num2str(num_layer_list(i)) + " layers")
end

for i=1:length(date_list)
    for layers=1:num_layer_list(i)
        sys_norm_inf(i, layers) = norm(model_modal{i, layers} - sys_d, Inf);
    end
    plot(2*(1:num_layer_list(i)), sys_norm_inf(i, :), '-.+', 'DisplayName', "modal")
end

for order=1:order_max
    rsys{order} = balred(model_ss, order);
    rsys_norm_inf(order) = norm(rsys{order} - sys_d, Inf);
end
plot(1:order_max, rsys_norm_inf, '-.*', 'DisplayName', '平衡化打ち切り')
legend('提案法', 'モード分解', '平衡化打ち切り')
xticks(1:order_max)
xlabel('Order')
ylabel('$|| \hat{G} - G ||_\infty $', 'Interpreter', 'latex')
set(gca, 'YScale', 'log')
saveas(gcf, folder + "Hinf_norm.png")


% モデルが2次の時，モード分解，平衡化打ち切りに対してボード線図で比較
figure
hold on
box on
grid on
for i=1:length(date_list)
    bodeplot(model{i, 1})
    bodeplot(model_modal{i, 1})
end
h = bodeplot(balred(sys_d, 2), sys_d);
p = getoptions(h); 
p.PhaseMatching = 'on'; 
setoptions(h,p);
legend('提案法',  'モード分解',  '平衡化打ち切り', '真値')
grid on
saveas(gcf, folder + "bode_order2.png")


% モデルが4次の時，モード分解，平衡化打ち切りに対してボード線図で比較
figure
hold on
box on
grid on
for i=1:length(date_list)
    bodeplot(model{i, 2})
    bodeplot(model_modal{i, 2})
end
h = bodeplot(balred(sys_d, 4), sys_d);
p = getoptions(h); 
p.PhaseMatching = 'on'; 
setoptions(h,p);
legend('提案法',  'モード分解',  '平衡化打ち切り', '真値')
grid on
saveas(gcf, folder + "bode_order4.png")


% モデルが低次元化前の時，ssestに対してボード線図で比較
figure
hold on
box on
grid on
for i=1:length(date_list)
    bodeplot(model{i, num_layer_list(i)})
    bodeplot(model_modal{i, num_layer_list(i)})
end
h = bodeplot(sys_d);
p = getoptions(h); 
p.PhaseMatching = 'on'; 
setoptions(h,p);
legend('提案法',  'ssest', '真値')
grid on
saveas(gcf, folder + "bode.png")


% data_e.y = data_e.y + 0.0001*randn('like', data_e.y);


model_arx = arx(data_e, [8, 6, 1])  % 同定対象と同様の構造を持つARXモデルを推定

% モデルが低次元化前の時，ARXとssestに対してボード線図で比較
figure
hold on
box on
grid on
for i=1:length(date_list)
    bodeplot(model{i, num_layer_list(i)})
end
h = bodeplot(model_arx);
h = bodeplot(model_ss);
h = bodeplot(sys_d);
p = getoptions(h); 
p.PhaseMatching = 'on'; 
setoptions(h,p);
legend('提案法', 'ARX', 'ssest', '真値')
grid on
saveas(gcf, folder + "bode_arx.png")

% 同定データに対する応答を比較
figure
compare(data_e(1:1000), model{1, num_layer_list(1)}, model_arx, model_ss)
grid on
legend('Training data', 'Proposed model', 'ARX', 'ssest')
saveas(gcf, folder + "compare_train.png")

% 検証データに対する応答を比較
figure
compare(data_v, model{1, num_layer_list(1)}, model_arx, model_ss)
grid on
legend('Validation data', 'Proposed model', 'ARX', 'ssest')
saveas(gcf, folder + "compare_val.png")

% 低次元化モデルのボード線図をプロット
figure
hold on
box on
grid on
for i=1:num_layer_list(1)
    bodeplot(model{1, i})
end
h = bodeplot(sys_d);
p = getoptions(h); 
p.PhaseMatching = 'on'; 
setoptions(h,p);
legend('2nd order', '4th order', '6th order', '8th order', 'True value')
grid on
saveas(gcf, folder + "bode_pruned.png")


% 
% figure
% hold on
% box on
% for i=1:length(date_list)
%         iopzplot(model{i, 1})
% end
% iopzplot(balred(sys_d, 2))
% iopzplot(sys_d)
% % grid on
% legend('提案法', '平衡化打ち切り', '真値')
% saveas(gcf, folder + "pzplot_order2.png")
% 
% % xlim([0.5, 1])
% % saveas(gcf, folder + "pzplot_order2_zoom.png")
% 
% figure
% hold on
% box on
% for i=1:length(date_list)
%         iopzplot(model{i, 4})
% end
% iopzplot(sys_d)
% % grid on
% legend('提案法', '真値')
% saveas(gcf, folder + "pzplot.png")
% 
% % xlim([0.5, 1])
% % saveas(gcf, folder + "pzplot_zoom.png")


