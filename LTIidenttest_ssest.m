% 試しにさまざまな次数の線形状態空間モデルで同定してみるコード

clear
close all

M = csvread('data/csv/inoutdata_power_network.csv');
u = M(:, 1);
y = M(:, 2);

data = iddata(y, u, 1e-2);
data_e = data(1:8000);
data_v = data(8000:end);


orders = [10, 20, 40, 60, 80];
fits = zeros(1, length(orders));

for i = 1:length(orders)
    
    opt = ssestOptions('EnforceStability', true);
    model{i} = ssest(data_e, orders(i), opt)

%    opt = compareOptions();
    opt = compareOptions('InitialCondition','z');
    [yhat, fit, ic] = compare(model{i}, data_v, opt)
    fits(i) = fit;
end

save('./TrainedModels/ssest_2022-05-31.mat')

%     plot(yhat.y - data_v.y)
