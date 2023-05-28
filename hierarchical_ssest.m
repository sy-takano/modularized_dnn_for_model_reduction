% 階層的LTIモデルを残差学習

clear
close all

path = "data/csv/drivetrain/";
% load('data/csv/2022-10-05_oscil100/variables.mat')
M = csvread(path + "drivetraindata_WLTC.csv");
u = M(:, 1);
y = M(:, 2);
% Ts = 1/610.35;
Ts = 1;

data = iddata(y, u, Ts);
data_e = data;
data_v = data;

% data_e = data(5000:100000);
% data_v = data(100000:120000);


resdata = {data_e};
L = 10;     % LTIモデルの層数
for i = 1:L
    models{i} =  ssest(resdata{i}, 2, 'Ts', -1);    % 残差データに対して2次の離散時間線形状態空間モデルをフィッティング

    [yhat, yhat_sd, x] = sim(models{i}, resdata{i}.u);
    res = resdata{i}.y - yhat;
    resdata{i+1} = iddata(res, x, data_e.Ts);   % 新たな残差データを作成して次層のモデルの学習で使用
end


A = models{1}.A;
B = models{1}.B;
C = models{1}.C;
D = models{1}.D;
for i=2:L
    A(2*i-1:2*i, 2*i-1:2*i) = models{i}.A;
    A(2*i-1:2*i, 2*i-3:2*i-2) = models{i}.B;
    
    B(2*i-1:2*i) = zeros(2, 1);
    
    C = [C, models{i}.C];
end

model = ss(A, B, C, D, data_e.Ts);  % 残差学習された各層のモデルを拡大系にまとめる

figure
compare(data_v, model, models{1})   % 検証データに対するL層モデルと1層モデルの応答を表示

writematrix(model.A, path + "A_ini.csv")
writematrix(model.B, path + "B_ini.csv")
writematrix(model.C, path + "C_ini.csv")
writematrix(model.D, path + "D_ini.csv")


