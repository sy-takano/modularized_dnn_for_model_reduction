#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from typing import Union

from torch.nn import init
from MyNets.dnn import DeepNeuralNetwork
from MyNets.hierarchical_dnn import HierarchicalNeuralNetwork
from MyNets.lti import HierarchicalLTI2

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

class RNNLFR(torch.nn.Module):
    # 非線形LFRモデルをRNNの特殊なケースとして実装
    # 非線形LFRの非線形関数部分は3層NN
    def __init__(self, hidden_size, x_size=2, u_size=1, y_size=1, w_size=1, z_size=1, send_gpu_state=True, nonlinearity='relu'):
        super().__init__()
        self.x_size = x_size
        self.u_size = u_size
        self.y_size = y_size
        self.w_size = w_size
        self.z_size = z_size
        
        self.A = nn.Linear(self.x_size, self.x_size, bias=False)
        self.Bu = nn.Linear(self.u_size, self.x_size, bias=False)
        self.Bw = nn.Linear(self.w_size, self.x_size, bias=False)
        self.Cy = nn.Linear(self.x_size, self.y_size, bias=False)
        self.Dyu = nn.Linear(self.u_size, self.y_size, bias=False)
        self.Dyw = nn.Linear(self.w_size, self.y_size, bias=False)
        self.Cz = nn.Linear(self.x_size, self.z_size, bias=False)
        self.Dzu = nn.Linear(self.u_size, self.z_size, bias=False)
        self.Dzw = nn.Linear(self.w_size, self.z_size, bias=False)     

        self.fc1 = nn.Linear(self.z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.w_size)
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()


        # self.f_bias = nn.Parameter(torch.tensor([[0.0]]))

        # self.A.weight = nn.Parameter(torch.tensor([[0.0, -0.3204],
        #                                             [1, 1.2]]))
        # self.Bu.weight = nn.Parameter(torch.tensor([[1.0],
        #                                             [0.0]]))
        # self.Cy.weight = nn.Parameter(torch.tensor([[0.4246, 0.3311]]))
        # self.Dyu.weight = nn.Parameter(torch.tensor([[0.0]]))

        # self.A.weight = nn.Parameter(torch.tensor([[0.7914]]))
        # self.Bu.weight = nn.Parameter(torch.tensor([[1.0]]))
        # self.Cy.weight = nn.Parameter(torch.tensor([[0.4246]]))
        # self.Dyu.weight = nn.Parameter(torch.tensor([[0.0]]))

        # rng = 0.01
        # nn.init.uniform_(self.fc1.weight, -rng, rng)
        # nn.init.uniform_(self.fc2.weight, -rng, rng)
        # self.fc1.bias = nn.Parameter(torch.tensor([[0.0]]))
        # self.fc2.bias = nn.Parameter(torch.tensor([[0.0]]))
        # self.initialize_layer(self.fc1, 0.01)
        # self.initialize_layer(self.fc2, 0.01)


        self.z_record = []
        self.w_record = []
        self.y_record = []
        self.x_record = []

        self.send_gpu_state = send_gpu_state
    
    def forward(self, input):
        N = len(input)

        input = input.reshape(1, -1)

        self.initialize_record()
        x = self.initialize_state()
        self.x_record.append(x)
        for time in range(N):
            u = input[:, time]
            z = self.get_output_z(x, u)
            w = self.get_output_w(z)
            y = self.get_output_y(x, u, w)
            x = self.one_step_ahead(x, u, w)

            self.record(z, w, y, x)

        y = torch.stack(self.y_record)
        
        return y
    
    def get_output_z(self, x, u):
        z = self.Cz(x) + self.Dzu(u)
        return z
    
    def fbar(self, z):
        return self.fc2(self.activation(self.fc1(z)))
    
    def f(self, z):
        return self.fc2(self.activation(self.fc1(z)))

    def get_output_w(self, z):
        w = self.f(z)
        # w = z * self.fbar(z) + self.f_bias
        return w
    
    def get_output_y(self, x, u, w):
        y = self.Cy(x) + self.Dyu(u) + self.Dyw(w)
        return y.squeeze()
    
    def one_step_ahead(self, x, u, w):
        x = self.A(x) + self.Bu(u) + self.Bw(w)
        return x
    
    def record(self, z_new, w_new, y_new, x_new):
        self.z_record.append(z_new)
        self.w_record.append(w_new)
        self.y_record.append(y_new)
        self.x_record.append(x_new)
    
    def initialize_state(self, init_state = 0):
        if init_state == 0:
            if self.send_gpu_state == True:
                x0 = try_gpu(torch.zeros(1, self.x_size))
            else:
                x0 = torch.zeros(1, self.x_size)
                
        else:
            x0 = init_state

        return x0
    
    def initialize_record(self):
        self.z_record = []
        self.w_record = []
        self.y_record = []
        self.x_record = []
    
    def initialize_layer(self, layer:nn.Linear, range):
        nn.init.uniform_(layer.weight, -range, range)
        layer.bias = nn.Parameter(torch.tensor([[0.0]]))


class HierarichicalRNNLFR(RNNLFR):
    # 非線形LFRの非線形関数部分に階層的DNNを使用
    def __init__(self, hidden_size, num_layer, x_size=2, u_size=1, y_size=1, w_size=1, z_size=1, send_gpu_state=True, nonlinearity='relu'):
        super().__init__(hidden_size, x_size=x_size, u_size=u_size, y_size=y_size, w_size=w_size, z_size=z_size, send_gpu_state=send_gpu_state, nonlinearity=nonlinearity)

        self.net = HierarchicalNeuralNetwork(input_size=self.z_size, hidden_size=hidden_size, output_size=self.w_size, num_layer=num_layer, nonlinearity=nonlinearity)
    
    def forward(self, input, num_used_layers=None):
        N = len(input)

        input = input.reshape(1, -1)

        self.initialize_record()
        x = self.initialize_state()
        self.x_record.append(x)
        for time in range(N):
            u = input[:, time]
            z = self.get_output_z(x, u)
            w = self.get_output_w(z, num_used_layers)
            y = self.get_output_y(x, u, w)
            x = self.one_step_ahead(x, u, w)

            self.record(z, w, y, x)

        y = torch.stack(self.y_record)
        
        return y
    
    def get_output_w(self, z, num_used_layers=None):
        w, _intermediate_w = self.net(z, num_used_layers=num_used_layers)
        return w


class HierarchicalLTILFR(HierarchicalLTI2):
    # 非線形LFRのLTI部分に階層的LTIを使用
    # 非線形関数部分はFNN
    def __init__(self, num_layer_NN, hidden_size, num_layer_LTI, u_size, y_size, order_of_a_LTImodel=2,  w_size=1, z_size=1, send_gpu_state=True, nonlinearity='relu'):
        super().__init__(num_layer_LTI, u_size+w_size, y_size+z_size, order_of_a_LTImodel, send_gpu_state)


        # self.x_size = x_size
        # self.u_size = u_size
        # self.y_size = y_size
        self.w_size = w_size
        self.z_size = z_size

        self.net = DeepNeuralNetwork(input_size=self.z_size, hidden_size=hidden_size, output_size=self.w_size, num_layer=num_layer_NN, nonlinearity=nonlinearity)
    
    def simulate(self, input, num_used_layers_LTI, init_state=None):
        N = len(input)
        input = input.reshape(-1, self.u_size-self.w_size)

        self.A, self.B, self.C, self.D = self.make_ABCDmatrices(num_used_layers_LTI)

        y_record, x_record, self.w_record, self.z_record = self.initialize_record()
        x = self.initialize_state(order=self.order_of_a_LTImodel*num_used_layers_LTI, init_state=init_state)
        w = self.initialize_w()
        x_record.append(x)
        for time in range(N):
            u = input[time, :].unsqueeze(1)
            w = w.unsqueeze(1)

            u_w = torch.cat((u, w), dim=0)

            y_z = self.get_output(x, u_w)
            y = y_z[:-self.z_size]
            z = y_z[self.y_size-self.z_size:]

            w = self.get_output_w(z)

            x = self.one_step_ahead(x, u_w)

            x_record.append(x)
            y_record.append(y)
            self.w_record.append(w)
            self.z_record.append(z)

        y = torch.stack(y_record)
        x = torch.stack(x_record[:-1])
        return y.squeeze(), x
    
    def initialize_w(self, init_w = 0):
        if init_w == 0:
            if self.send_gpu_state == True:
                w0 = try_gpu(torch.zeros(self.w_size))
            else:
                w0 = torch.zeros(self.w_size)
                
        else:
            w0 = init_w

        return w0
    
    def initialize_record(self):
        y_record = []
        x_record = []
        w_record = []
        z_record = []
        return y_record, x_record, w_record, z_record
    
    def get_output_w(self, z):
        w = self.net(z)
        return w

class HierarchicalNeuralLFR(HierarchicalLTILFR):
    # 階層的LTIと階層的DNNで非線形LFRモデルを構成
    def __init__(self, num_layer_NN, hidden_size, num_layer_LTI, u_size, y_size, order_of_a_LTImodel=2, w_size=1, z_size=1, send_gpu_state=True, nonlinearity='relu'):
        super().__init__(num_layer_NN, hidden_size, num_layer_LTI, u_size, y_size, order_of_a_LTImodel, w_size, z_size, send_gpu_state, nonlinearity)

        self.num_layer_NN = num_layer_NN
        self.num_layer_LTI = num_layer_LTI

        self.net = HierarchicalNeuralNetwork(input_size=self.z_size, hidden_size=hidden_size, output_size=self.w_size, num_layer=num_layer_NN, nonlinearity=nonlinearity)
    
    def forward(self, input, num_used_layers_LTI=None, num_used_layers_NN=None, init_state=None):
        if num_used_layers_LTI == None:
            num_used_layers_LTI = self.num_layer_LTI
        if num_used_layers_NN == None:
            num_used_layers_NN = self.num_layer_NN
        if num_used_layers_LTI < 1 or num_used_layers_NN < 1:
            raise NotImplementedError()
        
        y, _x = self.simulate(input, num_used_layers_LTI, num_used_layers_NN, init_state)

        return y

    def simulate(self, input, num_used_layers_LTI, num_used_layers_NN, init_state=None):
        N = len(input)
        input = input.reshape(-1, self.u_size-self.w_size)

        self.A, self.B, self.C, self.D = self.make_ABCDmatrices(num_used_layers_LTI)

        y_record, x_record, self.w_record, self.z_record = self.initialize_record()
        x = self.initialize_state(order=self.order_of_a_LTImodel*num_used_layers_LTI, init_state=init_state)
        w = self.initialize_w().unsqueeze(1)
        x_record.append(x)
        for time in range(N):
            u = input[time, :].unsqueeze(1)

            u_w = torch.cat((u, w), dim=0)

            y_z = self.get_output(x, u_w)
            y = y_z[:-self.z_size]
            z = y_z[self.y_size-self.z_size:].unsqueeze(1)

            w = self.get_output_w(z, num_used_layers_NN)

            x = self.one_step_ahead(x, u_w)

            x_record.append(x)
            y_record.append(y)
            self.w_record.append(w)
            self.z_record.append(z)

        y = torch.stack(y_record)
        x = torch.stack(x_record[:-1])
        return y.squeeze(), x

    def get_output_w(self, z, num_used_layers_NN):
        w, _intermediate_output = self.net(torch.transpose(z, 1, 0), num_used_layers_NN)
        return torch.transpose(w, 1, 0)


class DNNLFR(HierarichicalRNNLFR):
    # 非線形LFRの非線形関数部分にFNNを使用（LTI部分は通常通り）
    def __init__(self, hidden_size, num_layer, x_size=2, u_size=1, y_size=1, w_size=1, z_size=1, send_gpu_state=True, nonlinearity='relu'):
        super().__init__(hidden_size, num_layer, x_size=x_size, u_size=u_size, y_size=y_size, w_size=w_size, z_size=z_size, send_gpu_state=send_gpu_state, nonlinearity=nonlinearity)
        self.net = DeepNeuralNetwork(input_size=self.z_size, hidden_size=hidden_size, output_size=self.w_size, num_layer=num_layer, nonlinearity=nonlinearity)
    
    def get_output_w(self, z, num_used_layers=None):
        w = self.net(z)
        return w


class NonlinearLFR(RNNLFR):
    # 非線形関数を引数として外から与えるために作ったクラス．他はRNNLFRと同じ．
    def __init__(self, x_size, u_size, y_size, w_size, z_size, nonlinear_function, send_gpu_state=True):
        hidden_size = 1     # 一時凌ぎのダミー．使わない
        nonlinearity='relu' # 同上
        super().__init__(hidden_size=hidden_size, x_size=x_size, u_size=u_size, y_size=y_size, w_size=w_size, z_size=z_size, send_gpu_state=send_gpu_state, nonlinearity=nonlinearity)
        self.nonlinear_function = nonlinear_function
    
    def get_output_w(self, z):
        w = self.nonlinear_function(z)
        return w


class ModelClosure(torch.nn.Module):
    # 層数を変えた時の出力誤差の加重平均をとる評価関数を定義するためのクロージャ
    def __init__(self, model: Union[RNNLFR, HierarichicalRNNLFR], input_train, output_train, loss_fcn):
        super().__init__()
        self.model = model
        self.input = input_train.reshape(1, -1)
        self.output = output_train
        self.loss_fcn = loss_fcn
        self.data_length = len(input_train)

        self.hierarichical_rnn_lfr_flag = isinstance(model, HierarichicalRNNLFR)
        self.target_intermeiate_output = torch.zeros(1)
        if self.model.send_gpu_state == True:
            self.target_intermeiate_output = try_gpu(self.target_intermeiate_output)
        self.regularization_parameter = [0, 0.001, 0.01, 0.1]
    
    def forward(self):
        L = 0.0
        x = self.model.initialize_state()
        for time in range(self.data_length):
            u = self.input[:, time]
            z = self.model.get_output_z(x, u)
            w = self.model.get_output_w(z)
            y = self.model.get_output_y(x, u, w)

            L = L + self.loss_fcn(y, self.output[time])
            if self.hierarichical_rnn_lfr_flag:
                for idx in range(1, self.model.num_layer):
                    gamma = self.regularization_parameter[idx]
                    L = L + gamma * self.loss_fcn(self.model.intermeiate_output_w[idx], self.target_intermeiate_output)

            x = self.model.one_step_ahead(x, u, w)
        
        return L/self.data_length
    