#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from MyNets.dnn import DeepNeuralNetwork

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

class LinearStateSpace(nn.Module):
    # pytorchでの線形状態空間モデル
    def __init__(self, order, u_size, y_size, feedthrough=False, send_gpu_state=True):
        super().__init__()

        self.u_size = u_size
        self.x_size = order
        self.send_gpu_state = send_gpu_state

        self.A = nn.Linear(order, order, bias=False)
        self.B = nn.Linear(u_size, order, bias=False)
        self.C = nn.Linear(order, y_size, bias=False)
        self.D = nn.Linear(u_size, y_size, bias=False)
        if feedthrough == False:
            self.D.weight = nn.Parameter(torch.zeros_like(self.D.weight), requires_grad=False)
    
    def forward(self, input):
        y, _x = self.simulate(input)
        return y
    
    def simulate(self, input):
        N = len(input)

        input = input.reshape(self.u_size, -1)

        y_record, x_record = self.initialize_record()
        x = self.initialize_state()
        x_record.append(x)
        for time in range(N):
            u = input[:, time]
            y = self.get_output(x, u)
            x = self.one_step_ahead(x, u)

            x_record.append(x)
            y_record.append(y)

        y = torch.stack(y_record)
        x = torch.stack(x_record[:-1])
        return y, x
    
    def get_output(self, x, u):
        y = self.C(x) + self.D(u)
        return y.squeeze()
    
    def one_step_ahead(self, x, u):
        x = self.A(x) + self.B(u)
        return x
    
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
        y_record = []
        x_record = []
        return y_record, x_record

class HierarchicalLTI(nn.Module):
    # 階層的LTIモデル
    def __init__(self, num_layer, u_size, y_size, order_of_a_LTImodel=2):
        # order_of_a_LTImodel: 1層分の最小単位としての線形モデルの次数
        super().__init__()
        self.u_size = u_size
        self.y_size = y_size
        self.num_layer = num_layer
        self.order_of_a_LTImodel = order_of_a_LTImodel

        dim_in = [order_of_a_LTImodel]*(num_layer-1)
        dim_in.insert(0, u_size)

        dim_out = [y_size]*num_layer

        self.lti_list = nn.ModuleList([LinearStateSpace(order=order_of_a_LTImodel, u_size=dim_in[i], y_size=dim_out[i]) for i in range(num_layer)])

        # self.initialize_A(0.8)

    def forward(self, u, num_used_layers=None):
        if num_used_layers == None:
            num_used_layers = self.num_layer
        if num_used_layers < 1:
            raise NotImplementedError()

        o, x = self.lti_list[0].simulate(u)
        y = o

        for i in range(1, num_used_layers):
            o, x = self.lti_list[i].simulate(x)
            y = y + o

        return y
    
    def initialize_A(self, init):
        for i in range(self.num_layer):
            self.lti_list[i].A.weight = nn.Parameter(init * torch.eye(self.order_of_a_LTImodel))

    # def initialize_A_mode(self, init1, init2):
    #     for i in range(self.num_layer):
    #         A_diag = init1 * torch.eye(self.order_of_a_LTImodel)
    #         A_nondiag = init2 * torch.eye(self.order_of_a_LTImodel).permute((1, 0)) # 2x2のみ対応
    #         print(A_nondiag.permute((1, 0)))
    #         self.lti_list[i].A.weight = nn.Parameter(A_diag + A_nondiag)
    #         print(self.lti_list[i].A.weight)


class HierarchicalLTI2(HierarchicalLTI):
    # 階層的LTIモデル
    # 拡大系で表現することで，HierarchicalLTIよりも計算を高速化
    def __init__(self, num_layer, u_size, y_size, order_of_a_LTImodel=2, send_gpu_state=True):
        super().__init__(num_layer, u_size, y_size, order_of_a_LTImodel)
        self.send_gpu_state = send_gpu_state
        
    def forward(self, input, num_used_layers=None, init_state=None):
        if num_used_layers == None:
            num_used_layers = self.num_layer
        if num_used_layers < 1:
            raise NotImplementedError()
        
        y, _x = self.simulate(input, num_used_layers, init_state)
        return y
    
    def simulate(self, input, num_used_layers, init_state=None):
        N = len(input)
        input = input.reshape(-1, self.u_size)
        # input = input.reshape(self.u_size, -1)

        self.A, self.B, self.C, self.D = self.make_ABCDmatrices(num_used_layers)

        y_record, x_record = self.initialize_record()
        x = self.initialize_state(order=self.order_of_a_LTImodel*num_used_layers, init_state=init_state)
        x_record.append(x)
        for time in range(N):
            u = input[time, :].unsqueeze(1)
            y = self.get_output(x, u)
            x = self.one_step_ahead(x, u)

            x_record.append(x)
            y_record.append(y)

        y = torch.stack(y_record)
        x = torch.stack(x_record[:-1])
        return y, x

    def get_output(self, x, u):
        y = torch.mm(self.C, x) + torch.mm(self.D, u)
        return y.squeeze()
    
    def one_step_ahead(self, x, u):
        x = torch.mm(self.A, x) + torch.mm(self.B, u)
        return x
    
    def initialize_state(self, order, init_state = None):
        if init_state == None:
            if self.send_gpu_state == True:
                x0 = try_gpu(torch.zeros(order, 1))
            else:
                x0 = torch.zeros(order, 1)
                
        else:
            x0 = init_state.reshape(-1, 1)
            
        return x0
    
    def initialize_record(self):
        y_record = []
        x_record = []
        return y_record, x_record
    
    def make_ABCDmatrices(self, num_used_layers):
        if num_used_layers > 1:
            A = self.bidiagonal_matrix_lower([self.lti_list[i].A.weight for i in range(num_used_layers)], [self.lti_list[i].B.weight for i in range(1, num_used_layers)])
            
            B_zerovec = torch.zeros((self.order_of_a_LTImodel * (num_used_layers- 1), self.u_size))
            if self.send_gpu_state == True:
                B_zerovec = try_gpu(B_zerovec)

            B = torch.cat((self.lti_list[0].B.weight, B_zerovec), 0)
        elif num_used_layers == 1:
            A = torch.block_diag(self.lti_list[0].A.weight)
            B = torch.block_diag(self.lti_list[0].B.weight)
        else:
            raise NotImplementedError()

        C = torch.cat([self.lti_list[i].C.weight for i in range(num_used_layers)], 1)
        D = self.lti_list[0].D.weight

        return A, B, C, D
    
    def bidiagonal_matrix_lower(self, diag_matrices, lower_matrices):
        # diag_matricesを対角ブロック成分に，lower_matricesを対角ブロック成分の一つ下のブロック成分に並べる
        # 他の成分は零で埋める
        
        n = diag_matrices[0].shape[0]   # ブロック成分のサイズ n×n
        N = n*len(diag_matrices)        # 出力される行列のサイズ N×N

        if n != lower_matrices[0].shape[0]:
            raise NotImplementedError()
        if len(diag_matrices) - 1 != len(lower_matrices):
            raise NotImplementedError()
        
        X_diag = torch.block_diag(*diag_matrices)   # サイズ: N×N
        X_lower = torch.block_diag(*lower_matrices) # サイズ: N-n × N-n

        added_row = torch.zeros((n, N-n))
        added_column = torch.zeros((N, n))

        if self.send_gpu_state == True:
            added_row = try_gpu(added_row)
            added_column = try_gpu(added_column)

        X_lower = torch.cat((added_row, X_lower), 0)    # サイズ: N × N-n
        X_lower = torch.cat((X_lower, added_column), 1) # サイズ: N × N

        return X_diag + X_lower

class HierarchicalLTIwithAllInput(HierarchicalLTI2):
    # 1番目のモジュールのみに入力を与えるのではなく，全てのモジュールに入力を与える構造の階層的LTI
    def __init__(self, num_layer, u_size, y_size, order_of_a_LTImodel=2, send_gpu_state=True):
        super().__init__(num_layer, u_size, y_size, order_of_a_LTImodel, send_gpu_state)
        self.Bu_list = nn.ModuleList(nn.Linear(u_size, order_of_a_LTImodel, bias=False) for i in range(num_layer))

    def make_ABCDmatrices(self, num_used_layers):
        A, _B, C, D = super().make_ABCDmatrices(num_used_layers)

        B = torch.cat([self.Bu_list[i].weight for i in range(num_used_layers)], 0)
                
        return A, B, C, D

class HierarchicalLTIwithKalmanFilter(HierarchicalLTI2):
    # イノベーション表現を用いたオブザーバ付きの階層的LTI
    # 改めて考えるとここで用いている状態推定器はオブザーバであってカルマンフィルタとは言えない
    def __init__(self, num_layer, u_size, y_size, order_of_a_LTImodel=2, send_gpu_state=True):
        super().__init__(num_layer, u_size, y_size, order_of_a_LTImodel, send_gpu_state)

        self.kalman_gain = nn.Linear(y_size, num_layer*order_of_a_LTImodel, bias=False) # オブザーバゲイン
    
    def forward(self, input, output=None, num_used_layers=None, init_state=None, init_state_est_time=None):
        if num_used_layers == None:
            num_used_layers = self.num_layer
        if num_used_layers < 1:
            raise NotImplementedError()

        if output==None:
            y, _x = self.simulate(input, num_used_layers, init_state)
        else:
            y, _x = self.predict(input, output, num_used_layers, init_state, init_state_est_time)
        return y
    
    def predict(self, input, output, num_used_layers, init_state=None, init_state_est_time=None):
        if init_state_est_time==None:
            init_state_est_time = 0

        N = len(input)
        input = input.reshape(self.u_size, -1)
        output = output.reshape(self.y_size, -1)

        self.A, self.B, self.C, self.D, self.K = self.make_ABCDKmatrices(num_used_layers)

        y_record, x_record = self.initialize_record()
        # x = self.initialize_state(order=self.order_of_a_LTImodel*num_used_layers, init_state=init_state)
        x = self.initialize_state(order=self.order_of_a_LTImodel*self.num_layer, init_state=init_state)
        x_record.append(x)
        for time in range(N):
            if time < init_state_est_time:
                # 状態推定を行う
                u = input[:, time].reshape(-1, self.u_size)
                y_true = output[:, time].reshape(-1, self.y_size)
                y = self.get_output(x, u)
                x = self.one_step_ahead(x, u, y_true, y)
            else:
                # (time - init_state_est_time)段先予測を行う
                u = input[:, time].reshape(-1, self.u_size)
                y = self.get_output(x, u)
                x = self.one_step_ahead(x, u)

            x_record.append(x)
            y_record.append(y)

        y = torch.stack(y_record)
        x = torch.stack(x_record[:-1])
        return y, x
    
    def one_step_ahead(self, x, u, y_true=None, y=None):
        if y_true==None or y==None:
            return super().one_step_ahead(x, u)
        else:
            x = torch.mm(self.A, x) + torch.mm(self.B, u) + torch.mm(self.K, y_true - y)
            return x
    
    def make_ABCDKmatrices(self, num_used_layers):
        A, B, C, D = super().make_ABCDmatrices(self.num_layer)
        C = torch.cat([self.lti_list[i].C.weight for i in range(self.num_layer)], 1)
        C[:, self.order_of_a_LTImodel*num_used_layers:] = torch.zeros_like(C[:, self.order_of_a_LTImodel*num_used_layers:])
        K = self.kalman_gain.weight
        return A, B, C, D, K

class HierarchicalLTIwithStateTransformer(HierarchicalLTI2):
    # 真のシステムの初期状態のデータが与えられているとき，それを適当に座標変換するNNをもつ階層的LTI
    def __init__(self, num_layer, u_size, y_size, order_of_a_LTImodel=2, send_gpu_state=True):
        super().__init__(num_layer, u_size, y_size, order_of_a_LTImodel, send_gpu_state)

        self.state_transformer = DeepNeuralNetwork(input_size=num_layer*order_of_a_LTImodel+1, output_size=num_layer*order_of_a_LTImodel, hidden_size=64, num_layer=3)
    
    def forward(self, input, num_used_layers=None, init_state=None):
        if num_used_layers == None:
            num_used_layers = self.num_layer
        if num_used_layers < 1:
            raise NotImplementedError()
        
        if init_state != None:
            init_state = self.get_init_state(num_used_layers, init_state)

        y, _x = self.simulate(input, num_used_layers, init_state)
        return y
    
    def get_init_state(self, num_used_layers, x0_true):
        if self.send_gpu_state == True:
            init_state = self.state_transformer(torch.cat((x0_true, torch.tensor([num_used_layers], dtype=torch.float, device='cuda'))))
        else:
            init_state = self.state_transformer(torch.cat((x0_true, torch.tensor([num_used_layers], dtype=torch.float))))

        return init_state[:num_used_layers*self.order_of_a_LTImodel]


class HierarchicalLTIwithLinearStateTransformer(HierarchicalLTIwithStateTransformer):
    # 真のシステムの初期状態のデータが与えられているとき，それを適当に座標変換する行列をもつ階層的LTI
    def __init__(self, num_layer, u_size, y_size, order_of_a_LTImodel=2, x_size_true=None, send_gpu_state=True):
        super().__init__(num_layer, u_size, y_size, order_of_a_LTImodel, send_gpu_state)
        if x_size_true == None:
            x_size_true = num_layer*order_of_a_LTImodel
        self.state_transformer = nn.Linear(in_features=x_size_true, out_features=num_layer*order_of_a_LTImodel, bias=False)
    
    def get_init_state(self, num_used_layers, x0_true):        
        init_state = self.state_transformer(x0_true)

        return init_state[:num_used_layers*self.order_of_a_LTImodel]