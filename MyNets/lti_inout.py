#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

# 入出力形式のLTIモデル

class FIR(nn.Module):
    # Finite Impulse Responseモデル
    def __init__(self, order_u, u_size, y_size, feedthrough=False):
        super().__init__()

        if u_size != 1:
            # 多入力の場合は未実装
            raise NotImplementedError()

        if feedthrough == True:
            # 直達項有りの場合は未実装
            raise NotImplementedError()
        
        self.order_u = order_u
        self.u_size = u_size
        self.y_size = y_size
        
        self.B = nn.Linear(order_u, y_size, bias=False)
    
    def forward(self, u):
        return self.simulate(u)

    def simulate(self, u):
        reg = self.make_regressor(u)
        yhat = self.B(reg)
        return yhat.squeeze()

    def make_regressor(self, u):
        u_shifted = self.shift_signal(u, self.order_u)
        return u_shifted
    
    def shift_signal(self, x, n_delay):
        x_shifted = torch.stack([x[(n_delay-1-i):-2-i] for i in range(n_delay)], 1)
        # [x(t-1) x(t-2) ... x(t-nx)]
        return x_shifted

class ARX(nn.Module):
    # Auto-Regressive model with eXogenous input
    def __init__(self, order_u, order_y, u_size, y_size, feedthrough=False):
        super().__init__()

        if (u_size != 1) or (y_size != 1):
            # MIMOの場合は未実装
            raise NotImplementedError()

        if feedthrough == True:
            # 直達項有りの場合は未実装
            raise NotImplementedError()
        
        self.order_u = order_u
        self.order_y = order_y
        self.u_size = u_size
        self.y_size = y_size

        self.A = nn.Linear(order_y, y_size, bias=False)
        self.B = nn.Linear(order_u, y_size, bias=False)

    def forward(self, u, y=None):
        if y == None:
            yhat = self.simulate(u)
        else:
            yhat = self.predict(u, y)
        return yhat
    
    def simulate(self, u):
        # 出力データを用いず，入力データのみで回帰的に出力を計算
        N = len(u) - self.order_u - 1
        u_shifted = self.shift_signal(u, self.order_u)
        y_shifted = torch.zeros(1, self.order_y, device=u.device)

        for k in range(N):
            reg_u = u_shifted[k, :].unsqueeze(0)
            reg_y = y_shifted[k, :].unsqueeze(0)
            yhat_new = self.B(reg_u) + self.A(reg_y)

            y_shifted_new = torch.cat((yhat_new, reg_y[:, :-1]), 1)

            y_shifted = torch.cat((y_shifted, y_shifted_new),0)
        
        yhat = y_shifted[1:, 0]
        return yhat.squeeze()

    def predict(self, u, y):
        # 入出力データを用いて出力を計算
        u_shifted = self.shift_signal(u, self.order_u)
        y_shifted = self.shift_signal(y, self.order_y)
        yhat = self.B(u_shifted) + self.A(y_shifted)
        return yhat
    
    def shift_signal(self, x, n_delay):
        x_shifted = torch.stack([x[(n_delay-1-i):-2-i] for i in range(n_delay)], 1)
        # [x(t-1) x(t-2) ... x(t-nx)]
        return x_shifted