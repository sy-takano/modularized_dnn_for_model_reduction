#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import math
import random

from MyUtilities.ndarray_to_tensor import ndarray2tensor
from MyUtilities.try_gpu import try_gpu

# csvデータを読み込んで格納するクラス

class InOutDataNARX:
    def __init__(self, data_path, n_u_delay, n_y_delay, send_gpu=True):
        self.regressor, self.target, self.time, self.output, self.input, self.gear, self.num_sample = self.make_regressor_SISO(data_path, n_u_delay, n_y_delay)
        if send_gpu == True:
            self.regressor = try_gpu(self.regressor)
            self.target = try_gpu(self.target)
            self.output = try_gpu(self.output)
            self.input = try_gpu(self.input)

    def load_csvdata_as_tensor(self, path: str):
        data_ndarray = np.loadtxt(path, dtype=float, delimiter=',')  # データ読み込み
        data_tensor = ndarray2tensor(data_ndarray)  # tensor型に変換
        return data_tensor

    def shift_signal(self, x, n_delay):
        x_shifted = torch.stack([x[(n_delay-1-i):-2-i]
                                for i in range(n_delay)], 1)
        # [x(t-1) x(t-2) ... x(t-nx)]
        return x_shifted

    def make_regressor_SISO(self, path: str, n_u_delay, n_y_delay):
        # SISOシステムの入出力データ（csv）を読み込んでレグレッサに整形
        if n_u_delay != n_y_delay:
            print('nu must be the same as ny')

        data_tensor = self.load_csvdata_as_tensor(path)

        t = data_tensor[:, 0]   # 時刻
        u = data_tensor[:, 1]   # 動的システムへの入力
        y = data_tensor[:, 2]   # 動的システムの出力
        gear = data_tensor[:, 3]

        u_shifted = self.shift_signal(u, n_u_delay)
        y_shifted = self.shift_signal(y, n_y_delay)

        # y_target = y[n_y_delay:-1]    # y(t)
        y_target = y[n_y_delay:-1].reshape((-1, 1))

        num_sample = len(t) - n_y_delay   # データ数

        # [u(t-1) ... u(t-nu) y(t-1) ... y(t-ny)], t=n, n+1, ... を行ベクトルとした行列
        regressor = torch.cat([u_shifted, y_shifted], 1)

        return regressor, y_target, t, y, u, gear, num_sample


class InOutData(InOutDataNARX):
    def __init__(self, data_path, send_gpu=True):
        self.time, self.output, self.input = self.load(data_path)
        if send_gpu == True:
            self.output = try_gpu(self.output)
            self.input = try_gpu(self.input)

    def load(self, path: str):
        data_tensor = self.load_csvdata_as_tensor(path)

        t = data_tensor[:, 0]   # 時刻
        u = data_tensor[:, 1]   # 動的システムへの入力
        y = data_tensor[:, 2]   # 動的システムの出力

        return t, y, u

class InOutDataSilverbox(InOutDataNARX):
    def __init__(self, data_path, send_gpu=True):
        self.output, self.input = self.load(data_path)
        if send_gpu == True:
            self.output = try_gpu(self.output)
            self.input = try_gpu(self.input)

    def load(self, path: str):
        data_tensor = self.load_csvdata_as_tensor(path)

        u = data_tensor[:, 0]   # 動的システムへの入力
        y = data_tensor[:, 1]   # 動的システムの出力

        return y, u

class InOutDatawithState(InOutDataNARX):
    def __init__(self, data_path, dim_states, send_gpu=True):
        self.output, self.input, self.state = self.load(data_path, dim_states)
        if send_gpu == True:
            self.output = try_gpu(self.output)
            self.input = try_gpu(self.input)
            self.state = try_gpu(self.state)

    def load(self, path: str, dim_states):
        data_tensor = self.load_csvdata_as_tensor(path)

        u = data_tensor[:, 0]   # 動的システムへの入力
        y = data_tensor[:, 1]   # 動的システムの出力
        t = data_tensor[:, 2]
        x = data_tensor[:, 3:3+dim_states]

        return y, u, x

class MyLoader:
    def __init__(self, input_seq, target_seq, batch_size):
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.batch_size = batch_size

        self.seq_len = len(input_seq)
    
    def get_data(self):
        idx = math.floor(random.uniform(0, self.seq_len - self.batch_size))
        out_input = self.input_seq[idx : idx + self.batch_size]
        out_target = self.target_seq[idx : idx + self.batch_size]
        
        return out_input, out_target
