#!/usr/bin/env python
# conding: utf-8

import torch
import torch.nn as nn

def rmse(yhat, y):
    # torch.tensor型のモデル出力yhatと真値yについて，Root Mean Squared Errorを計算
    mse = nn.MSELoss()
    return torch.sqrt(mse(yhat, y))
