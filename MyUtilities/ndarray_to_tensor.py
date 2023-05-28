#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

def ndarray2tensor(x_ndarray):
    # numpy.ndarrayをtorch.tensorに型変換
    x_ndarray = x_ndarray.astype(np.float32)
    x_tensor = torch.from_numpy(x_ndarray).clone()
    return x_tensor