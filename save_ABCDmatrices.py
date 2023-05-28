#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch

from MyNets.lti import  HierarchicalLTI2, HierarchicalLTIwithLinearStateTransformer, HierarchicalLTIwithAllInput

# 学習された階層的LTIモデルのパラメータを保存

# モデル読み込み
date = '2022-11-23-4_4_oscil100_3000_losstime10'
FIGURE_FOLDER = './figures/HierLTI/' + date + '/'
model_path = './TrainedModels/HierLTI_' + date + '.pth'

NUM_LAYER = 4

model = HierarchicalLTI2(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2, send_gpu_state=False)
# model = HierarchicalLTIwithAllInput(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2, send_gpu_state=False)
# model = HierarchicalLTIwithLinearStateTransformer(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2, send_gpu_state=False)
# model = try_gpu(model)

model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER)


np.savetxt(f'{FIGURE_FOLDER}A.csv', model.A.detach().numpy(), delimiter = ',')
np.savetxt(f'{FIGURE_FOLDER}B.csv', model.B.detach().numpy(), delimiter = ',')
np.savetxt(f'{FIGURE_FOLDER}C.csv', model.C.detach().numpy(), delimiter = ',')
np.savetxt(f'{FIGURE_FOLDER}D.csv', model.D.detach().numpy(), delimiter = ',')

# np.savetxt(f'{FIGURE_FOLDER}T.csv', model.state_transformer.weight.detach().numpy(), delimiter = ',')

