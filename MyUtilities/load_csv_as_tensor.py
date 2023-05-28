from MyUtilities.ndarray_to_tensor import ndarray2tensor
import numpy as np

def load_csv_as_tensor(path):
    # csvデータを読み込んでtorch.tensor型にする
    data_ndarray = np.loadtxt(path, dtype=float, delimiter=',')  # データ読み込み
    data_tensor = ndarray2tensor(data_ndarray)  # tensor型に変換
    return data_tensor