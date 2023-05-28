#!/usr/bin/env python
# coding: utf-8

import time

# 経過時間を計測

def tic():
    # require to import time
    # 計測開始
    print('tic has been called')
    global start_time_tictoc
    start_time_tictoc = time.time()


def toc(tag="elapsed time"):
    # 計測終了
    if "start_time_tictoc" in globals():
        elapsed_time = time.time() - start_time_tictoc
        print("{}: {:.9f} [sec]".format(tag, elapsed_time))
        return elapsed_time
    else:
        print("tic has not been called")