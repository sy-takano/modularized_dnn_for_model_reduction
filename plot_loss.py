#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 学習中の評価関数の変化をプロット

    # date = '2022-09-03'
    # FIGURE_FOLDER = './figures/HierDNN/' + date + '/'
    FIGURE_FOLDER = 'figures/HierDNN/2023-01-09-3-5-64/'

    num_layer = [5]
    # hidden_size = [64 ,37, 23, 15]

    # fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    for i in range(len(num_layer)):
        log = np.loadtxt(f'figures/HierDNN/2023-01-09-3-5-64/log.csv', delimiter=',')
        # log = np.append(log, [[20000, log[-1, 1], 0, 0]], axis=0)
        plt.plot(log[:, 0], log[:, 1], '-.b', label='Residual learning')

        # log_termwise = np.loadtxt(f'figures/HierDNN/2023-01-08-4-5-64/loss_termwise.csv', delimiter=',')
        # log_termwise = np.append(log_termwise, [[log_termwise[-1, 0], log_termwise[-1, 1], log_termwise[-1, 2]]], axis=0)
        # plt.plot(log[:, 0], log_termwise[:, 1], '-.b', label='Residual learning')

        log = np.loadtxt(f'figures/HierDNN/2022-04-25-5-64/log.csv', delimiter=',')
        # plt.plot(log[:, 0], 0.25*log[:, 1], ':r', label='Proposed method (overall)')

        log_termwise = np.loadtxt(f'figures/HierDNN/2022-04-25-5-64/loss_termwise.csv', delimiter=',')
        plt.plot(log[:, 0], log_termwise[:, 1], '-r', label='Proposed method')

        plt.yscale('log')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('MSE for the training data')
        # plt.xlim(-800, 15800)
        plt.legend(loc='best')
        # plt.set_title(f'{num_layer[i]} layers, {hidden_size[i]} nodes')

    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_loss_comparison.png')
    plt.show()

def main2():
    # 学習中の評価関数の変化をプロット（複数回分）

    FIGURE_FOLDER = 'figures/HierDNN/residuals/sinc/'
    # FIGURE_FOLDER = 'figures/HierDNN/colds/sinc/'
    folder = lambda trial, num_layer: f'figures/HierDNN/2023-01-18-{trial}-{num_layer}-5-64'
    # folder = lambda trial, num_layer: f'figures/HierDNN/colds/sinc/layerwise/2023-01-18-{trial}-{num_layer}-5-64-cold'

    num_iter = 10
    num_layer = 3

    plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    for i in range(num_iter):
        log = np.loadtxt(f'{folder(i, num_layer)}/log.csv', delimiter=',')
        log_termwise = np.loadtxt(f'{folder(i, num_layer)}/loss_termwise.csv', delimiter=',')
        plt.plot(log[:, 0], log_termwise[:, num_layer-2], '-.b', label='Residual learning', alpha=0.3)

        log = np.loadtxt(f'figures/HierDNN/proposed/sinc/2023-01-24-{i}-5-64/log.csv', delimiter=',')
        log_termwise = np.loadtxt(f'figures/HierDNN/proposed/sinc/2023-01-24-{i}-5-64/loss_termwise.csv', delimiter=',')
        plt.plot(log[:, 0], log_termwise[:, num_layer-2], '-r', label='Proposed method', alpha=0.3)


    # log = np.loadtxt(f'figures/HierDNN/2022-04-25-5-64/log.csv', delimiter=',')
    # log_termwise = np.loadtxt(f'figures/HierDNN/2022-04-25-5-64/loss_termwise.csv', delimiter=',')
    # plt.plot(log[:, 0], log_termwise[:, num_layer-2], '-r', label='Proposed method')

    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('MSE for the training data')
    # plt.xlim(-800, 15800)
    # plt.legend(loc='best')

    plt.savefig(f'{FIGURE_FOLDER}fig_loss_comparison_{num_layer}.png')
    plt.savefig(f'{FIGURE_FOLDER}fig_loss_comparison_{num_layer}.pdf')
    plt.show()


def main():
    # 学習中の評価関数の変化をプロット

    # date = '2022-09-03'
    # FIGURE_FOLDER = './figures/HierDNN/' + date + '/'
    FIGURE_FOLDER = 'figures/HierDNN/2023-01-09-3-5-64/'

    num_layer = [5]
    # hidden_size = [64 ,37, 23, 15]

    # fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    for i in range(len(num_layer)):
        log = np.loadtxt(f'figures/HierDNN/2023-01-09-3-5-64/log.csv', delimiter=',')
        # log = np.append(log, [[20000, log[-1, 1], 0, 0]], axis=0)
        plt.plot(log[:, 0], log[:, 1], '-.b', label='Residual learning')

        # log_termwise = np.loadtxt(f'figures/HierDNN/2023-01-08-4-5-64/loss_termwise.csv', delimiter=',')
        # log_termwise = np.append(log_termwise, [[log_termwise[-1, 0], log_termwise[-1, 1], log_termwise[-1, 2]]], axis=0)
        # plt.plot(log[:, 0], log_termwise[:, 1], '-.b', label='Residual learning')

        log = np.loadtxt(f'figures/HierDNN/2022-04-25-5-64/log.csv', delimiter=',')
        # plt.plot(log[:, 0], 0.25*log[:, 1], ':r', label='Proposed method (overall)')

        log_termwise = np.loadtxt(f'figures/HierDNN/2022-04-25-5-64/loss_termwise.csv', delimiter=',')
        plt.plot(log[:, 0], log_termwise[:, 1], '-r', label='Proposed method')

        plt.yscale('log')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('MSE for the training data')
        # plt.xlim(-800, 15800)
        plt.legend(loc='best')
        # plt.set_title(f'{num_layer[i]} layers, {hidden_size[i]} nodes')

    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_loss_comparison.png')
    plt.show()

def main3():
    # 学習中の評価関数の変化をプロット（複数回分）

    # FIGURE_FOLDER = 'figures/HierDNN/residuals/sinc/'
    FIGURE_FOLDER = 'figures/HierDNN/colds/sinc/'
    # folder = lambda trial, num_layer: f'figures/HierDNN/2023-01-18-{trial}-{num_layer}-5-64'
    folder = lambda trial, num_layer: f'figures/HierDNN/colds/sinc/layerwise/2023-01-18-{trial}-{num_layer}-5-64-cold'

    num_iter = 10
    num_layer = 3

    plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    for i in range(num_iter):
        # log = np.loadtxt(f'{folder(i, num_layer)}/log.csv', delimiter=',')
        # log_termwise = np.loadtxt(f'{folder(i, num_layer)}/loss_termwise.csv', delimiter=',')
        # plt.plot(log[:, 0], log_termwise[:, num_layer-2], '-.b', label='Residual learning', alpha=0.3)

        log = np.loadtxt(f'figures/HierDNN/proposed/sinc/2023-01-24-{i}-5-64/log.csv', delimiter=',')
        log_termwise = np.loadtxt(f'figures/HierDNN/proposed/sinc/2023-01-24-{i}-5-64/loss_termwise.csv', delimiter=',')
        plt.plot(log[:, 0], log_termwise[:, num_layer-2], '-r', label='Proposed method', alpha=0.5)


        log = np.loadtxt(f'figures/HierDNN/proposed/2023-01-26-{i}-5-64/log.csv', delimiter=',')
        log_termwise = np.loadtxt(f'figures/HierDNN/proposed/2023-01-26-{i}-5-64/loss_termwise.csv', delimiter=',')
        plt.plot(log[:, 0], log_termwise[:, num_layer-2], '--g', label='Proposed method', alpha=0.5)

    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('MSE for the training data')
    # plt.xlim(-800, 15800)
    # plt.legend(loc='best')

    # plt.savefig(f'{FIGURE_FOLDER}fig_loss_comparison_{num_layer}.png')
    # plt.savefig(f'{FIGURE_FOLDER}fig_loss_comparison_{num_layer}.pdf')
    plt.show()


if __name__ == '__main__':
    main2()
    # main3()