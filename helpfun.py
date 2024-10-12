import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math, time
from random import random

""" Вспомогательные функции для тервера """

def check(B, S):
    """
    Проверяет достаточное условие, что нейроны подсистемы S и только они
    спайкают, уталкивая остальные нейроны на бесконечность.
    Выдаёт теоретическую частоту спайков a и flag = True, если выполнен 
    достаточный признак, что не уталкивает, и False иначе
    """
    S.sort()
    k = len(S)
    n = len(B[0])
    if k == 0 or k == n:
        return [], True
    BS = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            BS[i][j] = B[S[i]][S[j]]
    f = np.ones((k, 1))
    a = np.transpose(LA.inv(BS)) @ f
    pulse = 0
    flag1 = True
    flag2 = True
    for i in range(k):
        if a[i] <= 0:
            flag1 = False
    for i in S:
        for j in range(n):
            if j not in S:
                pulse += a[S.index(i)] * B[i][j]
    if pulse >= n-k:
        flag2 = False
    # print(S)
    # print(flag1, flag2)
    return a, flag1 and flag2, flag1, flag2

         
def enough_condition(B, tell = 0, give = 0):
    """
    Получает на вход матрицу спайков B и необязательные tell (=0) и give (=0), выдаёт booling, выполнено ли
    достаточное условие для отсутствия стабильной собственной подсистемы.
    Если tell == 1, пишет подсистему S, на которой условие нарушается.
    Если give == 1, кроме того, выдаёт S.
    """
    n = len(B[0])
    for j in range(2**n):
        sbin = bin(j)[2:]
        if len(sbin) < n:
            sbin = '0'*(n-len(sbin)) + sbin
        S = []
        for i in range(n):
            if sbin[i] == '1':
                S.append(i)
        print(S)
        if not check(B, S)[1]:
            if tell == 1:
                print(S)
            # if give == 1:
                # return False, S
            # return False
    return True



def proper_test(B, N):
    """
    Получает на вход матрицу спайков B и число спайков N в тесте,
    где far_neurons = [], хотя enough_condition(B). Проводит тест с числом
    спайков 2*N, выдаёт: bool о том, остаётся ли far_neurons пустым
    """
    flag = enough_condition(B)
    if flag:
        if N >= 10**7:
            print('Too much N')
            return False
    else:
        if N % 3**7 != 0:
            return proper_test(B, N*3**7)
        else:
            if N >= 10**7:
                print('Too much N')
                return False
    # print(N)
    n = len(B[0])
    N *= 2
    neurons_global = np.zeros((n, N))
    for i in range(n):
        neurons_global[i][0] = 0.01 * i
    for k in range(1, N):   # моделируем одну прогонку
        neurons = [neurons_global[i][k-1] for i in range(n)]
        m = min(neurons)
        spike = neurons.index(m)
        for i in range(n):
            neurons_global[i][k] = neurons[i] - m + B[spike][i]
    far_neurons = []    # убежавшие нейроны
    for i in range(n):
        max_first = max(neurons_global[i, 0:N//2])      # максимум в первой N-ке
        max_last = max(neurons_global[i, N//2:N])   # максимум в последней N-ке
        if max_last > 1.5*max_first:
            far_neurons.append(i)
        else:
            if max_last > max_first:
                return proper_test(B, N)
    return far_neurons == []



