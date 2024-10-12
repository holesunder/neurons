
""" Архив со всякой фигнёй, которая уже давно не нужна, но на всякий случай оставлю"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from math import e, sqrt, log
from random import random
from helpfun import check, B_random, enough_condition, proper_test


""" Один тест со случайной матрицей спайков """
n = 3       # количество нейронов
N = 1000   # количество спайков
nu = 0.01  # снос за единицу времени


# генерируем случайную матрицу B с условием 2: bii = Hi, bij = wi, Hi > wi
# также заводим соответствующие векторы H, w

# B, H, w = B_random(n, 1)
B = B_random(n)

sneslo = 0
z_0 = [2*nu]*n   # точка старта 
neurons_global = np.zeros((n, N))   
spikes_count = [0]*n
for i in range(n):
    neurons_global[i][0] = z_0[i]
neurons_prev = [neurons_global[i][0] for i in range(n)]
for k in range(1, N):
    neurons = neurons_prev
    m = min(neurons)
    sneslo += m
    spike = neurons.index(m)
    for i in range(n):
        neurons_global[i][k] = neurons[i] - m
        neurons_prev[i] = neurons[i] - m + B[spike][i]


for i in range(n):
    plt.plot(neurons_global[i])
    plt.show()


print(enough_condition(B))




""" Один тест с заданной матрицей спайков (не оптимизирован, счёт по времени)"""
n = 3    # количество нейронов
T = 100000   # количество итераций по времени
nu = 0.1   # снос за единицу времени
B = np.zeros((n, n))    # матрица спайков
z_0 = np.zeros((n, 1))
B = np.array(
    [[0.64114225, 0.58834016, 0.0406359 , 0.54789385, 0.11494609, 0.28896509],
     [0.23135646, 0.70716916, 0.666855  , 0.04759596, 0.25785353, 0.55063766],
     [0.28053612, 0.35491815, 0.38418306, 0.06338342, 0.18886781, 0.2219791 ],
     [0.6961898,  0.27365859, 0.31703293, 0.73658251, 0.16908771, 0.59755286],
     [0.30761545, 0.35236285, 0.42993529, 0.64132523, 0.79580742, 0.04220529],
     [0.32795584, 0.34848779, 0.27718139, 0.38498218, 0.46498877, 0.47423947]]
    )



B = np.array([[8, 7, 7],
      [8, 9, 8],
      [4, 4, 5]])

z_0 = [2*nu]*n   # точка старта
neurons_global = np.zeros((n, T))   
spikes_count = [0]*n
for i in range(n):
    neurons_global[i][0] = z_0[i]
for t in range(1, T):
    neurons = [neurons_global[i][t-1] - nu for i in range(n)]   # потенциалы после сноса
    m = min(neurons)
    if m <= 0:  # спайк
        spike = neurons.index(m)
        spikes_count[spike] += 1
        for i in range(n):
            neurons[i] += B[spike][i]
    for i in range(n):
        neurons_global[i][t] = neurons[i]
        
for i in range(n):  # графики потенциалов
    plt.plot(neurons_global[i])
    plt.show()

spikes_freq = [spikes_count[i]/T for i in range(n)]     # частоты спайков практические


# print('Частота теоретическая =', a)
print('Частота практическая =', spikes_freq)

# neurons_tilda = np.zeros((n, T))
# neurons_tilda[0] = neurons_global[0] + neurons_global[1]
# neurons_tilda[1] = neurons_global[0] - neurons_global[1]
# neurons_tilda[2] = neurons_global[2]

# for i in range(n):
#     plt.plot(neurons_tilda[i])
#     plt.show()


""" Один тест с заданной матрицей спайков (оптимизирован, счёт по числу спайков) """
N = 10000   # количество спайков
nu = 0.01   # снос за единицу времени

# матрица спайков:
B = np.array(
    [[0.75, 0.1, 0.5, 0.3], [0.06, 0.1, 0.06, 0.07], [0.1, 0.29, 0.3, 0.13, ], [0.2, 0.5, 0.2, 0.6]]



)
n = len(B[0])   # количество нейронов

z_0 = np.zeros((n, 1))

sneslo = 0
z_0 = [2*nu]*n   # точка старта
neurons_global = np.zeros((n, N))   
spikes_count = [0]*n
for i in range(n):
    neurons_global[i][0] = z_0[i]
neurons_prev = [neurons_global[i][0] for i in range(n)]
for k in range(1, N):
    neurons = neurons_prev
    m = min(neurons)
    sneslo += m
    spike = neurons.index(m)
    for i in range(n):
        neurons_global[i][k] = neurons[i] - m
        neurons_prev[i] = neurons[i] - m + B[spike][i]
        
for i in range(n):  # графики потенциалов
    plt.plot(neurons_global[i])
    plt.show()

print(sneslo/nu)
print(enough_condition(B, 1))





""" Проверка выполнимости достаточного условия, чтобы не было подсистемы,
уталкивающей остальные нейроны на бесконечность """
n = 10
number_of_tests = 1000
count = 0
tt, tf, ft, ff = 0, 0, 0, 0
for k in range(number_of_tests):
    B = B_random(n)
    if not enough_condition(B):
        count += 1
        S = enough_condition(B, 0, 1)[1]
        f1, f2 = check(B,S)[2], check(B,S)[3]
        tt += f1*f2
        tf += f1*(not f2)
        ft += (not f1)*f2
        ff += (not f1)*(not f2)

print(count/number_of_tests)
print('True True', tt/count)
print('True False', tf/count)
print('False True', ft/count)
print('False False', ff/count)





""" Проверка необходимости того достаточного условия """

n = 4   # количество нейронов
N = 100   # количество спайков
nu = 0.01  # снос за единицу времени
number_of_test = 1 # количество тестов
z_0 = [2*nu]*n      # старт
number_of_far_neurons = []*number_of_test    # количество убежавших нейронов в каждом тесте
for k in range(number_of_test):
    # генерируем случайную матрицу спайков
    # B, H, w = B_random(n, 1)
    B = B_random(n)
    neurons_global = np.zeros((n, N))   
    for k in range(1, N):   # моделируем одну прогонку
        neurons = [neurons_global[i][k-1] for i in range(n)]
        m = min(neurons)
        spike = neurons.index(m)
        for i in range(n):
            neurons_global[i][k] = neurons[i] - m + B[spike][i]
    far_neurons = []    # убежавшие нейроны
    for i in range(n):
        if neurons_global[i][N-1] > 1.5*max(neurons_global[i, 0:N//2]):
            far_neurons.append(i)
    number_of_far_neurons.append(len(far_neurons))
    flag = enough_condition(B)
    if len(far_neurons) != 0 and flag:   # если enough_cond, но есть far_neurons
        if not proper_test(B, N):
            print(flag, far_neurons)
            print(', '.join(str(B).split()))
            for i in range(n):
                plt.plot(neurons_global[i])
                plt.show()
            print()
     
    if len(far_neurons) == 0 and not flag:   # если не enough_cond и нету far_neurons
        if proper_test(B, N):
            print(flag)
            print(', '.join(str(B).split()))
            for i in range(n):
                plt.plot(neurons_global[i])
                plt.show()
            print()





""" Оценочка для какого-то доказательства """
a = 0.4
H = 1e0
n = 2

def summ(t):
    s1 = 0
    k = 1
    s = e**((-k**2)/(2*t))/k
    while (s - s1)/log(t) > 1e-11:
        k+=1
        s1 = s
        s += e**((-k**2*H**2)/(2*t))/k/H
        if k > 10**10:
            print('for t = ', t, '- Govno')
            break
    print(k)
    return s / log(t)
start = time.time()
t = [1e300*i for i in range(1, n)]
summ = [summ(t[i]) for i in range(n-1)]
print(summ[n-2])
plt.plot(t,summ)
plt.show()

