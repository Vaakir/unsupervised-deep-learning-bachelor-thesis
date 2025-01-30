# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:51:59 2025

@author: kiran
"""

import numpy as np

import matplotlib.pyplot as plt

np.random.seed(1)
n = 100
x = np.arange(0, n)
y = np.random.normal(1, 0.1, n) + np.sin(x/5)


w = 3
tops = []
for i in range(w, len(x)):
    #print(i, y[i], y[i-w : i+w+1])
    if y[i] == max(y[i-w:i+w+1]):
        tops.append(i)

Tp = np.mean(np.diff(np.array(tops)))

print(tops, Tp)
# 2PI*freq (freq=1/tp)

plt.plot(x,y)
