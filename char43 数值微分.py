# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:36:21 2020

@author: Administrator
"""

import numpy as np
from matplotlib.pylab import plt

#一般的数值积分
def diff(f,x):
    h=1e-50
    return (f(x+h)-f(x))/h

#中心差分的数值健微分
def center_diff(f,x):
    h=1e-50
    return (f(x+h)-f(x-h))/(2*h)


def f1(x):
    return 0.01*x**2+0.1*x

print(diff(f1,10))

a=np.arange(0,20,0.1)
y=f1(a)
plt.xlabel('x')
plt.ylabel('f1')
plt.plot(a,y)
plt.plot(5,f1(5),'ro')
plt.plot(10,f1(10),'bo')
plt.show()


