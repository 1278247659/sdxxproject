# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:06:10 2020

@author: Administrator
"""
import os,sys
sys.path.append(os.pardir)
from char3.mnist import load_mnist
import numpy as np
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

size = x_train.shape[0]#数据总量为6W个
train = 10
a=np.random.choice(size,train)
x_batch = x_train[a]
t_batch = t_train[a]


def cross_entrpy_error(y,t):#单个数据的时输入
    if y.ndim == 1:#一维
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
        
    batch_size=y.shape[0]
    return -np.sum(t * np.log(y + 1e-7))/batch_size