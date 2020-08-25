import matplotlib.pyplot as plt
import numpy as np
#  定义一个sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
# 定义一个跃迁函数
def foot(x):
    y = x > 0
    return np.array(x > 0,dtype=np.int)
# ReLU函数
def ReLU(x):
    return np.maximum(0,x)

x=np.arange(-5,5,0.1)
y=foot(x)
y1=sigmoid(x)
y2=ReLU(x)
plt.ylim(0,2)
plt.plot(x,y1,'g-')
plt.plot(x,y,'r-')
plt.plot(x,y2,'b-')
plt.show()
