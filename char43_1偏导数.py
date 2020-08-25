import numpy as np
from matplotlib.pylab import plt


def num_diff(f,x):#求导数
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

def fun(x):
    return 0.01*x**2+0.1*x
# x在x=5和x=10处的导数
print(num_diff(fun,5))
print(num_diff(fun,10))

def fun2(x):
    return x[0]**2+x[1]**2

# 求x0^2+x1^2的偏导
def pd1(x):
    # x1=4时的偏导
    x1=4
    return x**2+x1**2
print(num_diff(pd1,3))

# 同时求偏导

def grad_pd(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    print(x.size)
    for idx in range(x.size):
        temp=x[idx]
        #求f(x+h)
        x[idx]=temp+h
        fxh1=f(x)
        #求f(x-h)
        x[idx]=temp-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=temp#还原值
    return grad
print('下降法求梯度:%s'%grad_pd(fun2,np.array([3.0,4.0])))

# 梯度下降法求最小值

def min_grad(f,x,lr=0.01,step=100):
    for i in range(step):
        # 求x0，x1的偏导
        grad=grad_pd(f,x)
        x -=lr*grad
    return x

print(min_grad(fun2,np.array([-3.0,4.0]),lr=0.1,step=100))


# 神经网络的梯度
from common.functions import cross_entropy_error,softmax
from common.gradient import numerical_gradient

class Simple:
    def __init__(self):
        self.W=np.random.rand(2,3)

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(x)
        loss=cross_entropy_error(y,t)

        return loss

net=Simple()
x=np.array([0.6,0.9])
t=np.array([0,0,1])
print('神经网络求的值为：%s'%net.predict(x))
print('神经网络的损失函数值为：%s'%net.loss(x,t))

def f(W):
    return net.loss(x,t)
print(net.W)

print(numerical_gradient(f,net.W))