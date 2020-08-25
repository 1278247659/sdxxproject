import numpy as np

def softmax(a):
    c=np.max(a)#防止溢出
    exp_a=np.exp(a-c)#防止溢出
    sum_exp_a=np.sum(exp_a)
    y=exp_a / sum_exp_a
    return y

a=np.array([0.3,2.9,4.0])
b=softmax(a)
print(b)

c=np.array([1010,1000,990])
b=softmax(c)
print(b)