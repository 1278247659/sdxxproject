import numpy as np

# 定义均方误差函数
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

# 定义交叉熵误差
def cross_entropy_error(y,t):
    
    delta=1e-7
    return -np.sum(t * np.log(y++delta))

print(-np.log(0.1))
print(cross_entropy_error(0.1,1))
print(mean_squared_error(0.6,1))


