import numpy as np
# 矩阵乘积实现

A=np.array([[1,2],[2,3]])
B=np.array([[2,2],[4,2]])
C=np.dot(A,B)#点积
print("C=%s"%C)

A1=np.array([[1,2],[2,3],[2,4]])
B1=np.array([[2,2,4],[1,2,4]])
C1=np.dot(A1,B1)#点积
print("C1=%s"%C1)