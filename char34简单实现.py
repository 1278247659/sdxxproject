import numpy as np
from char31 import sigmoid

# first
x=np.array([[1,0.5]])
w=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
b=np.array([0.1,0.2,0.3])
a1=np.dot(x,w)+b
z1=sigmoid(a1)

# second
w2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
b2=np.array([0.1,0.2])
a2=np.dot(z1,w2)+b2
z2=sigmoid(a2)

def function(x):
    return x

w3=np.array([[0.1,0.3],[0.2,0.4]])
b3=np.array([0.1,0.2])

a3=np.dot(z2,w3)+b3
z3=function(a3)
print(z3)