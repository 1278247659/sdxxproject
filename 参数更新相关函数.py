import numpy as np

parmar={}
grads={}
# SGD函数
class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr

    def sgd(self):
        for key,val in parmar.items():
            parmar[key]=parmar[key]-self.lr*grads[key]

# momentem函数
class Momentem:
    def __init__(self,lr=0.01,a=0.9):
        self.lr=lr
        self.a=a
        self.v=None

    def momentem(self):
        if self.v==None:
            self.v={}
            for key,val in self.parmar.items():
               self.v[key]=np.zeros_like(val)

        for key in parmar.key():
            self.v[key]=self.v[key] * self.a -self.lr * grads[key]
            parmar[key]=parmar[key]+self.v[key]
        return parmar[key]

# adagrad函数
class Adagrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None

    def adagrad(self):
        if self.h == None:
            self.h={}
            for key,val in self.parmar.items():
                self.h[key]=np.zeros_like(val)

        for key in parmar.key():
            self.h[key] = self.h[key]+ grads[key] * grads[key]
            parmar[key]=parmar[key] - self.lr*(1/(self.h[key]+1e-7))*grads[key]
        return parmar[key]