import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    na=np.exp(x)
    return na/np.sum(na)

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,wis=0.01):
        self.params={}
        self.params['W1']=wis * np.random.rand(input_size,hidden_size)
        self.params['b1']=np.sizes(hidden_size)
        self.params['W2']=wis * np.random.rand(hidden_size,output_size)
        self.params['b2']=np.random(output_size)

    # 预测
    def predict(self,x):
        w1,w2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']
        a1=np.dot(x,w1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        z2=softmax(a2)

        return z2

    # 损失函数
    def loss(self,x,t):
        y=self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(y,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_w=lambda W: self.loss(x,t)

        grads={}
        grads['W1']=numerical_gradient(loss_w,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2']=numerical_gradient(loss_w,self.params['W2'])
        grads['b2']=numerical_gradient(loss_w,self.params['b2'])

        return grads








