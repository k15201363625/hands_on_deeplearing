import numpy as np
import matplotlib.pyplot as plt
import math

class FullyConnectedLayer(object):
    def __init__(self,shape,output_size):
        self.input_shape = shape
        self.batch_size = shape[0]
        self.output_shape = [self.batch_size,output_size]
        self.output_size = output_size

        num = 1
        for i in shape[1:]:
            num *= i
        weight_scaler = np.sqrt(num/2)
        self.weights = np.random.randn(num,output_size)/weight_scaler
        self.bias = np.random.randn(output_size)/weight_scaler

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)
    def forward_propagate(self,x):
        self.x = x.reshape([self.batch_size,-1])
        return np.dot(self.x,self.weights) + self.bias

    def gradient_cal(self,delta):
        # 对于w_grad 对于batch中的每一个sample进行反向传播之后sum
        for i in range(delta.shape[0]):
            x_i = self.x[i][:,np.newaxis]
            delta_i = delta[i][:,np.newaxis].T
            self.w_grad += np.dot(x_i,delta_i)
            self.b_grad += np.reshape(delta_i,self.bias.shape)
        # 如果对于整体进行快速处理 可以使用以下code
        delta_transposed = np.transpose(delta[...,np.newaxis],[0,2,1])
        x_extend = self.x[...,np.newaxis]
        # w_grad_everybatch = np.dot(x_extend,delta_transposed) # 默认忽略第一维度 进行运算
        w_grad_everybatch = np.matmul(x_extend,delta_transposed) # 默认忽略第一维度 进行运算
        # print(x_extend.shape,delta_transposed.shape,w_grad_everybatch.shape)
        w_grad = np.sum(w_grad_everybatch,axis=0)
        b_grad = np.sum(delta,axis=0)
        # demo fpr consistence of calculation
        # print(w_grad-self.w_grad)
        # print(b_grad-self.b_grad)

        # input_delta
        input_delta = np.dot(delta,self.weights.T).reshape(self.input_shape)
        return input_delta

    def backward_propagate(self,delta,learning_rate=1e-3,weight_decay=1e-3):
        # zero gradient
        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

        input_delta = self.gradient_cal(delta)

        # print('max grad of fc layer:',np.max(self.b_grad),np.max(self.w_grad))

        # use weight decay -> l2 regularization
        self.weights *= (1-weight_decay)
        self.bias *= (1-weight_decay)
        self.weights -= learning_rate * self.w_grad
        self.bias -= learning_rate * self.b_grad
        return input_delta


