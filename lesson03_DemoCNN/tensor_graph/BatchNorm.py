import numpy as np
import math
from tensor_graph.Operator import Operator
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE

class BatchNorm(Operator):
    def __init__(self,input_variable=Variable,name=str,scope='',epsilon=1e-4,moving_decay=1-3e-3):
        self.input_variable = input_variable
        self.shape = input_variable.shape
        self.batch_size = self.shape[0]
        self.bndims = tuple([i for i in range(len(self.shape)-1)])
        self.output_variable = Variable(self.shape,name='out',scope=name)

        # default BN for final dim
        self.alpha = Variable([self.shape[-1]],name='alpha',scope=name,grad=True,trainable=True,init='ones')
        self.beta =  Variable([self.shape[-1]],name='beta',scope=name,grad=True,trainable=True,init='zeros')

        self.moving_mean = np.zeros(self.shape[-1])
        self.moving_var = np.zeros(self.shape[-1])

        self.moving_decay = moving_decay
        self.epsilon = epsilon

        Operator.__init__(self,[self.input_variable],[self.output_variable],name,scope)

    def forward(self,state='test'):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.wait_forward = False
            self._BN(self.input_variable,self.output_variable,state)


    def _BN(self,input_variable=Variable,output_variable=Variable,state='test'):
        self.mean = np.mean(input_variable.data,axis=self.bndims)
        # var 对于第一个dim很敏感 如果>1 需要使用样本方差公式进行修正
        if self.batch_size == 1:
            self.var = np.var(input_variable.data,axis=self.bndims)
        else:
            self.var = self.batch_size / (self.batch_size - 1) * (np.var(input_variable.data,axis=self.bndims))

        # calc moving mean and var
        if np.sum(self.moving_mean) == 0 and np.sum(self.moving_var==0):
            self.moving_mean = self.mean
            self.moving_var = self.var
        else:
            # update mean var (as momentum)
            self.moving_mean = self.moving_decay*self.moving_mean + (1-self.moving_decay)*self.mean
            self.moving_var = self.moving_decay*self.moving_var + (1-self.moving_decay)*self.var

        if state == 'train':
            self.bned_x = (self.input_variable.data - self.mean)/np.sqrt(self.var+self.epsilon)
        elif state == 'test':
            self.bned_x = (self.input_variable.data - self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)
        else:
            raise AttributeError('error state to BN forward op')
        self.output_variable.data = self.bned_x*self.alpha.data + self.beta.data

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.wait_forward = True
            # 计算梯度(权重梯度)
            self.alpha.diff = np.sum(self.output_variable.diff*self.bned_x,
                                     axis=self.bndims)
            self.beta.diff = np.sum(self.output_variable.diff,axis=self.bndims)

            # help to calc alpha_grad and beta_grad
            bned_x_grad = self.output_variable.data * self.alpha.data
            var_grad = np.sum(-1.0 / 2 * bned_x_grad * (self.input_variable.data - self.mean) / (self.var + self.epsilon) ** (3.0 / 2),
                                  axis=self.bndims)
            mean_grad = np.sum(-1 / np.sqrt(self.var + self.epsilon) * bned_x_grad, axis=self.bndims)
            # 计算回传梯度
            x_grad = bned_x_grad * np.sqrt(self.var + self.epsilon) + 2 * (self.input_variable.data - self.mean) * var_grad / self.batch_size + mean_grad / self.batch_size
            self.input_variable.diff = x_grad
            #　有关更新参数的过程　在ｇｒａｐｈ中统一是现在了variable的方法中　
            #　所以在operator类型中只需要计算梯度　以及输入输出数据
