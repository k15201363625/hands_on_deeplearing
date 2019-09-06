import numpy as np
import math
import matplotlib.pyplot as plt

class MaxPooling(object):
    def __init__(self,shape,ksize,stride):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.input_channels = shape[0]
        self.output_channels = shape[-1]
        # 生效位置索引
        self.index = np.zeros(shape)
        self.output_shape = [shape[0],shape[1]//ksize,shape[2]//ksize,shape[-1]]

    def forward_propagate(self,x):
        # calculate and record index
        shape = x.shape
        out = np.zeros([shape[0],shape[1]//self.stride,
                       shape[2]//self.stride,shape[-1]])
        self.index = np.zeros(self.input_shape)
        for b in range(shape[0]):
            for c in range(shape[-1]):
                for i in range(0,shape[1],self.stride):
                    for j in range(0,shape[2],self.stride):
                        if i+self.ksize > self.input_shape[1] or j+self.ksize > self.input_shape[1]:

                            continue
                        else:
                            out[b,i//self.stride,j//self.stride,c] = np.max(
                                    x[b,i:i+self.stride,j:j+self.stride,c])
                            idx = np.argmax(x[b,i:i+self.stride,j:j+self.stride,c])
                            self.index[b,i + idx//self.ksize,j + idx%self.ksize,c] += 1
        return out

    def backward_propagate(self,delta):
        # 进行扩充 之后与idx进行点乘
        repeated_delta = np.repeat(np.repeat(delta,self.stride,axis=1),self.stride,axis=2)
        # print(repeated_delta.shape) BUG:operands could not be broadcast together with shapes (32,6,6,128) (32,7,7,128)
        if repeated_delta.shape != self.input_shape:
            pad_x = self.input_shape[1]-repeated_delta.shape[1]
            pad_y = self.input_shape[2]-repeated_delta.shape[2]
            repeated_delta = np.pad(repeated_delta,((0,0),(0,pad_x),(0,pad_y),(0,0)),mode='constant',constant_values=0)
        return repeated_delta * self.index

class AveragePooling(object):
    def __init__(self,shape,ksize,stride):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.input_channels = shape[0]
        self.output_channels = shape[-1]
        self.output_shape = [shape[0],shape[1]//ksize,shape[2]//ksize,shape[-1]]

    def forward_propagate(self,x):
        # calculate and record index
        shape = x.shape
        out = np.zeros([shape[0],shape[1]//self.stride,
                       shape[2]//self.stride,shape[-1]])

        for b in range(shape[0]):
            for c in range(shape[-1]):
                for i in range(0,shape[1],self.stride):
                    for j in range(0,shape[2],self.stride):
                        if i+self.ksize > self.input_shape[1] or j+self.ksize > self.input_shape[1]:
                            continue
                        else:
                            out[b,i//self.stride,j//self.stride,c] = np.mean(
                                    x[b,i:i+self.stride,j:j+self.stride,c])

        return out

    def backward_propagate(self,delta):
        # 进行扩充 之后与idx进行点乘
        repeated_delta = np.repeat(np.repeat(delta,self.stride,axis=1),self.stride,axis=2)
        return repeated_delta / (self.ksize*self.ksize)


