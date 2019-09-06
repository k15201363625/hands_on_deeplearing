import numpy as np
import matplotlib.pyplot as plt
import math

class Conv2D(object):
    def __init__(self,input_shape,output_channels,ksize,stride=1,padding='same'):
        self.input_shape = input_shape
        self.input_channels = input_shape[-1]
        self.output_channels = output_channels
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        # default batch_size = shape[0]
        self.batch_size = input_shape[0]

        # initialize params with standardnormal + scaler
        # msra方法设置scaler
        weight_scaler = math.sqrt(ksize**2*self.input_channels/2)
        self.filter = np.random.randn(self.output_channels,ksize,ksize,self.input_channels)/weight_scaler
        self.bias = np.random.randn(self.output_channels)/weight_scaler

        # back propagation params
        shape = self.input_shape
        if padding == 'valid':
            self.delta = np.zeros((shape[0],(shape[1]-ksize+1)//self.stride,
                                  (shape[2]-ksize+1)//self.stride,self.output_channels))
        elif padding == 'same':
            self.delta = np.zeros((shape[0],shape[1]//self.stride,
                                   shape[2]//self.stride,self.output_channels))
        else:
            raise AttributeError('unsupported padding method str')
        self.output_shape = self.delta.shape
        # print(self.output_shape)
        self.filter_grad = np.zeros(self.filter.shape)
        self.bias_grad = np.zeros(self.bias.shape)
    def _conv(self,x):
        """
        faster convolution with matrix method than naive dot product(element-wise)
        four steps:
        1.reshape filter
        2.padding
        3.im2col save image
        4.matmul(input_features,kernel_matrix)=output_features
        """
        # shape=(self.output_channels,ksize,ksize,self.input_channels)
        col_filter = np.transpose(self.filter,[1,2,3,0])
        col_filter = col_filter.reshape([-1,self.output_channels])
        if self.padding == 'same':
            x = np.pad(x,((0,0),(self.ksize//2,self.ksize//2),(self.ksize//2,self.ksize//2),(0,0)),
                      mode='constant',constant_values = 0)
        # 整个batch一起处理
        #self.img_cols = self._img2col(x)

        # 每个sample in batch 分别处理
        self.img_cols = []
        self.conv_out = []
        for i in range(self.batch_size):
            img_i = x[i][np.newaxis,:] # 保障4dim
            nowcol = self._img2col(img_i,self.ksize,self.stride)
            self.img_cols.append(nowcol)
            self.conv_out.append(np.reshape(
                np.dot(nowcol,col_filter)+self.bias,
                self.delta[0].shape))

        self.img_cols = np.array(self.img_cols)
        self.conv_out = np.array(self.conv_out)
        return self.conv_out

    def _img2col(self,image,ksize,stride):
        """tool function of transform img-> matrix
        only to one sample
        otherwise:reshape 等操作需要改变处理方式
        """
        # image is a 4d tensor([batchsize, width ,height, channel])
        img_cols = []
        for i in range(0,image.shape[1]-ksize+1,stride):
            for j in range(0,image.shape[2]-ksize+1,stride):
                nowcol = image[:,i:i+ksize,j:j+ksize,:].reshape([-1])
                img_cols.append(nowcol)
        img_cols = np.array(img_cols)
        return img_cols

    def forward_propagate(self,x):
        return self._conv(x)

    def _dconv(self):
        """
        deconv of padded eta with flippd kernel to get next_eta
        """
        if self.padding == 'valid':
            pad_delta = np.pad(self.delta,
                              ((0,0),(self.ksize-1,self.ksize-1),(self.ksize-1,self.ksize-1),(0,0)),
                              mode='constant',constant_values=0)

        elif self.padding == 'same':
            pad_delta = np.pad(self.delta,
                              ((0,0),(self.ksize//2,self.ksize//2),(self.ksize//2,self.ksize//2),(0,0)),
                              mode='constant',constant_values=0)
        # only to 0,1 dims (fliplr,flipud)
        # 使用swapaxes与transpose类似功能 but只能交换两个维度
        # (kszie,ksize,output_channels,input_channels)
        flipped_filter = np.transpose(self.filter,[1,2,0,3])
        flipped_filter = np.fliplr(np.flipud(flipped_filter))
        col_flipped_filter = flipped_filter.reshape([-1,self.input_channels])
        # delta img2col with ** list generator **
        col_pad_delta = np.array(
            [self._img2col(pad_delta[i][np.newaxis,:],
                           self.ksize,self.stride) for i in range(self.batch_size)])
        # dconv (matmul)
        input_delta = np.dot(col_pad_delta,col_flipped_filter)
        # 直接reshape就可以实现 因为已经分开batch处理了
        input_delta = input_delta.reshape(self.input_shape)
        return input_delta

    def update_params(self,learning_rate=1e-3, weight_decay=1e-3):
        """
        use L2 regularization
        并且使用了新的更新方式 ： simple sgd + weight decay
        """
        # weight_decay = L2 regularization
        self.filter *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.filter -= learning_rate * self.filter_grad
        self.bias -= learning_rate * self.bias_grad

    def backward_propagate(self,delta,learning_rate=1e-5, weight_decay=1e-4):
        """
        deconv of padded delta with flippd kernel to get next_delta
        four steps:
        1.对输入的delta根据method进行padding操作。
        2.对卷积核参数进行翻转。
        3.对卷积核进行转置操作。(小心 numpy运行的未必是想象的结果)
        4.对翻转后的卷积核与padding后的pad_delta进行卷积计算，方法同conv.forward
        """
        self.delta = delta
        # reshape
        col_delta = np.reshape(self.delta,
                              [self.batch_size,-1,self.output_channels])

        # 还原 保障
        self.filter_grad = np.zeros(self.filter.shape)
        self.bias_grad = np.zeros(self.bias.shape)

        # weights gradient update
        for i in range(self.batch_size):
            self.filter_grad += np.dot(self.img_cols[i].T,
                                       col_delta[i]).reshape(self.filter.shape)
        self.bias_grad+=np.sum(col_delta,axis=(0,1))

        # print('max grad of conv layer:',np.max(self.bias_grad),np.max(self.filter_grad))
        self.update_params(learning_rate,weight_decay)


        # dconv to backpropagation grad
        return self._dconv()


