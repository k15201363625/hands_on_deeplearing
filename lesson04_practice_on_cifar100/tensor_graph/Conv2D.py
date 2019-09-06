import numpy as np
import math
from tensor_graph.Operator import Operator
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE
from tensor_graph.utils import img2col

class Conv2D(Operator):

    def __init__(self, kernel_shape=list, input_variable=Variable, name=str, scope='', stride=1, padding='SAME'):
        """
        conv layer -> Operator
        :param kernel_shape:
        :param input_variable: only one and shape must be 4 dim
        get input variable and output variable to call Operator.__init__
        kernel params save as Variable(trainable=True)
        scope name of all variables is self.name
        """
        # kernel_shape = [ksize, ksize, input_channels, output_channels]
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator Conv2D name: %s kernel shape is not list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)

        if len(input_variable.shape)!=4:
            raise Exception("Operator Conv2D name: %s's input_variable's shape != 4d Variable!" % name)

        self.ksize = kernel_shape[0]
        self.stride = stride
        self.output_num = kernel_shape[-1]
        self.padding = padding

        self.col_image = []

        self.weights = Variable(kernel_shape, scope=name, name='weights',grad=True, trainable=True)
        self.bias = Variable([self.output_num], scope=name, name='bias', grad=True, trainable=True)

        self.batch_size = input_variable.shape[0]
        # calc output variable shape
        if self.padding == 'SAME':
            output_shape = [self.batch_size, input_variable.shape[1]//stride, input_variable.shape[2]//stride,
                             self.output_num]
        if self.padding == 'VALID':
            output_shape = [self.batch_size, (input_variable.shape[1] - self.ksize + 1)//stride,
                             (input_variable.shape[2] - self.ksize + 1)//stride, self.output_num]
        else:
            raise AttributeError('unsupported padding method')

        self.input_variables = input_variable
        self.output_variables = Variable(output_shape, name='out', scope=name)  # .name

        Operator.__init__(self, [self.input_variables], [self.output_variables],name,scope)

    def forward(self):
        """complete as needed by template"""
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.wait_forward = False
            self.conv(self.input_variables, self.output_variables, self.weights, self.bias, self.ksize, self.stride)
        else:
            pass

    def backward(self):
        """complete as needed by template"""
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.deconv(self.input_variables, self.output_variables, self.weights, self.bias)
            self.wait_forward = True


    def conv(self, input=Variable, output=Variable, weights=Variable, bias=Variable ,ksize=int ,stride=int):
        """
        weights and bias need not be Variable,because need not change value of input params
        convolution operation
        support outer call
        get input.data and output.data.shape -> calc output.data
        """

        # padding input_img according to method
        if self.padding == 'SAME':
            batch_img = np.pad(input.data,((0, 0),(ksize//2,ksize//2),(ksize//2,ksize//2),(0, 0)),
                               mode='constant', constant_values=0)
        else:
            batch_img = input.data

        # reshape weights to col
        col_weights = weights.data.reshape(-1, self.output_num)
        '''
        # through maxtrix op and for loop to each batch 
        conv_out = []
        self.col_image = []
        # do dot for every image in batch by img2col dot col_weight
        for i in range(self.batch_size):
            img_i = batch_img[i][np.newaxis, :]
            col_image_i = img2col(img_i, ksize, stride)
            out = np.reshape(np.dot(col_image_i, col_weights) + bias, output.data[0].shape)
            self.col_image.append(col_image_i)
            conv_out.append(out)
        self.col_image = np.array(self.col_image)
        conv_out = np.array(conv_out)
        '''

        # through matrix operation is faster than for loop operation on each batch (统一处理batch)
        self.col_image = np.array([img2col(batch_img[i][np.newaxis,:],ksize,stride) for i in range(self.batch_size)])
        # print(self.col_image.shape,col_weights.shape,bias.shape)
        conv_out = np.reshape(np.dot(self.col_image,col_weights) + bias.data, output.shape)

        output.data = conv_out

    def deconv(self, input=Variable, output=Variable, weights=Variable, bias=Variable):
        """
        weights and bias type should be Variable,because need to change value of input param (rather than quotation)
        deconvolution operation
        support outer call
        get weights grad and backward propagate delta to input of current layer
        -> weights.diff bias.diff and input.diff
        """
        # 1. weights grad
        col_delta = np.reshape(output.diff, [self.batch_size, -1, self.output_num])
        for i in range(self.batch_size):
            # sum in batch
            weights.diff += np.dot(self.col_image[i].T, col_delta[i]).reshape(self.weights.shape)
        bias.diff += np.sum(col_delta, axis=(0, 1))
        # 2. input grad
        # deconv of padded delta with flippd kernel to get input_delta
        if self.padding == 'VALID':
            padded_delta = np.pad(output.diff,((0,0),(self.ksize-1,self.ksize-1),(self.ksize-1,self.ksize-1),(0,0)),
                             mode='constant', constant_values=0)

        if self.padding == 'SAME':
            padded_delta = np.pad(output.diff,((0,0),(self.ksize//2,self.ksize//2),(self.ksize//2,self.ksize//2),(0, 0)),
                             mode='constant', constant_values=0)

        # img2col
        col_padded_delta = [img2col(padded_delta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)]
        col_padded_delta = np.array(col_padded_delta)

        # flip weights
        flip_weights = np.flipud(np.fliplr(weights.data))
        # [...,inputchannel,outputchannel] - > [...,outputchannel,inputchannel]
        flip_weights = flip_weights.transpose((0,1,3,2))
        col_flip_weights = flip_weights.reshape([-1, weights.shape[2]])

        
        input.diff = np.reshape(np.dot(col_padded_delta, col_flip_weights),input.shape)


