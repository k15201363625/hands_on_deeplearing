import numpy as np
import math
from functools import reduce


def initializer(shape,method='std_const'):
    if method == 'zeros':
        return np.zeros(shape)
    elif method == 'ones':
        return np.ones(shape)
    elif method == 'std_const':
        return np.random.standard_normal(shape) / 1e2
    elif method == 'MSRA':
        # 使用msra 初始化方法 讲shape的[:-1]唯独乘积开方得到scale
        # 之后用standard normal * 1/scale 得到initialization value
        scale = math.sqrt(reduce(lambda x,y:x*y,shape))/shape[-1]
        return np.random.standard_normal(shape) / scale
    else:
        raise AttributeError('unsupported initial method')

def learning_rate_exponential_decay(learning_rate,global_step,
                                    decay_rate=1e-1,decay_steps=1000):
    """
    Applies exponential decay to learning rate
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step/decay_steps)
    :return: learning_rate with exponential decay
    """
    return learning_rate * pow(decay_rate,float(global_step/decay_steps))


def img2col(img,ksize,stride):
    """
    :param img is a 4d tensor([batchsize, width ,height, channel])
    Note: now batchsize must be 1
    be used to fast conv operation (matrix formation)
    """
    img_col = []
    for i in range(0,img.shape[1]-ksize+1,stride):
        for j in range(0,img.shape[2]-ksize+1,stride):
            nowcol = img[:,i:i+ksize,j:j+ksize,:].reshape([-1])
            img_col.append(nowcol)
    img_col = np.array(img_col)
    return img_col
