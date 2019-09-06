from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE
from tensor_graph.Operator import Operator
from tensor_graph.Acticativator import *
from tensor_graph.Conv2D import Conv2D
from tensor_graph.MaxPooling import MaxPooling
from tensor_graph.FullyConnected import FullyConnect
from tensor_graph.BatchNorm import BatchNorm
from tensor_graph.Dropout import DropOut
from tensor_graph.Softmax import Softmax
from tensor_graph.Acticativator import *
from tensor_graph.utils import learning_rate_exponential_decay
from sklearn.datasets import fetch_mldata

import numpy as np
import time
import os

VERSION = 'TENSOR_Adam_RELU'


def load_mnist_data(data_name='MNIST original',data_home='./mnist_dataset'):
    mnist = fetch_mldata(dataname=data_name,data_home=data_home)
    datas = np.array(mnist.data)
    labels = np.array(mnist.target,dtype=np.int32)


    # image data 归一化 不然影响cnn的卷积参数 导致梯度过大 不同与标准化 不会改变分布
    # 这个原始数据没有归一化有坑 导致一开始梯度过于大 直接conv参数爆炸
    # print(np.max(datas))
    datas = (datas - np.min(datas)) / (np.max(datas) - np.min(datas))
    print(np.max(datas),np.min(datas))

    randidx = np.random.permutation(mnist.data.shape[0])
    # randidx = np.random.shuffle(np.range(mnist.data.shape[0]))
    datas = datas[randidx]
    labels = labels[randidx]

    (train_datas,train_labels) = (datas[:50000],labels[:50000])
    (test_datas,test_labels) = (datas[50000:],labels[50000:])
    return (train_datas,train_labels,test_datas,test_labels)

def build_model(x=Variable,y=Variable,output_num=int):
    """
    build model and forward propagate to get output
    """
    conv1_out = Conv2D([5,5,1,16],input_variable=x,name='conv1',padding='VALID').output_variables
    relu1_out = Relu(input_variable=conv1_out,name='relu1').output_variables
    dropout1_output = DropOut(input_variable=relu1_out,name='dropout1',state='train').output_variables
    pool1_out = MaxPooling(input_variable=dropout1_output,ksize=2,name='pool1').output_variables
    bn1_out = BatchNorm(input_variable=pool1_out,name='bn1').output_variable

    conv2_out = Conv2D([3,3,16,32], input_variable=bn1_out, name='conv2', padding='VALID').output_variables
    relu2_out = Relu(input_variable=conv2_out, name='relu2').output_variables
    dropout2_output = DropOut(input_variable=relu2_out, name='dropout2', state='train').output_variables
    pool2_out = MaxPooling(input_variable=dropout2_output, ksize=2, name='pool2').output_variables
    bn2_out = BatchNorm(input_variable=pool2_out,name='bn2').output_variable

    fc_out = FullyConnect(input_variable=bn2_out,output_num=output_num,name='fc').output_variables
    softmax_outs = Softmax(fc_out,y,'sf').output_variables
    return softmax_outs




def train_model(model_out, img_placeholder, label_placeholder,
                train_imgs, train_labels, epochs=10,batch_size=32,method='adam',l2_decay_rate=3e-4):
    global_step = 0

    sfloss,sfprediction = model_out

    # 对于全局变量方法以及状态进行设置
    for k in GLOBAL_VARIABLE_SCOPE.keys():
        var = GLOBAL_VARIABLE_SCOPE[k]
        if isinstance(var,Variable) and var.trainable:
            if method == 'adam':
                var.set_method_adam()
            elif method == 'momentum':
                var.set_method_momentum()
            elif method == 'nga':
                var.set_method_nga()
            elif method == 'sgd':
                var.set_method_sgd()
            else:
                raise  AttributeError('unsupported method')
        if isinstance(var,Operator) and hasattr(var,'state'):
            var.state = 'train' #default == 'test'



    print('------------------------------start_train----------------------------')

    with open('logs/mnist/%s_train_log.txt'%VERSION,'w') as logf:
        for epoch in range(epochs):
            # random shuffle
            idx = np.arange(train_imgs.shape[0])
            np.random.shuffle(idx) # 原地操作　没有返回
            imgs = train_imgs[idx]
            labels = train_labels[idx]
            train_val_split_dot = imgs.shape[0]//5*4
            train_imgs = imgs[:train_val_split_dot]
            train_labels = labels[:train_val_split_dot]
            val_imgs = imgs[train_val_split_dot:]
            val_labels = labels[train_val_split_dot:]

            batch_loss = 0
            batch_acc = 0

            # train
            train_acc = 0
            train_loss = 0

            # val
            val_acc = 0
            val_loss = 0

            for i in range(train_imgs.shape[0] // batch_size):
                # 每个batch 推进　learning_rate 衰减　配合adam使用
                learning_rate = learning_rate_exponential_decay(5e-4, epoch, 0.1, 10)

                # feed
                img_placeholder.data = train_imgs[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
                label_placeholder.data = train_labels[i * batch_size:(i + 1) * batch_size]

                # forward
                # 只调用一次forward避免bn影响不同forward不同结果
                sfloss.eval()
                _loss = sfloss.data
                _prediction = sfprediction.data

                batch_loss += _loss
                train_loss += _loss

                for j in range(batch_size):
                    if np.argmax(_prediction[j]) == label_placeholder.data[j]:
                        batch_acc += 1
                        train_acc += 1

                # backward　对trainable var 进行update value
                img_placeholder.diff_eval()
                for k in GLOBAL_VARIABLE_SCOPE.keys():
                    var = GLOBAL_VARIABLE_SCOPE[k]
                    if isinstance(var,Variable) and var.trainable:
                        var.apply_gradient(learning_rate=learning_rate, decay_rate=l2_decay_rate, batch_size=batch_size)
                    # 每次需要梯度清零
                    if isinstance(var, Variable):
                        var.diff = np.zeros(var.shape)
                    # 每次应用梯度更新一个参数　global_step += 1
                    global_step += 1

                if i % 50 == 0 and i != 0:
                    record_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                    "  %s epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (
                    VERSION, epoch,i, batch_acc/float(batch_size), batch_loss/batch_size,learning_rate)

                    print(record_str)
                    logf.write(record_str)

                batch_loss = 0
                batch_acc = 0

            record_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) +\
                  "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
                  epoch, train_acc/float(int(train_imgs.shape[0]//batch_size) * batch_size),
                  train_loss/float(int(train_imgs.shape[0]//batch_size) * batch_size))
            print(record_str)
            logf.write(record_str)

            # validation
            for i in range(val_imgs.shape[0]//batch_size):
                img_placeholder.data = val_imgs[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
                label_placeholder.data = val_labels[i * batch_size:(i + 1) * batch_size]

                # val / test 过程只需要前向传播　所以需要设置正确的var op 状态　并且对于state需要设置
                for k in GLOBAL_VARIABLE_SCOPE.keys():
                    var = GLOBAL_VARIABLE_SCOPE[k]
                    if isinstance(var, Variable):
                        var.wait_bp = False
                    if isinstance(var, Operator):
                        var.wait_forward = True
                    if isinstance(var, Operator) and hasattr(var, 'state'):
                        var.state = 'test'

                _loss = sfloss.eval()
                _prediction = sfprediction.data

                val_loss += _loss

                for j in range(batch_size):
                    if np.argmax(_prediction[j]) == label_placeholder.data[j]:
                        val_acc += 1

            record_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) +\
                  "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
                epoch, val_acc/float(int(val_imgs.shape[0]//batch_size)*batch_size),
                val_loss/float(int(val_imgs.shape[0]//batch_size)*batch_size))
            print(record_str)
            logf.write(record_str)


def test_model(model_out, img_placeholder, label_placeholder, test_imgs, test_labels, batch_size=32):
    sfloss,sfprediction = model_out

    idx = np.arange(test_imgs.shape[0])
    np.random.shuffle(idx)  # 原地操作　没有返回
    test_imgs = test_imgs[idx]
    test_labels = test_labels[idx]

    print('-------------------------start test-----------------------------')

    test_loss = 0
    test_acc = 0

    with open('logs/mnist/%s_test_log.txt' % VERSION, 'w') as logf:
        for i in range(test_imgs.shape[0]//batch_size):
            img_placeholder.data = test_imgs[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
            label_placeholder.data = test_labels[i * batch_size:(i + 1) * batch_size]

            for k in GLOBAL_VARIABLE_SCOPE.keys():
                var = GLOBAL_VARIABLE_SCOPE[k]
                if isinstance(var, Variable):
                    var.wait_bp = False
                if isinstance(var, Operator):
                    var.wait_forward = True
                if isinstance(var, Operator) and hasattr(var, 'state'):
                    var.state = 'test'

            _loss = sfloss.eval()
            _prediction = sfprediction.data

            test_loss += _loss

            for j in range(batch_size):
                if np.argmax(_prediction[j]) == label_placeholder.data[j]:
                    test_acc += 1

        record_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +\
                     "test_acc: %.4f  avg_test_loss: %.4f" % (
                     test_acc/float(int(test_imgs.shape[0]//batch_size)*batch_size), test_loss/float(int(test_imgs.shape[0]//batch_size)*batch_size))
        print(record_str)
        logf.write(record_str)


def main():
    if not os.path.exists('./logs/mnist'):
        os.makedirs('./logs/mnist')

    train_imgs, train_labels, test_imgs, test_labels = load_mnist_data(data_home='./mnist_dataset')
    batch_size = 32
    # 准备输入输出
    img_placeholder = Variable((batch_size,28,28,1),'input')
    label_placeholder = Variable((batch_size,1),'label')

    model_out = build_model(img_placeholder,label_placeholder,10)
    train_model(model_out, img_placeholder,label_placeholder, train_imgs,train_labels,epochs=10,batch_size=batch_size)
    test_model(model_out,img_placeholder, label_placeholder, test_imgs,test_labels,batch_size=batch_size)


main()