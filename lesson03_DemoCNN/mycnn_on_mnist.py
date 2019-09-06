import time

import numpy as np
from sklearn.datasets import fetch_mldata

from layers.activation_func_relu import Relu
from layers.conv_layer import Conv2D
from layers.fc_layer import FullyConnectedLayer
from layers.pooling_layers import MaxPooling
from layers.softmax_layer import Softmax

mnist = fetch_mldata('MNIST original',data_home='./dataset/')
datas = np.array(mnist.data)
labels = np.array(mnist.target)


# image data 归一化 不然影响cnn的卷积参数 导致梯度过大 不同与标准化 不会改变分布
# 这个原始数据没有归一化有坑 导致一开始梯度过于大 直接conv参数爆炸
# print(np.max(datas))
datas = (datas - np.min(datas)) / ((np.max(datas) - np.min(datas)))
print(np.max(datas),np.min(datas))

randidx = np.random.permutation(mnist.data.shape[0])
datas = datas[randidx]
labels = labels[randidx]

(train_datas,train_labels) = (datas[:50000],labels[:50000])
(test_datas,test_labels) = (datas[50000:],labels[50000:])


### 双层神经网络搭建
batch_size = 16
conv1 = Conv2D([batch_size,28,28,1],8,5,1)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.output_shape,2,2)
conv2 = Conv2D(pool1.output_shape,16,3,1)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.output_shape,2,2)
# conv3 = Conv2D(pool2.output_shape,32,3,1)
# relu3 = Relu(conv3.output_shape)
# pool3 = MaxPooling(relu3.output_shape,2,2)
fc = FullyConnectedLayer(pool2.output_shape,10)
problayer = Softmax(fc.output_shape)

epochs = 10

for epoch in range(epochs):
    learning_rate = 1e-3
    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0

    # train
    for i in range(train_datas.shape[0]//batch_size):
        batch_loss = 0
        batch_acc = 0
        # batch data
        img = train_datas[i*batch_size:(i+1)*batch_size].reshape(
            [batch_size,28,28,1])
        target = train_labels[i*batch_size:(i+1)*batch_size]
        # forward
        conv1_out = conv1.forward_propagate(img)
        relu1_out = relu1.forward_propagate(conv1_out)
        pool1_out = pool1.forward_propagate(relu1_out)
        conv2_out = conv2.forward_propagate(pool1_out)
        relu2_out = relu2.forward_propagate(conv2_out)
        pool2_out = pool2.forward_propagate(relu2_out)
        # conv3_out = conv3.forward_propagate(pool2_out)
        # relu3_out = relu3.forward_propagate(conv3_out)
        # pool3_out = pool3.forward_propagate(relu3_out)
        fc_out = fc.forward_propagate(pool2_out)
        #         print(fc_out.shape)
        #         print(np.array(target).shape)
        loss = problayer.cal_loss(fc_out,np.array(target))
        train_loss += loss
        batch_loss += loss

        # print(problayer.prob)
        for j in range(batch_size):
            if np.argmax(problayer.prob[j]) == int(train_labels[j]):
                batch_acc += 1
                train_acc += 1

        # backward
        # softmax 层自己已经有了loss 从而可以向后传播loss得到的梯度
        problayer.backward_propagate()
        #使用嵌套方式计算梯度 并且自动更新参数
        # conv1.backward_propagate(relu1.backward_propagate(
        #     pool1.backward_propagate(conv2.backward_propagate(
        #         relu2.backward_propagate(pool2.backward_propagate(
        #             conv3.backward_propagate(relu3.backward_propagate(
        #                 pool3.backward_propagate(fc.backward_propagate(problayer.delta,
        #                     learning_rate=learning_rate))),learning_rate=learning_rate))),learning_rate=learning_rate))),learning_rate=learning_rate)

        conv1.backward_propagate(relu1.backward_propagate(
            pool1.backward_propagate(conv2.backward_propagate(
                relu2.backward_propagate(pool2.backward_propagate(
                    fc.backward_propagate(problayer.delta,learning_rate=learning_rate))),
                learning_rate=learning_rate))), learning_rate=learning_rate)

        if i % 20 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                  "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (
                      epoch,i, batch_acc / float(batch_size), batch_loss / float(batch_size), learning_rate))

    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
        epoch, train_acc / float(train_datas.shape[0]), train_loss / float(train_datas.shape[0])))

    # test
    for i in range(images.shape[0]//batch_size):
        img = test_datas[i*batch_size:(i+1)*batch_size].reshape(
            [batch_size,28,28,1])
        target = test_labels[i*batch_size:(i+1)*batch_size]
        # forward
        conv1_out = conv1.forward_propagate(img)
        relu1_out = relu1.forward_propagate(conv1_out)
        pool1_out = pool1.forward_propagate(relu1_out)
        conv2_out = conv2.forward_propagate(pool1_out)
        relu2_out = relu2.forward_propagate(conv2_out)
        pool2_out = pool2.forward_propagate(relu2_out)
        # conv3_out = conv3.forward_propagate(pool2_out)
        # relu3_out = relu3.forward_propagate(conv3_out)
        # pool3_out = pool3.forward_propagate(relu3_out)
        fc_out = fc.forward_propagate(pool2_out)
        test_loss += problayer.cal_loss(fc_out,np.array(target))
        for j in range(batch_size):
            if np.argmax(problayer.prob[j]) == train_labels[j]:
                test_acc += 1
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, test_acc / float(test_datas.shape[0]), test_loss / float(test_datas.shape[0])))

