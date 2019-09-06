import numpy as np
import pickle

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar100_data(data_home='./cifar100_dataset/cifar-100-python/',label_mode='coarse'):
    """
    data : (50000,3072) (10000,3072)
    labels : (50000,) (10000,)
    :param data_home:
    :return:
    """
    if label_mode == 'coarse':
        label_mode = b'coarse_labels'
    elif label_mode == 'fine':
        label_mode = b'fine_labels'

    train_dict = unpickle(data_home+'train')
    test_dict = unpickle(data_home+'test')

    train_datas = train_dict[b'data']
    train_labels = train_dict[label_mode]
    test_datas = test_dict[b'data']
    test_labels = test_dict[label_mode]
    # datas 是uint8 的ndarray 但是label需要我们自己转换
    train_labels = np.array(train_labels,dtype=np.uint8)
    test_labels = np.array(test_labels,dtype=np.uint8)

    # image data 归一化 不然影响cnn的卷积参数 导致梯度过大 不同与标准化 不会改变分布
    # 这个原始数据没有归一化
    data_max = np.max(train_datas)
    data_min = np.min(train_datas)
    train_datas = (train_datas - data_min) / (data_max - data_min)
    test_datas = (test_datas - data_min) / (data_max - data_min)

    randidx = np.random.permutation(train_datas.shape[0])
    # randidx = np.random.shuffle(np.range(mnist.data.shape[0]))
    train_datas = train_datas[randidx]
    train_labels = train_labels[randidx]

    randidx = np.random.permutation(test_datas.shape[0])
    # randidx = np.random.shuffle(np.range(mnist.data.shape[0]))
    test_datas = test_datas[randidx]
    test_labels = test_labels[randidx]

    return (train_datas,train_labels,test_datas,test_labels)

if __name__ == '__main__':
    data_home = './cifar100_dataset/cifar-100-python/'
    # train_dict = unpickle(data_home+'train')
    # test_dict = unpickle(data_home+'test')
    # desc_dict = unpickle(data_home+'meta')
    # print(len(desc_dict[b'fine_label_names']),len(desc_dict[b'coarse_label_names']))
    # print(len(train_dict[b'data']))
    # print(len(test_dict[b'data']))
    # print(train_dict[b'batch_label'])
    # print(test_dict[b'batch_label'])
    #
    # print(train_dict[b'fine_labels'][:10])
    # print(train_dict[b'coarse_labels'][:10])
    # # 100 fine classes   20 coarse classes
    #
    # print(train_dict[b'data'].shape) # (50000, 3072)
    # print(np.max(train_dict[b'data']),np.min(train_dict[b'data'])) # 255 0
    # print(type(train_dict[b'data'][0][0])) # <class 'numpy.uint8'>
    # print(type(train_dict[b'coarse_labels'][0])) # <class 'numpy.uint8'>

    load_cifar100_data(data_home=data_home)

