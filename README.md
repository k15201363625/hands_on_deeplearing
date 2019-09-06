# 项目简介

- 动手实现CNN RNN 
- 基于手动实现的Layer 构建自己的CNN　并且实现训练测试流程
- 用自己构建的模型(version1 + version2)在mnist数据集上进行测试
- 用改进优化的模型(从代码组织上的改进，从网络结构以及网络细节的改进)　在cifar100上进行模型的训练

## 主要实现内容
- 对于CNN
  - numpy实现ConvLayer PoolingLayer FCLayer
- 对于RNN
  -  numpy实现BasicRNNCell  LSTMCell FCLayer
- 对于网络训练方面
  - numpy实现了BatchNormLayer Activator  SoftMaxLossLayer DropoutLayer
- 对于网络模型构建方面
　- 借鉴tensorflow中的设计思维　实现了基于图结构的模型
　- 实现了两个基类Operator Variable 
　- 实现了register_graph方法　以及　GLOBAL_VARIABLE_SCOPE命名空间用于存储所有定义的Operator 以及　Variable 从而便于管理以及更新参数
　- 通过图结构　以及　递归调用　实现了模型自动前向传播反向传播以及利用梯度更新过程　

## 代码说明
- 运行环境:python 3.7.3 numpy1.16.4 
- 开发环境:pycharm-ce jupyter-lab1.0.2
- jupyter-lab上的代码在当前文件夹下的lesson01 02 的ipynb文件中　有lstm basicrnncell conv maxpooling等层次的基本实现 
- 在lesson03_DemoCNN  以及lesson04_practive_on_cifar100中有CNN的两个项目
  - lesson03是两个版本的层次实现并在mnist上进行实验
  - lesson04是在cifar100数据集上进行测试　
　　- lesson03/layers目录中实现的是version1版本的Layer代码
　　- lesson03/tensor_graph目录中实现的是version2版本　基于图结构优化后的Layer代码
　　- lesson03_DemoCNN中有中有对于两个版本的层次进行调用实现网络模型构建的代码
　　- 在对应的dataset目录中有下载的相应的数据 mnist and cifar100
　　- 运行日志保存在logs目录下

## 本次学习的报告
- 请见deeplearning_learning_report01.md
