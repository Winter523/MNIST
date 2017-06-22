import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import numpy as np

# 获取数据集
mnist = input_data.read_data_sets('/path/to/MNIST', one_hot=True)

# 定义网络结构
(m, n) = mnist.train.images.shape
(_, o) = mnist.train.labels.shape

layers = [n,500,o]
num_layers = len(layers)
name_layers = []
for i in range(num_layers):
    name_layers.append('layer'+str(i))
value_layers = []

# 定义初始化
PARAMETERS_INITIALIZER = tf.random_normal_initializer

# 计算神经网络的前向计算结果
def inference(input_tensor, regularizer = None):
    # 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
    value_layers.append(input_tensor)
    in_dimension = layers[0]
    for i in range(1, num_layers):
        with tf.variable_scope(name_layers[i]):
            out_dimension = layers[i]
            weights = tf.get_variable('weights', shape=(in_dimension, out_dimension),
                                      initializer=PARAMETERS_INITIALIZER(stddev=0.1))
            biases = tf.get_variable('biases', shape=(1, out_dimension),
                                     initializer=tf.constant_initializer(0.0))
            if regularizer != None:
                tf.add_to_collection('loss', regularizer(weights))
                # print('how many times?')
            value = tf.matmul(value_layers[-1], weights) + biases
            if i == num_layers - 1:
                # 最后一层配合softmax则放弃激活函数
                value_layers.append(value)
            else:
                value_layers.append(tf.nn.relu(value))
            in_dimension = out_dimension
    return value_layers[-1]