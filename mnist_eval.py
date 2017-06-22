import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import numpy as np
import os

# 加载mnist_inference和mnist_train中定义的常量和函数
import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    validation = mnist.validation
    test = mnist.test

    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=(None, mnist_inference.layers[0]), name='x-input')
        y_ = tf.placeholder(tf.float32, shape=(None, mnist_inference.layers[-1]), name='y-input')

        validation_feed = {x: validation.images, y_: validation.labels}
        test_feed = {x: test.images, y_: test.labels}

        y = mnist_inference.inference(x)
        correct_prediction = tf.equal(tf.arg_max(y, dimension=1), tf.arg_max(y_, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        # 每隔EVAL_INTERVAL_SECS调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # 此函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validation_feed)

                    print('After %s steps, the score is %g' % (global_step, accuracy_score))
                else:
                    print('No such model')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    print(4)
    evaluate(mnist_inference.mnist)

if __name__ == '__main__':
    print(3)
    # tf.app.run(mnist_train.train(mnist_inference.mnist))
    tf.app.run(main)