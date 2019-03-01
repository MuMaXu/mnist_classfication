import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播函数
import mnist_inference

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径和文件名。
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist.ckpt"

# 训练
def train(mnist):
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型再验证集上的表现，验证和测试的过程有独立程序完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 1000轮保存一次模型。
            if i % 1000 == 0:
                # 输出当前训练情况。这里只是输出了模型再当前训练batch上的损失函数大小。
                # 通过损失函数的大小可以大概了解训练情况。在验证集上正确率有一个单独程序完成。
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 保存当前的模型。这里给出了global_step参数，这样可以让每个被保存的模型的文件名末尾加上训练的轮数。
                # 比如model.ckpt-1000表示训练1000轮之后得到的模型。
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
