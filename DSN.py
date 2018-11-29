# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:DSN.py
@time:2018/11/2514:08
"""
import tensorflow as tf
import numpy as np
import cv2
import SimpleITK as sitk
import os
from Data_Generator import Data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CLASSES = 2
BLOCK_SIZE = [32, 512, 512]
path = 'E:/liverdata/nii/nrrd3D/CompleteData'
stride = [12, 127, 127]

layers = [['block0', [['conv', [9, 9, 7, 1, 8], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
          ['block1', [['conv', [9, 9, 7, 8, 16], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['maxpool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME']]],
          ['block2', [['conv', [7, 7, 5, 16, 32], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
          ['block3', [['conv', [7, 7, 5, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['maxpool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME']], ],
          ['block4', [['conv', [5, 5, 3, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
          ['block5', [['conv', [1, 1, 1, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
          ['block6', [['deconv', [3, 3, 3, 32, 32], [1, 2, 2, 2, 1], 'SAME']]],
          ['block7', [['deconv', [3, 3, 3, 2, 32], [1, 2, 2, 2, 1], 'SAME']]],
          ]

main_stream = []
surprise_layers = [3, 6]

#  卷积层+dropout+BN层
def conv3D(inlayer, name, kernel, stride, padding, dropout):
    w = tf.get_variable(name + 'w', shape=kernel, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0, 0.01))
    b = tf.get_variable(name + 'b', shape=[kernel[-1]], initializer=tf.constant_initializer(0.1))

    out = tf.nn.bias_add(tf.nn.conv3d(inlayer, w, stride, padding), b)
    drop = tf.nn.dropout(out, dropout)
    BN = tf.layers.batch_normalization(drop, training=True)
    l2_loss = tf.contrib.layers.l2_regularizer(0.003)(w)
    tf.add_to_collection('l2_loss', l2_loss)
    bias = tf.nn.relu(BN)
    return bias

#反卷积层
def deconv3D(inlayer, name, kernel, stride, padding):
    in_shape = tf.shape(inlayer)
    w = tf.get_variable(name + 'W', shape=kernel, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0.0, 0.01))
    b = tf.get_variable(name + 'b', shape=[kernel[-2]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    output_shape = tf.stack(
        [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, kernel[-2]])
    deconv = tf.nn.conv3d_transpose(inlayer, w, output_shape, strides=stride, padding=padding)
    bias = tf.nn.relu(tf.nn.bias_add(deconv, b))
    return bias


g = tf.Graph()

with g.as_default():
    with tf.variable_scope('input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 1])  # [batch,batchsize,w,h,c]
        Y = tf.placeholder(dtype=tf.int64, shape=[None, None, None, None])  # [batch,batchsize,w,h]
        W = tf.placeholder(dtype=tf.float32, shape=[4])
    main_stream.append(X)
    for block in layers:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                if layer[0].startswith('conv'):
                    conv = conv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4])
                    main_stream.append(conv)
                if layer[0].endswith('pool'):
                    pool = tf.nn.max_pool3d(main_stream[-1], layer[1], layer[2], layer[3])
                    main_stream.append(pool)
                if layer[0].startswith('deconv'):
                    conv = deconv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3])
                    main_stream.append(conv)
    with tf.variable_scope('surprise_1'):
        surprise_1 = deconv3D(main_stream[3], 'deconv', [3, 3, 3, 2, 16], [1, 2, 2, 2, 1], 'SAME')

    with tf.variable_scope('surprise_2'):
        up1 = deconv3D(main_stream[6], 'deconv1', [3, 3, 3, 32, 32], [1, 2, 2, 2, 1], 'SAME')
        surprise_2 = deconv3D(up1, 'deconv2', [3, 3, 3, 2, 32], [1, 2, 2, 2, 1], 'SAME')

    with tf.variable_scope('out'):
        pre = tf.nn.softmax(main_stream[-1])

    with tf.variable_scope('out1'):
        pre1 = tf.nn.softmax(surprise_1[-1])

    with tf.variable_scope('out2'):
        pre2 = tf.nn.softmax(surprise_2[-1])

    with tf.variable_scope('loss'):
        flat_labels = tf.reshape(tf.one_hot(Y, CLASSES), [-1, CLASSES])
        flat_logits = tf.reshape(pre, [-1, CLASSES])
        flat_loss1 = tf.reshape(pre1, [-1, CLASSES])
        flat_loss2 = tf.reshape(pre2, [-1, CLASSES])
        loss_map = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_logits, labels=flat_labels)

        loss1 = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_loss1, labels=flat_labels)

        loss2 = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_loss2, labels=flat_labels)

        class_weights = W[0:2]
        weight_map = tf.multiply(flat_labels, class_weights)
        weight_maps = tf.reduce_sum(weight_map, axis=1)
        weighted_loss = tf.multiply(loss_map + loss1 * W[2] + loss2 * W[3], weight_maps)

        loss = tf.reduce_mean(weighted_loss)

        #计算模型准确度
        pre_img = tf.argmax(pre, -1)   #返回最大的那个数值所在的下标。
        ans = tf.equal(pre_img, Y)   #tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
        acc = tf.reduce_mean(tf.cast(ans, tf.float32))  #tf.cast(ans, tf.float32) 原来x的数据格式是bool， 那么将其转化成float以后，就能够将其转化成0和1的序列。

        steps = 1000
        g_steps = tf.Variable(0)

        rates = tf.train.exponential_decay(0.2, g_steps, 50, 0.95, staircase=True)     #学习率指数下降
        # train = tf.train.GradientDescentOptimizer(rates).minimize(loss, global_step=g_steps)
        train = tf.train.MomentumOptimizer(
            learning_rate=rates, momentum=0.2).minimize(loss=loss, global_step=g_steps)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc', acc)
        merged = tf.summary.merge_all()


data = Data(path, BLOCK_SIZE,stride)
if __name__ == '__main__':
    # saver = tf.train.import_meta_graph('./test_model_save/test.ckpt.meta')
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()    #训练网络后想保存训练好的模型，保存和恢复都需要实例化一个 tf.train.Saver。
        tf.global_variables_initializer().run()
        key = 0.0045  # 0.005
        summary_writer = tf.summary.FileWriter('./summary', graph=sess.graph)  #指定一个文件用来保存图。
        w = [0.1, 0.2, 0.3, 0.4]
        ans3 = 1000
        # cv2.imwrite('./prediction/test_.jpg', np.uint8(
        # (pic[0, :, :, 0] < pic[0, :, :, 1])) * 255)
        count = 0
        iteration = 0
        while iteration < 10000:
            try:
                try:
                    x, y = data.next()
                except Exception as e:
                    data = Data(path, BLOCK_SIZE,stride)
                    x, y = data.next()

                flat = y.flatten().tolist()

                portion = sum(flat) * 1.0 / (len(flat) - sum(flat))   #二值的面积与非二值的面积的比值

                if portion < 0.1:     #判断这一小块是否包含二值有用信息
                    continue
                iteration += 1
                w = [portion, 1, 0.8 * (0.99 ** (iteration // 200)),
                     1.0 * (0.99 ** (iteration // 200))]

                ans1, ans2, ans3, ans4 = sess.run(
                    [loss, acc, rates, merged], feed_dict={X: x, Y: y, W: w})
                sess.run(train, feed_dict={X: x, Y: y, W: w})
                if iteration % 100 == 0:
                    count += 1
                    summary_writer.add_summary(ans4, count)

                print(
                    "Iteration:{0},loss:{1},acc:{2},rates:{3},weight:{4}".format(str(iteration),
                                                                                 ans1,
                                                                                 ans2, ans3, w))
                pic = sess.run(pre_img, feed_dict={X: x, Y: y, W: w})
                cv2.imwrite(
                    './prediction/pre_' + str(iteration) + '.jpg',      #训练数据的预测标签
                    np.uint8(pic[0, 0, ...]) * 255)
            except Exception as e:
                print("出现异常，保存模型")
                saver.save(sess, './test_model_save/test' + str(iteration) + '.ckpt')

            if iteration % 1000 == 0:
                saver.save(sess, './test_model_save/test' + str(iteration) + '.ckpt')

            if ans3 - 0 < 0.00001:
                break

        saver.save(sess, './test_model_save/test.ckpt')
