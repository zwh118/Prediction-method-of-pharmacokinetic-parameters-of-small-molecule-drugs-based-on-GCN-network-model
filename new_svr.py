# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.03.25
"""

from sklearn import svm
import tensorflow as tf
import numpy as np
import xlrd
from pprint import pprint
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


initial_learning_rate = 0.001  # learning rate
training_epochs = 2000
batch_size = 59  # Number of training data (PPBR data)
val_batch_size = 30
display_step = 1

train_num = 149
val_num = 50

# Define placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 8])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])


# create models
def model(x, y):
    model = svm.SVR(kernel='sigmoid', C=10, gamma=0.1, epsilon=.1)
    y_svr = model.fit(x, y)

# predict
pred = model.predict(x)

# loss function
mae = tf.losses.absolute_difference(pred, y)
cost = tf.reduce_sum(mae)

# Gradient descent to obtain the optimal value
global_step = tf.Variable(tf.constant(0))
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=100, decay_rate=0.99, staircase=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# input data
def data(r1, r2, ws):
    y_ = []
    x_ = []
    for r in range(r1, r2):
        m = ws.cell(r, 10).value
        # print(m)
        m = [m]
        y_.append(m)
        x1_ = []
        for c in range(2, 10):
            n = ws.cell(r, c).value
            x1_.append(n)
        x_.append(x1_)

    return [x_, y_]

def accuracy(s,pred_,y_):
    a = []
    d = 0
    for u in range(s):
        b = abs(pred_[u][0] - y_[u][0])
        # print('b:', b)
        if b <= 0.1:
            a.append(b)
    # print(a)
    d1 = len(a)
    d += d1

    return d


train_y = []
val_pred = []

slist = []
slist1 = []
saver = tf.train.Saver()
bestacc = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file = xlrd.open_workbook(r'F:\WY\PPBR\data\PPBR(0.3).xlsx')




    for epoch in range(training_epochs):
        avg_cost1 = 0
        total_batch = int(train_num / batch_size)

        for i in range(total_batch):
            ws = file.sheet_by_name('train')
            r1 = i * batch_size + 1
            r2 = r1 + batch_size

            k = data(r1, r2, ws)# Import training set data

            _, co = sess.run([optimizer, cost], feed_dict={x: k[0], y: k[1]})

            x_, y_ = sess.run([x, y], feed_dict={x: k[0], y: k[1]})
            pred1 = sess.run(pred, feed_dict={x:  k[0], y: k[1]})


            train_y.append(y_)
            #print('pred', pred1)

            global_ = sess.run(global_step, feed_dict={x: k[0], y: k[1]})


            w = accuracy(batch_size, pred1, y_)

            Accuracy1 = w / batch_size
            #print('train accuracy：', Accuracy1)

            val_total_batch = int(val_num / val_batch_size)

            Accuracy3 = 0

            for ii in range(val_total_batch):
                ws = file.sheet_by_name('valid')
                r1 = ii * val_batch_size + 1
                r2 = r1 + val_batch_size

                kk = data(r1, r2, ws)# Import validation set data

                val_x_, val_y_ = sess.run([x, y], feed_dict={x: kk[0], y: kk[1]})
                #print(val_x_)
                pred2 = sess.run(pred, feed_dict={x: kk[0], y: kk[1]})
                #print(len(pred2))
                #print('pred2', pred2)


                ww = accuracy(val_batch_size, pred2, val_y_)

                Accuracy22 = ww / val_batch_size
                #print('valid accuracy：', Accuracy22)
                # average accuracy
                Accuracy3 = Accuracy3 + Accuracy22
                Accuracy2 = Accuracy3 / val_total_batch
                #print('acc valid accuracy：', Accuracy2)

                if Accuracy2 > bestacc:
                    bestacc = Accuracy2
                    bestglo = global_
                    bestpred = pred2
                    besty = val_y_
                    bestAccuracy1 = Accuracy1



    print('train accuracy：', bestAccuracy1)
    print('valid accuracy：', bestacc)


    # Draw an optimal model of the validation set
    y1 = bestpred
    x1 = [m for m in range(1, 1 + val_batch_size)]
    plt.scatter(x1, y1, marker='v', label='pred_y')
    plt.scatter(x1, besty, marker='o', label='y')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
    plt.show()


    saver.save(sess, "F:\WY\PPBR\model（0.3）\my_model.ckpt", global_step=bestglo)
            #print('valid accuracy：', Accuracy2)






