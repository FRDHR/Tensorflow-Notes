#coding:utf-8
#酸奶成本一元，利润九元
#预测少了损失大，故生成的模型会尽量多的预测
#0导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_=[[X1+X2+(rdm.rand()/10.0-0.05)] for (X1 ,X2) in X]

#1定义神经网络的输入。参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32,shape=(None,2))
y_= tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#2定义损失函数及反向传播方法
#定义损失函数为MSE，及反向传播方法为梯度下降
loss_mse = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE)%32
		end = start + BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
		if i%500==0:
			print "After %d training steps, w1 is: " % (i)
			print sess.run(w1),"\n"
	print "Final w1 is:\n",sess.run(w1)
