#coding:utf-8
#0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

#基于seed产生随机数
rng = np.random.RandomState(seed)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rng.rand(32,2)
#从这个32行2列的矩阵中 取出一行 判断如果小于1 给y赋值1 如果不小于1 给y赋值0
#作为输入数据集的标签（正确答案）
Y = [[int(x0 + x1 < 1)] for (x0 ,x1) in X]
print "X:\n",X
print "Y:\n",Y

#1定义神经网络输入，参数，输出，定义前向传播
x = tf.placeholder(tf.float32,shape=(None,2))
y_= tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#2定义损失函数及反向传播
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	#输出目前（未经训练）的参数取值
	print "w1:\n",sess.run(w1)
	print "w2:\n",sess.run(w2)
	print "\n"

	#训练模型
	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE)%32
		end = start + BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i%500==0:
			total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
			print("After %d training step(s),loss on all data is %g" % (i,total_loss))

	#输出训练后的参数值
	print "\n"
	print "w1:\n",sess.run(w1)
	print "w2:\n",sess.run(w2)


