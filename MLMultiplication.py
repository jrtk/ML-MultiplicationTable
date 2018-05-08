# A tiny try to see how ML find multiplier from given table
'''
Example:
  1 * X = 3
  2 * X = 6
  3 * X = 9
  4 * X = 12
  ..........
  100 * x = 300
  Let's Train the Model and find the value of X?
'''


import tensorflow as tf
import numpy as np

sess = tf.Session()

inputValues = np.arange(1,100)

groundTruth = np.arange(3,300,3)
print (groundTruth)


pHInputValues   = tf.placeholder(shape=[1], dtype=tf.float32)
pHGroundTruth   = tf.placeholder(shape=[1], dtype=tf.float32)

X = tf.Variable(tf.random_normal(shape=[1]))

computationResult  = tf.multiply (pHInputValues, X)
loss  = tf.square(computationResult - pHGroundTruth)
init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_step = optimizer.minimize(loss)

print ('Before We Start Training.. step # ' + 'X = ' + str(sess.run(X)))


for i in range(100):
    idx = np.random.choice(99)
    randX = [inputValues[idx]]
    randY = [groundTruth[idx]]
    print ("Random Index Value  is ", idx)
    print ("X Value", randX)
    print ("Y Value ", randY)
    sess.run(train_step, feed_dict={pHInputValues:randX, pHGroundTruth:randY})
    print ('A = ' + str(sess.run(X)))
    print ('Loss is ' + str(sess.run(loss, feed_dict={pHInputValues:randX, pHGroundTruth:randY})))
    print ('--------------------------------')

