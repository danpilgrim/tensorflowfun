# Tensorflow 101 on youtube

import tensorflow as tf


''' EXAMPLE 1
c = tf.multiply(a,b, name="multiply")
addtwo = a + b

print(sess.run(addtwo, {a:[3,4], b:[0,10]}))
'''


''' EXAMPLE 2
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

a = tf.constant(2, name="A")
b = tf.constant(3,name="B")

c = tf.multiply(a,b, name="multiply_c")
d = tf.add(a,b)
e = tf.add(c,d)
sess = tf.Session()
output = sess.run(e)

## dumps results into mygraph directory
writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()
sess.close()

# use tensorboard --logdir=mygraph and navigate to http://127.0.0.1:6006 to tensorboard
'''
