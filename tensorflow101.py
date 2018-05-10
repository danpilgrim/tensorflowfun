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

#Example 3 - Building a Linear Model
sess = tf.Session()

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

#because we are initalizing variables, not contstants/placeholders
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))
# Output: [0.         0.3        0.6        0.90000004]

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) #takes SE
loss = tf.reduce_sum(squared_deltas) # Mean of SE
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# Output: 23.66

fixW = tf.assign(W, [-1.]) #array of type float32
fixb = tf.assign(b, [1.])
sess.run([fixW,fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W,b]))
