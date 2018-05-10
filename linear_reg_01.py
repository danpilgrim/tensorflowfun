#!/danz/bin/env python

import tensorflow as tf
import numpy

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# training data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

#sets weights of module
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# linear model definition
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# initializes the gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer() # inits all variables

#training session
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            # THE CORE
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    #allows for Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

print(rng)

# upon run, getting this error:
'''
/backend_macosx.py", line 17, in <module>
    from matplotlib.backends import _macosx
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to f
unction correctly if Python is not installed as a framework. See the Python documentation for mo
re information on installing Python as a framework on Mac OS X. Please either reinstall Python a
s a framework, or try one of the other backends. If you are using (Ana)Conda please install pyth
on.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in t
he Matplotlib FAQ for more information.
'''
