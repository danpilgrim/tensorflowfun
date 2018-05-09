

import tensorflow as tf

# Creates constant values
a = tf.constant(2)
b = tf.constant(3)

# while session is running, you can make have output from any computation with sess.run(X)
with tf.Session() as sess:
    print "a: %i" % sess.run(a), "b: %i" % sess.run(b)
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multiplication with constants: %i" % sess.run(a*b)

# Which why not just do this, but I didnt make tensorflow
print "Same thing %i" % (1+1)

'''
# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
'''

# defines a,b as 16 bit integers (python be dammed)
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# addition and multiplication operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    print "Test 2"
    print "Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3})
    print "Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3})



'''
Matrix multiplication

creates a 1x2 matrix constant, op is added as a node to the default Graph

value returned by the constructor represents the output of the constant
'''
# 1x2 matrix
matrix1 = tf.constant([[3., 3.]])
# 2x1 matrix
matrix2 = tf.constant([[2.],[ 2.]])


# Create a Matmul op that takes matrix1 and matrix2 as inputs
# Returned value, 'product', represents the result of the matrix multiplication
product = tf.matmul(matrix1, matrix2)

'''
to run matmul we have to call run() method with product as input
the output will be matmul op

run(product) runs the executions of everything attached to product,
    output returned as numpy ndarray object
'''

with tf.Session() as sess:
    result = sess.run(product)
    print "Output: (numpy ndarray object)"
    print result
