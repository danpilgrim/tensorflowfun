
# From Tensorflow's tutorial documentation


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()


'''
# Notes
# tf.layers :
# conv2d() = 2D convolutional layer, ARGS: kernel, filter, padding, activation print_function
# max_pooling2d() = 2D pooling layer, ARGS: fitler size, stride
# dense = dense layer, ARGS: number of nuerons and activation func as arguments

'''

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  '''
  two dimensional image convolutional layers: [batch_size, image_height, image_width, channels]
  batch_size = size of subsets to use during training's gradient Descent
    (-1 when infered on feature input values)
  channels = # of color channels in example
  data_format = "channels_last/channels_first"
  Since input format is 28x28 pixel images, shape of input layer is [batch_size,28,28,1]

  '''
# Input Layer, converts feature map to this shape
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

# Convolutional Layer #1, applies 32 5x5 filters with ReLU activ. func
conv1 = tf.layers.conv2d(
  inputs=input_layer,
  filters=32,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)

# Pooling Layer #1, preforms max pooling with 2x2 filter and stride of 2
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 (64 5x5 filters, ReLU)
conv2 = tf.layers.conv2d(
  inputs=pool1,
  filters=64,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)

# Pooling Layer #2 (2x2 filer, stride: 2)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer, (1,024 neurons, dropout regularization rate: 0.4)
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

# Logits Layer (Dense layer #2, 10 neurons, one for each digit target class)
logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
  # Generate predictions (for PREDICT and EVAL mode)
  "classes": tf.argmax(input=logits, axis=1),
  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
  # `logging_hook`.
  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}

if mode == tf.estimator.ModeKeys.PREDICT:
return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# Calculate Loss (for both TRAIN and EVAL modes)
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

# Configure the Training Op (for TRAIN mode)
if mode == tf.estimator.ModeKeys.TRAIN:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())
return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

# Add evaluation metrics (for EVAL mode)
eval_metric_ops = {
  "accuracy": tf.metrics.accuracy(
      labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
