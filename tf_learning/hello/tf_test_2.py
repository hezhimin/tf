import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(1000).astype(np.float32)
#y_data = x_data * 0.3 + np.random.rand(1000).astype(np.float32) * 0.2
y_data = x_data * x_data

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(20000):
    sess.run(train)
    if step % 1000 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(W) * x_data * x_data + sess.run(b), label='Fitted line')
plt.legend()
plt.show()