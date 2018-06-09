import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input=input_data.read_data_sets("MNIST_data/", one_hot=True)
print(input.train.images.shape, input.train.labels.shape)
print(input.test.images.shape, input.test.labels.shape)
print(input.validation.images.shape, input.validation.labels.shape)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(2000):
    batch_x,batch_y=input.train.next_batch(100)
    train_step.run({x:batch_x, y_:batch_y})
    correct_predictions=tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print(i, accuracy.eval({x:input.test.images, y_:input.test.labels}))