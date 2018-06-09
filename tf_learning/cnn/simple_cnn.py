import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    values=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(values)

def bias_variable(shape):
    values=tf.constant(0.1, shape=shape)
    return tf.Variable(values)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


input=input_data.read_data_sets("MNIST_data/", one_hot=True)
print(input.train.images.shape, input.train.labels.shape)
print(input.test.images.shape, input.test.labels.shape)
print(input.validation.images.shape, input.validation.labels.shape)

sess=tf.InteractiveSession()

in_units=784
h1_units=300
for dev in ["/gpu:0"]:
    with tf.device(dev):
        x=tf.placeholder(tf.float32,[None, 784])
        y_=tf.placeholder(tf.float32,[None,10])
        keep_prob=tf.placeholder(tf.float32)
        x_image=tf.reshape(x, [-1,28,28,1])

        W_conv1=weight_variable([5,5,1,32])
        b_conv1=bias_variable([32])
        h_conv1=tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1=max_pool_2x2(h_conv1)

        W_conv2=weight_variable([5,5,32,64])
        b_conv2=bias_variable([64])
        h_conv2=tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2=max_pool_2x2(h_conv2)

        W_fc1=weight_variable([7*7*64, 1024])
        b_fc1=bias_variable([1024])
        h_pool2_flat=tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)

        W_fc2=weight_variable([1024,10])
        b_fc2=bias_variable([10])
        y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
        #train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        tf.global_variables_initializer().run()

for i in range(10000):
    batch_x,batch_y=input.train.next_batch(100)
    train_step.run({x:batch_x, y_:batch_y, keep_prob: 0.8})
    if i%100 == 0:
        correct_predictions=tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        print(i, accuracy.eval({x:input.test.images, y_:input.test.labels, keep_prob:1.0}))