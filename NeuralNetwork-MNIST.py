import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#data
x = tf.placeholder(tf.float32,shape=[None,784])

#prediction
y_ = tf.placeholder(tf.float32,shape=[None,10])

#Weights
W = tf.Variable(tf.zeros([784,10]))

#bias
b = tf.Variable(tf.zeros([10]))

#neural network model
model = tf.nn.softmax(tf.matmul(x,W)+b)

#cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=model))

#optimizer
learning_rate = 0.5
optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

iterations = 1000

for iter in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(optimize, feed_dict = {x:batch_xs, y_:batch_ys})

corrected_prediction = tf.equal(tf.argmax(model,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(corrected_prediction,tf.float32))
test_accuracy = session.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
print("Test Accuracy {0}%".format(test_accuracy*100.0))
session.close()