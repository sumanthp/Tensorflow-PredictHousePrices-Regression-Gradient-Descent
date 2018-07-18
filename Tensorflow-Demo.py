import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
# from tensorflow.examples.tutorials.mnist import input_data

# mnist=input_data.read_data_sets('MNIST_data',one_hot=true)

# def Train_size(num):
#     print('Total Training images in dataset: '+str(mnist.train.images.shape))
#     print('----------------------------------------------------------------')
#     x_train=mnist.train.images[:num,:]
#     print('X_TRAIN examples taken : '+str(x_train.shape))
#     y_train=mnist.train.labels[:num,:]
#     print('X_TRAIN examples taken : '+str(y_train.shape))
#     print('')
#     print('----------------------------------------------------------------')
#     return x_train,y_train

# def Test_size(num):
#     print('Total Test images in dataset: '+str(mnist.test.images.shape))
#     print('----------------------------------------------------------------')
#     x_test=mnist.test.images[:num,:]
#     print('X_TEST examples taken : '+str(x_test.shape))
#     y_test=mnist.test.labels[:num,:]
#     print('Y_TRAIN examples taken : '+str(y_test.shape))
#     print('')
#     print('----------------------------------------------------------------')
#     return x_test, y_test

num_houses = 150
np.random.seed(45)
house_size = np.random.randint(low=1000,high=5000,size=num_houses)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=75000, size=num_houses)
plt.plot(house_size,house_price,"bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

def normalize(array):
    return (array - array.mean()) / array.std()

num_train_samples = math.floor(num_houses * 0.7)

#Train Data
train_house_size = np.asarray(house_size[:num_train_samples])
#train_price = np.asanyarray(house_price[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

#Train Data Normalized
train_house_size_normalized = normalize(train_house_size)
train_price_normalized = normalize(train_price)

#Test Data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

#Test Data Normalized
test_house_size_normalized = normalize(test_house_size)
test_house_price_normalized = normalize(test_house_price)

#Tensorflow placeholders of type float for house)size and house_price
tf_house_size_tensor = tf.placeholder("float", name="house_size")
tf_house_price_tensor = tf.placeholder("float", name="house_price")


tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

#Inference to predict house price
tf_predict_house_price = tf.add(tf.multiply(tf_size_factor,tf_house_size_tensor), tf_price_offset)

#Mean squared error Calculation
tf_mse_cost = tf.reduce_sum(tf.pow(tf_predict_house_price - tf_house_price_tensor, 2))/(2*num_train_samples)

#Define  Learning rate to optimized the prediction
learning_rate = 0.1

#Gradient Descent Optimizer to optimized the Mean Squared Error Loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_mse_cost)

#tensorflow variables are initialized
init = tf.global_variables_initializer()

#tensorflow variables are executed
with tf.Session() as session:
    session.run(init)

    display_every = 2
    num_iterations = 50

    for iteration in range(num_iterations):

        for(x, y) in zip(train_house_size_normalized, train_price_normalized):
            session.run(optimizer, feed_dict={tf_house_size_tensor: x, tf_house_price_tensor: y})
        
        if(iteration + 1) % display_every == 0:
            c = session.run(tf_mse_cost, feed_dict={tf_house_size_tensor: train_house_size_normalized, tf_house_price_tensor: train_price_normalized})
            print("iteration #:",'%04d' %(iteration + 1), "cost=", "{:9f}".format(c), \
                "size_factor=", session.run(tf_size_factor), "price_offset=", session.run(tf_price_offset))
    
    print("Training completed")
    training_cost = session.run(tf_mse_cost, feed_dict={tf_house_size_tensor: train_house_size_normalized, tf_house_price_tensor: train_price_normalized})
    print("Trained cost=", training_cost, "size_factor=", session.run(tf_size_factor), "price_factor=", session.run(tf_price_offset))

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("size (Sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training Data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing Data')
    plt.plot(train_house_size_normalized * train_house_size_std + train_house_size_mean,
            (session.run(tf_size_factor)*train_house_size_normalized + session.run(tf_price_offset))* train_price_std + train_price_mean,
            label='Learned Regression')
    plt.legend(loc='upper left')
    plt.show()
