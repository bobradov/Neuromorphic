import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import mlp_unipolar_nn as uni

# Use fixed random sequences
tf.set_random_seed(1234)

learning_rate = 0.005
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
n_hidden = 100
alpha = 0.5


# Assign variable for approximate absolute value
delta = tf.Variable(initial_value=0.01, name="delta",  trainable=False)
delta_sqr = delta * delta
print('type of delta: ', type(delta))
print('type of delta_sqr: ', type(delta_sqr))

print('delta: ', delta)

        

#
mlp = uni.mlp_unipolar_nn(n_input, n_classes, n_hidden, alpha, delta_sqr, features)

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=mlp.logits, labels=labels))


#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
#    .minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1)\
#    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(mlp.logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

import math

save_file = './train_model.ckpt'
batch_size = 256
n_epochs = 200

best_valid = -1
best_train = -1
best_save_file = './best_model.ckpt'

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Before load, delta=', sess.run(delta))
    delta.load(0.01, sess)
    print('After load, delta=', sess.run(delta))

    # Training cycle
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                optimizer,
                feed_dict={features: batch_features, labels: batch_labels})

        # Print status for every 10 epochs
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})

            train_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: batch_features,
                    labels: batch_labels})

            if valid_accuracy > best_valid:
                best_valid = valid_accuracy
                best_train = train_accuracy
                saver.save(sess, best_save_file)
                print('Best model saved.')


            print('Epoch {:<3} - Validation Accuracy: {}  Train Accuracy: {}'.format(
                epoch,
                valid_accuracy,
                train_accuracy))

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')

    print('Best accuracies: valid: ', best_valid, ' train: ', best_train)