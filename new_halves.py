import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learning_rate = 0.01
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
n_hidden = 100
n_half = 50
delta = 0.0001
delta_sqr = delta * delta

var_delta = tf.get_variable("var_delta", shape=(), trainable=False)

# First synaptic layer
# Consists of two halves
weights_11 = tf.Variable(tf.random_normal([n_input, n_half]))
bias_11 = tf.Variable(tf.random_normal([n_half]))
weights_12 = tf.Variable(tf.random_normal([n_input, n_half]))
bias_12 = tf.Variable(tf.random_normal([n_half]))

# Second layer
# Also consists of two halves
weights_21 = tf.Variable(tf.random_normal([n_half, n_classes]))
weights_22 = tf.Variable(tf.random_normal([n_half, n_classes]))
bias_2= tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b and sigmoidal activation

# Build approximate abs and neg_abs functions for the weights_11 and weights_12
weights_11_abs     = tf.sqrt( tf.add( tf.square(weights_11), delta_sqr ) )
weights_12_neg_abs = -1.0*tf.sqrt( tf.add( tf.square(weights_12), delta_sqr ) )

#weights_11_abs = tf.abs(weights_11)
#weights_12_neg_abs = tf.abs(weights_12)

# Use the functions in the logit calculations
logits_11 = tf.add(tf.matmul(features, weights_11_abs), bias_11, name="logits_11")
#a_11 = tf.sigmoid(logits_11, name="act_11")
a_11 = tf.nn.relu(logits_11, name="act_11")

logits_12 = tf.add(tf.matmul(features, weights_12_neg_abs), bias_12, name="logits_12")
#a_12 = tf.sigmoid(logits_12, name="act_12")

a_12 = tf.nn.relu(logits_12, name="act_12")

# Also apply abs/neg_abs functions to weights_21, weights_22

weights_21_abs = tf.sqrt( tf.add( tf.square(weights_21), delta_sqr ) )
weights_22_neg_abs = -1.0*tf.sqrt( tf.add( tf.square(weights_22), delta_sqr ) )

#weights_21_abs = tf.abs(weights_21)
#weights_22_neg_abs = -1.0*tf.abs(weights_22)

partial_logits = tf.add( tf.matmul(a_11, weights_21),
                         tf.matmul(a_12, weights_22), name="partial_logits" )


# Final logits
logits_2 = tf.add( partial_logits,  bias_2, name="logits_2" )

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=labels))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
#    .minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1)\
#    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits_2, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

import math

save_file = './train_model.ckpt'
batch_size = 256
n_epochs = 325

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

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


            print('Epoch {:<3} - Validation Accuracy: {}  Train Accuracy: {}'.format(
                epoch,
                valid_accuracy,
                train_accuracy))

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')