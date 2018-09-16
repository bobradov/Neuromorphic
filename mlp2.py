""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf


class mlp_nn(object):
    '''
    Object for constructing MLPs
    '''


    def __init__(self, layers, X, Y):
        self.n_inputs = layers[0]
        self.n_classes = layers[-1]
        self.n_layers = len(layers)

        # Create weights and biases
        self.weights = [ tf.Variable(tf.random_normal( [layers[i], layers[i+1] ]),
                                name='weight_' + str(i))
                                for i in range(0, self.n_layers - 1) ]

        print('weights: ', self.weights)

        self.biases = [ tf.Variable(tf.random_normal( [layers[i+1]]),
                                name='bias_' + str(i))
                                for i in range(0, self.n_layers - 1) ]

        print('biases: ', self.biases)


        # Create NN function     
        cur_model = X
        for layer_index in range(0, self.n_layers - 1):
            # Add a z-value
            cur_model = tf.add(
                                tf.matmul(cur_model, self.weights[layer_index]), 
                                self.biases[layer_index],
                                name='logit_' + str(layer_index)
                                )       
            # Add activation
            # Sigmoid, except at last layer
            if layer_index < self.n_layers - 2:
                print('Adding sigmoid ...')
                cur_model = tf.sigmoid(cur_model, 
                                        name='sigmoid_' + str(layer_index))
                #cur_model = tf.nn.relu(cur_model)

        # Store completed model
        self.logits = cur_model
        self.pred = tf.nn.softmax(self.logits, name='Prediction')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=Y), name='Loss')
        
            
                


# Parameters
learning_rate = 0.01
training_epochs = 300
batch_size = 256
display_step = 20

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input], name='X')
Y = tf.placeholder("float", [None, n_classes], name='Y')



# Construct model
nn_model = mlp_nn([784, 200, 10], X, Y)
#nn_model = mlp_nn([n_input, 50, n_classes], X, Y)
#nn_model = mlp_nn([n_input, 50, 50, 50, 50, n_classes], X, Y)
#logits = nn_model.model

#print(logits)

#exit()



# Define loss and optimizer
loss_op = nn_model.loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optim')
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, name='optim_min')
# Initializing the variables
init = tf.global_variables_initializer()


# Test model
pred = nn_model.pred  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1),
                            name='correct_prediction')
# Calculate accuracy
test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),
                                name='accuracy')

with tf.Session() as sess:
    sess.run(init)

   

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), end=' ')

            # Display test accuracy per epoch
            print("Train Accuracy:", test_accuracy.eval( {X: batch_x, Y: batch_y} ), end=' ')
            print("Test Accuracy:", test_accuracy.eval(  
                                     {X: mnist.test.images, Y: mnist.test.labels} ) )


    print("Optimization Finished!")
    print("Test Accuracy:", test_accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

     # Save the graph structure
    print('type:')
    print(type(nn_model.pred))
    writer = tf.summary.FileWriter('./graph_logs', graph=sess.graph)

    
    