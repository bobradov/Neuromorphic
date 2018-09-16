
import tensorflow as tf

'''
Pair of functions for defining approximate absolute value
(positive and negative).
Parameterized by the single parameter delta_sqr
'''
def abs_fun(weights, delta_sqr):
    return tf.sqrt( tf.add( tf.square(weights), delta_sqr ) )

def neg_abs_fun(weights, delta_sqr):
    return -1.0*tf.sqrt( tf.add( tf.square(weights), delta_sqr ) )

class mlp_unipolar_nn(object):
    '''
    Object for constructing MLPs
    The constructed MLPs have dedicated positive and negative weights

    Inputs:
        1. alpha : fraction of positive weights
        2. delta_sqr : tf function which computes D^2 for approximate
                       absolute value
        3. features: input X vector
    '''


    def __init__(self, inputs, classes, hidden, alpha, delta_sqr, features):
        self.n_inputs  = inputs
        self.n_classes = classes
        self.n_hidden  = hidden
        self.n_layers  = 4

        n_plus = int(alpha * self.n_hidden)
        n_minus = self.n_hidden - n_plus

        # First synaptic layer
        # Consists of two halves
        weights_10_plus = tf.Variable(tf.random_normal([self.n_inputs, n_plus]))
        weights_10_abs  = abs_fun(weights_10_plus, delta_sqr)
        bias_1_plus     = tf.Variable(tf.random_normal([n_plus]))

        weights_10_minus    = tf.Variable(tf.random_normal([self.n_inputs, n_minus]))
        weights_10_neg_abs  = neg_abs_fun(weights_10_minus, delta_sqr)
        bias_1_minus        = tf.Variable(tf.random_normal([n_minus]))

        # Second layer
        # Also consists of two halves
        weights_21_plus = tf.Variable(tf.random_normal([n_plus,  n_plus]))
        weights_22_plus = tf.Variable(tf.random_normal([n_minus, n_plus]))
        weights_21_abs  = abs_fun(weights_21_plus, delta_sqr)
        weights_22_abs  = abs_fun(weights_22_plus, delta_sqr)


        weights_21_minus = tf.Variable(tf.random_normal([n_plus,  n_minus]))
        weights_22_minus = tf.Variable(tf.random_normal([n_minus, n_minus]))
        weights_21_neg_abs  = neg_abs_fun(weights_21_minus, delta_sqr)
        weights_22_neg_abs  = neg_abs_fun(weights_22_minus, delta_sqr)

        bias_2_plus     = tf.Variable(tf.random_normal([n_plus]))
        bias_2_minus    = tf.Variable(tf.random_normal([n_minus]))

        # Final layer
        weights_32_plus  = tf.Variable(tf.random_normal([n_plus,  self.n_classes]))
        weights_32_minus = tf.Variable(tf.random_normal([n_minus, self.n_classes]))

        #weights_32_abs     = abs_fun(weights_32_plus, delta_sqr)
        #weights_32_neg_abs = neg_abs_fun(weights_32_minus, delta_sqr)

        bias_3           = tf.Variable(tf.random_normal([self.n_classes]))

        # Logits - xW + b and ReLU activation

        z_11 = tf.add( tf.matmul(features, weights_10_abs), bias_1_plus)
        z_12 = tf.add( tf.matmul(features, weights_10_neg_abs), bias_1_minus)
        a_11 = tf.nn.relu(z_11, name="a_11")
        a_12 = tf.nn.relu(z_12, name="a_12")

        z_21_partial = tf.add( tf.matmul(a_11, weights_21_abs), tf.matmul(a_12, weights_22_abs) )
        z_21 = tf.add(z_21_partial, bias_2_plus)
        a_21 = tf.nn.relu(z_21, name="z_21")

        z_22_partial = tf.add( tf.matmul(a_11, weights_21_neg_abs), tf.matmul(a_12, weights_22_neg_abs) )
        z_22 = tf.add(z_22_partial, bias_2_minus)
        a_22 = tf.nn.relu(z_22, name="z_22")

        # Final layer
        a_11_shape = a_11.get_shape()
        a_12_shape = a_12.get_shape()
        a_21_shape = a_21.get_shape()
        a_22_shape = a_22.get_shape()
        weights_32_plus_shape = weights_32_plus.get_shape()
        weights_32_minus_shape = weights_32_minus.get_shape()
        print('a_11:', a_11_shape, ' a_12:', a_12_shape, '\na_21: ', a_21_shape, ' a_22: ', a_22_shape, 
            '\nw32+:', weights_32_plus_shape, ' w32-: ', weights_32_minus_shape )

        #z_3_partial = tf.add( tf.matmul(a_21, weights_32_abs), tf.matmul(a_22, weights_32_neg_abs) )
        z_3_partial = tf.add( tf.matmul(a_21, weights_32_plus), tf.matmul(a_22, weights_32_minus) )
        z_3 = tf.add(z_3_partial, bias_3)
                
        self.logits = z_3