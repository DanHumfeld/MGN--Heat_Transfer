# Code snippet for discovering the method to address a subset of Variables
# in a TensorFlow model, for use in calculating losses and gradients with
# respect to those losses.

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as k

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,), name='layer_1'),
])

# Access all trainable variables as a list of tf.Variable objects
trainable_params = model.trainable_variables
print(trainable_params[0])
modifiable = trainable_params[0]
n_feed = np.column_stack(np.arange(10))
n_feed = tf.Variable(n_feed.reshape(len(n_feed[0]),1), trainable=True, dtype=tf.float32)

components = tf.multiply(trainable_params[0], tf.reshape(n_feed, [1,10]))
components = components[0,0:4]
components = tf.multiply(components, 3)

with tf.GradientTape(persistent=True) as tape_1:
    # Watch parameters
    tape_1.watch(n_feed)

    # Define functions
    #print(n_feed)
    loss1 = k.sum(tf.multiply(trainable_params[0], tf.reshape(n_feed, [1,10])))
    print(loss1)
    loss2 = k.sum(tf.multiply(modifiable, tf.reshape(n_feed, [1,10])))

    #components = tf.multiply(trainable_params[0], tf.reshape(n_feed, [1,10]))
    #components = components[0,0:4]
    #components = tf.multiply(components, 3)
    print(components)
    loss3 = k.sum(components)
    print(loss3)

# Take gradient
gradients = tape_1.gradient(loss1, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
print(gradients)
gradients2 = tape_1.gradient(loss2, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
print(gradients2)
gradients3 = tape_1.gradient(loss3, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
print(gradients3)

# Correctly calculates the gradients of the functions with respect to the 
# trainable variables of the model. 
# 
# Shows that the assignment of the Python variable to the model Variable 
# can occur before the GradientTape.
#
# Shows the method of extracting a set of variables: do it after it's a Tensor 
# due to tf.multiply. This structure is easy to manipulate and has the 
# connection back to the 1 * the trainable variables. You can manipuate the 
# extracted variables into their contributions to the loss.