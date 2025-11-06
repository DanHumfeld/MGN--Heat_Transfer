# Code snippet to discover the hyperparameters necessary for a DNN with tanh
# activation functions to make an acceptable approximation of 1/norm, i.e.
# (x**2 + y**2)**(-1/2). The quality of fit is likely to be sensitive to the
# input data, as the function is unbounded as x, y approach zero. The code will
# be run on a fixed list of 5000 data points.
# Quality of fit was thrown off by one datum with x,y close to zero. That datum
# has been removed, yielding a model more accurate over only a certain range
# of 1/norm (up to 10).

#################################################################
# Imports
#################################################################
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import csv
from time import time

#################################################################
# Fixed Inputs
#################################################################
# Fixed parameters
input_count = 2
output_count = 1
training_epochs = 1000
trials = 10
data_file_name = 'data_norm.csv'
results_file_name = 'inverse_norm_fitting.csv'
output_file_name = 'inverse_example.csv'

# Optimizer parameters
learning_rate = 0.001 #0.0005
my_optimizer = optimizers.Adam(learning_rate)
initializer = 'glorot_uniform'

#################################################################
# Data Preparation
#################################################################
if os.path.exists(data_file_name):
    input_data_df = pd.read_csv(data_file_name, dtype=np.float32)
    data_x = np.array(input_data_df['x'])
    data_y = np.array(input_data_df['y'])
    data_r = np.array(input_data_df['norm'])
    x_values = np.column_stack([data_x, data_y])
    y_values = data_r.reshape((len(data_r),1))
    #print(x_values.shape)
    #print(y_values.shape)
else:
    print('Anticipated data file does not exist.')
    exit()

#################################################################
# Main Loop
#################################################################
for trial in range(trials):
    # Variable model hyperparameters
    width = 30
    depth = trial

    # Screen reporting
    print("Beginning trial: Depth:", depth, "  Width:", width)
    start_time = time()

    # Build model
    inputs = keras.Input(shape=(input_count,))
    dense = layers.Dense(width, activation="tanh")(inputs)
    for layer_number in range(depth-1):
        dense = layers.Dense(width, activation="tanh")(dense)
    outputs = layers.Dense(output_count, activation="linear")(dense)
    model = keras.Model(inputs=inputs, outputs=outputs, 
                        name="inv_norm_approximator")
    model.compile(loss = 'mse', optimizer = my_optimizer)

    # Train model
    history = model.fit(x_values, y_values, epochs=training_epochs, verbose=0)
    final_loss = history.history['loss'][-1]

    # Save results
    results_line = [depth, width, training_epochs, final_loss]
    with open(results_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results_line)

    if (depth == 2):
        results = np.concatenate((x_values, model(x_values)),axis=1)
        np.savetxt(output_file_name, results, delimiter=',') 

    # Screen acknowledgment
    duration = round(time() - start_time, 2)
    print("Results: Depth:", depth, "  Width:", width, 
          "  Loss:", final_loss, "  Duration (s):", duration)

# As collated in 'inverse_norm_fitting.csv', 2 hidden layers is enough to 
# achieve a good approximation of 1/sqrt(x^2 + y^2), whether training for
# 100 epochs or 1000 epochs.