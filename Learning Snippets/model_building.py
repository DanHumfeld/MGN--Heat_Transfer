import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parameter_count_ee = 30
parameter_count_ev = 30
parameter_count_d = 120
parameter_count_df = 320

# Build a model that has multiple structures built into it

# Build the head of the model
inputs = keras.Input(shape=(1,))

# Build each of three structures; a one-layer, another one-layer and a 
# three-layer dense sub-model.
dense_ee = layers.Dense(parameter_count_ee, activation="linear")(inputs)

dense_ev = layers.Dense(parameter_count_ev, activation="linear")(inputs)

dense_d = layers.Dense(parameter_count_d, activation="tanh")(inputs)
dense_d = layers.Dense(parameter_count_d, activation="tanh")(dense_d)
dense_d = layers.Dense(parameter_count_df, activation="tanh")(dense_d)

# Concatenate the final layer of these structures 
concatenated = layers.Concatenate(axis=-1)([dense_ee, dense_ev, dense_d])

# Define an (unused) output layer and finalize the model
outputs = layers.Dense(1)(concatenated)
model = keras.Model(inputs=inputs, outputs=outputs, name="test_model")

# Show the structure of the model. 
model.summary()

# Demonstrates that all of the variables required for all of the structures
# that are desired for multiple models, can easily be built all into a single
# model, for the purpose of being able to access all variables and take the 
# gradient of the loss with respect to any of them, where that loss is 
# calculated from the weights and other things, and *maybe submodels* (not
# yet tested) rather than the output of the model.
