# Code snippet for discovering the method to build a model containing the 
# trainable variables from a collection of other models, to ensure that the
# model.gradient(loss, model.trainable_variables) captures all dependencies.
# It is possible that calling model_mpe then model_mpv repeatedly might not
# enable model_mpe.gradient(loss, model_mpe.trainable_variables) to calculate
# all of the dependencies correctly.
# It is also possible that the separate-only model structure would trace all
# dependencies successfully. Knowing this construction, method, it will be 
# easy to take the implementation of the MeshGraphNet with separate-only models
# and bulid a top-level model to determine if the top-level model is necessary
# or not.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parameter_count_v = 12
parameter_count_e = 4
latent_space_dimension = 120

# Build each desired model
inputs_ee = keras.Input(shape=(parameter_count_e,))
dense_ee = layers.Dense(latent_space_dimension, activation="tanh")(inputs_ee)
dense_ee = layers.Dense(latent_space_dimension, activation="tanh")(dense_ee)
dense_ee = layers.Dense(latent_space_dimension, activation="tanh")(dense_ee)
model_ee = keras.Model(inputs=inputs_ee, outputs=dense_ee, 
                       name="edge_encoder")

inputs_ev = keras.Input(shape=(parameter_count_v,))
dense_ev = layers.Dense(latent_space_dimension, activation="tanh")(inputs_ev)
dense_ev = layers.Dense(latent_space_dimension, activation="tanh")(dense_ev)
dense_ev = layers.Dense(latent_space_dimension, activation="tanh")(dense_ev)
model_ev = keras.Model(inputs=inputs_ev, outputs=dense_ev, 
                       name="vertex_encoder")

inputs_mpe = keras.Input(shape=(2*latent_space_dimension,))
dense_mpe = layers.Dense(latent_space_dimension, activation="tanh")(inputs_mpe)
dense_mpe = layers.Dense(latent_space_dimension, activation="tanh")(dense_mpe)
dense_mpe = layers.Dense(latent_space_dimension, activation="tanh")(dense_mpe)
model_mpe = keras.Model(inputs=inputs_mpe, outputs=dense_mpe, 
                        name="message_passing_edge")

inputs_mpv = keras.Input(shape=(4*latent_space_dimension,))
dense_mpv = layers.Dense(latent_space_dimension, activation="tanh")(inputs_mpv)
dense_mpv = layers.Dense(latent_space_dimension, activation="tanh")(dense_mpv)
dense_mpv = layers.Dense(latent_space_dimension, activation="tanh")(dense_mpv)
model_mpv = keras.Model(inputs=inputs_mpv, outputs=dense_mpv, 
                        name="message_passing_vertex")

# There is no edge_decoder

inputs_dv = keras.Input(shape=(latent_space_dimension,))
dense_dv = layers.Dense(latent_space_dimension, activation="tanh")(inputs_dv)
dense_dv = layers.Dense(latent_space_dimension, activation="tanh")(dense_dv)
dense_dv = layers.Dense(parameter_count_v, activation="tanh")(dense_dv)
model_dv = keras.Model(inputs=inputs_dv, outputs=dense_dv, 
                       name="vertex_decoder")

# Build a model that contains all of the prior models
# Concatenate the final layer of the previous models
#concatenated_inputs = layers.Concatenate(axis=-1)([inputs_ee, inputs_ev, inputs_mpe, inputs_mpv, inputs_dv])
concatenated_inputs = [inputs_ee, inputs_ev, inputs_mpe, inputs_mpv, inputs_dv]
concatenated_outputs = layers.Concatenate(axis=-1)(
    [dense_ee, dense_ev, dense_mpe, dense_mpv, dense_dv])

# Define an (unused) output layer and finalize the model
outputs = layers.Dense(1)(concatenated_outputs)
model = keras.Model(inputs=concatenated_inputs, outputs=outputs, 
                    name="test_model")

# Build the model using the concatenated layer as the output
model2 = keras.Model(inputs=concatenated_inputs, outputs=concatenated_outputs, 
                     name="test_model2")

# Show the structure of the model. 
model.summary()
model2.summary()

# Demonstrates that multiple models can be built independently so they are 
# available to be evaluated from their own inputs, and then that a top-level
# model can be built from those inputs and outputs so that the gradient of
# the loss can be calculated with respect to all of the parameters in all
# of the models. Use the model2 version, that does not have the extraneous
# dense layer at the end that serves no purpose.