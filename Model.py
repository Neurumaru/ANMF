import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop


def get_model(u_dim, i_dim, latent_dim):
    # user auto-encoder
    u_input = Input(shape=(u_dim,), dtype='float32', name='u_input')
    u_input_c = Input(shape=(i_dim,), dtype='float32', name='u_input_c')

    u_encoded = Dense(latent_dim, activation='linear')(u_input)
    u_encoded_c = Dense(latent_dim, activation='linear')(u_input_c)
    x = keras.layers.concatenate([u_encoded, u_encoded_c])
    # u_encoded = Dense(64, activation='relu')(u_encoded)
    u_middle = Dense(latent_dim, activation='relu', name='u_middle')(x)
    # u_decoded = Dense(64, activation='relu')(u_middle)
    # u_decoded = Dense(128, activation='relu')(u_decoded)
    u_decoded = Dense(u_dim, activation='sigmoid', name='u_output')(u_middle)
    u_decoded_c = Dense(i_dim, activation='sigmoid', name='u_output_c')(u_middle)

    u_autoencoder = Model([u_input, u_input_c], [u_decoded, u_decoded_c])

    # item auto-encoder
    i_input = Input(shape=(i_dim,), dtype='float32', name='i_input')
    i_input_c = Input(shape=(u_dim,), dtype='float32', name='i_input_c')
    # i_encoded = Dense(64, activation='relu')(i_input)
    i_encoded = Dense(latent_dim, activation='linear')(i_input)
    i_encoded_c = Dense(latent_dim, activation='linear')(i_input_c)
    x = keras.layers.concatenate([i_encoded, i_encoded_c])
    i_middle = Dense(latent_dim, activation='relu', name='i_middle')(x)
    # i_decoded = Dense(64, activation='relu')(i_middle)
    i_decoded = Dense(i_dim, activation='sigmoid', name='i_output')(i_middle)
    i_decoded_c = Dense(u_dim, activation='sigmoid', name='i_output_c')(i_middle)

    i_autoencoder = Model([i_input, i_input_c], [i_decoded, i_decoded_c])

    predict_vector = keras.layers.Multiply()([u_middle, i_middle])
    prediction = Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform',
        name='prediction')(predict_vector)
    hybrid_model = Model([u_input, u_input_c, i_input, i_input_c], prediction)
    return hybrid_model, u_autoencoder, i_autoencoder


def compile_model(model, u_autoencoder, i_autoencoder, learner, learning_rate):
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
        u_autoencoder.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='mean_squared_error')
        i_autoencoder.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
        u_autoencoder.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mean_squared_error')
        i_autoencoder.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
        u_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        i_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')
        u_autoencoder.compile(optimizer=SGD(learning_rate=learning_rate), loss='mean_squared_error')
        i_autoencoder.compile(optimizer=SGD(learning_rate=learning_rate), loss='mean_squared_error')


def fit_model_one_epoch(model, u_autoencoder, i_autoencoder, model_dataset, user_dataset, item_dataset, batch_size):
    u_autoencoder.fit(user_dataset, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
    i_autoencoder.fit(item_dataset, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
    model.fit(model_dataset, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
