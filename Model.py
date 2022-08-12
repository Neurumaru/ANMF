import os
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop


def get_model(u_dim, i_dim, latent_dim, noise, ld, delta):
    # regularizers
    u_regularizer = keras.regularizers.L2(ld)
    i_regularizer = keras.regularizers.L2(delta)

    # user auto-encoder
    u_input = Input(shape=(u_dim,), dtype='float32', name='u_input')
    u_input_c = Input(shape=(i_dim,), dtype='float32', name='u_input_c')

    u_gaussian = keras.layers.GaussianNoise(noise)(u_input)
    u_gaussian_c = keras.layers.GaussianNoise(noise)(u_input_c)

    x = keras.layers.concatenate([u_gaussian, u_gaussian_c])
    u_middle = Dense(latent_dim, activation='relu', kernel_regularizer=u_regularizer, name='u_middle')(x)

    u_decoded = Dense(u_dim, activation='sigmoid', kernel_regularizer=u_regularizer, name='u_output')(u_middle)
    u_decoded_c = Dense(i_dim, activation='sigmoid', kernel_regularizer=u_regularizer, name='u_output_c')(u_middle)

    # item auto-encoder
    i_input = Input(shape=(i_dim,), dtype='float32', name='i_input')
    i_input_c = Input(shape=(u_dim,), dtype='float32', name='i_input_c')

    i_gaussian = keras.layers.GaussianNoise(noise)(i_input)
    i_gaussian_c = keras.layers.GaussianNoise(noise)(i_input_c)

    x = keras.layers.concatenate([i_gaussian, i_gaussian_c])
    i_middle = Dense(latent_dim, activation='relu', kernel_regularizer=i_regularizer, name='i_middle')(x)

    i_decoded = Dense(i_dim, activation='sigmoid', kernel_regularizer=i_regularizer, name='i_output')(i_middle)
    i_decoded_c = Dense(u_dim, activation='sigmoid', kernel_regularizer=i_regularizer, name='i_output_c')(i_middle)

    # prediction model
    predict_vector = keras.layers.Multiply()([u_middle, i_middle])
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform', name='prediction')(predict_vector)
    
    hybrid_model = Model([u_input, u_input_c, i_input, i_input_c], [prediction, u_decoded, u_decoded_c, i_decoded, i_decoded_c])
    prediction_model = Model([u_input, u_input_c, i_input, i_input_c], prediction)
    
    return hybrid_model, prediction_model


def compile_model(model, learner, learning_rate, alpha, beta, phi, psi):
    user_loss_weight = phi * alpha
    user_c_loss_weight = phi * (1 - alpha)
    item_loss_weight = psi * beta
    item_c_loss_weight = psi * (1 - beta)
    # [1, user_loss_weight, user_c_loss_weight, item_loss_weight, item_c_loss_weight]
    loss_weights = {
        "prediction": 1,
        "u_output": user_loss_weight,
        "u_output_c": user_c_loss_weight,
        "i_output": item_loss_weight,
        "i_output_c": item_c_loss_weight
    }
    loss = {
        "prediction": 'binary_crossentropy',
        "u_output": 'mean_squared_error',
        "u_output_c": 'mean_squared_error',
        "i_output": 'mean_squared_error',
        "i_output_c": 'mean_squared_error'
    }

    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss=loss, loss_weights=loss_weights)
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=loss, loss_weights=loss_weights)
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, loss_weights=loss_weights)
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss=loss, loss_weight=loss_weights)


def save_weights(model, prediction_model, meta, checkpoint_dir):
    if os.path.isdir(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    checkpoint_meta = f'{checkpoint_dir}/checkpoint.json'
    checkpoint_model = f'{checkpoint_dir}/model.h5'
    checkpoint_pred_model = f'{checkpoint_dir}/prediction_model.h5'
    model.save_weights(checkpoint_model)
    prediction_model.save_weights(checkpoint_pred_model)
    with open(checkpoint_meta, 'w') as f:
        json.dump(meta, f)


def load_weights(model, prediction_model, meta, checkpoint_dir):
    checkpoint_meta = f'{checkpoint_dir}/checkpoint.json'
    checkpoint_model = f'{checkpoint_dir}/model.h5'
    checkpoint_pred_model = f'{checkpoint_dir}/prediction_model.h5'
    if os.path.isfile(checkpoint_meta) and os.path.isfile(checkpoint_model) and os.path.isfile(checkpoint_pred_model):
        model.load_weights(checkpoint_model)
        prediction_model.load_weights(checkpoint_pred_model)
        with open(checkpoint_meta, 'r') as f:
            meta = json.load(f)
    return model, prediction_model, meta


def fit_model_one_epoch(model, dataset, batch_size, verbose=1):
    model.fit(dataset, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
