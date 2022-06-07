import tensorflow as tf
import numpy as np
import keras
from keras import Input
from keras.layers import Dense


def gen():
    a = np.random.normal(0, 1, 10).astype(dtype=np.float32)
    b = np.random.normal(0, 1, 5).astype(dtype=np.float32)
    yield {"input_1": a, "input_2": b}, 1


def gen2():
    ragged_tensor = tf.ragged.constant([np.random.normal(0, 1, 10), np.random.normal(0, 1, 5)], dtype=tf.float32)
    yield ragged_tensor, ragged_tensor


def gen3():
    a = np.random.normal(0, 1, 10).astype(dtype=np.float32)
    b = np.random.normal(0, 1, 5).astype(dtype=np.float32)
    yield {"input_1": a, "input_2": b}, {"output_1": a, "output_2": b}


def gen4():
    a = np.random.normal(0, 1, 10).astype(dtype=np.float32)
    b = np.random.normal(0, 1, 5).astype(dtype=np.float32)
    yield {"input_1": a, "input_2": b}


dataset = tf.data.Dataset.from_generator(
    gen3,
    output_signature=(
        {"input_1": tf.TensorSpec(shape=None, dtype=tf.float32), "input_2": tf.TensorSpec(shape=None, dtype=tf.float32)},
        {"output_1": tf.TensorSpec(shape=None, dtype=tf.float32), "output_2": tf.TensorSpec(shape=None, dtype=tf.float32)}
    )
).batch(64)

print(list(dataset.take(1)))

u_dim = 10
i_dim = 5
latent_dim = 3

u_input = Input(10, dtype='float32', name='input_1')
u_input_c = Input(5, dtype='float32', name='input_2')

u_encoded = Dense(10, activation='linear', name="output_1")(u_input)
u_encoded_c = Dense(5, activation='linear', name="output_2")(u_input_c)

u_autoencoder = keras.Model([u_input, u_input_c], [u_encoded, u_encoded_c])

u_autoencoder.compile(optimizer=tf.optimizers.Adagrad(learning_rate=0.001), loss='mean_squared_error')
u_autoencoder.fit(dataset, batch_size=64, epochs=1, verbose=1, shuffle=True)
