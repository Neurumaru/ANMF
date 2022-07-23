from _ast import arg

import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from time import time

neg_sample = None


def get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg):
    global neg_sample
    neg_sample = neg

    dataset = tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            {
                "u_input": tf.TensorSpec(shape=None, dtype=tf.float32),
                "u_input_c": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_input": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_input_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            },
            {
                "prediction": tf.TensorSpec(shape=(), dtype=tf.int32),
                "u_output": tf.TensorSpec(shape=None, dtype=tf.float32),
                "u_output_c": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_output": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_output_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            }
        ),
        args=(train, num_negatives, uSimMat, iSimMat, DiDrAMat)
    ).shuffle(4096).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def get_generator(train, num_negatives, uSimMat, iSimMat, DiDrAMat):
    indices = np.arange(len(train))
    np.random.shuffle(indices)

    for index in indices:
        # positive instance
        instance = train[index]
        u = int(instance[0])
        i = int(instance[1])
        user_input = np.array(uSimMat[u])
        item_input = np.array(iSimMat[i])
        user_input_c = np.array(DiDrAMat[:, u])
        item_input_c = np.array(DiDrAMat[i])
        yield \
            {
                "u_input": user_input,
                "u_input_c": user_input_c,
                "i_input": item_input,
                "i_input_c": item_input_c
            }, \
            {
                "prediction": 1,
                "u_output": user_input,
                "u_output_c": user_input_c,
                "i_output": item_input,
                "i_output_c": item_input_c
            }

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(len(neg_sample[u]))
            ins = neg_sample[u][j]
            user_input = np.array(uSimMat[u])
            item_input = np.array(iSimMat[ins])
            user_input_c = np.array(DiDrAMat[:, u])
            item_input_c = np.array(DiDrAMat[ins])
            yield \
                {
                    "u_input": user_input,
                    "u_input_c": user_input_c,
                    "i_input": item_input,
                    "i_input_c": item_input_c
                }, \
                {
                    "prediction": 0,
                    "u_output": user_input,
                    "u_output_c": user_input_c,
                    "i_output": item_input,
                    "i_output_c": item_input_c
                }