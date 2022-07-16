from _ast import arg

import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from time import time

neg_sample = None


# def get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg):
def get_dataset(train, uSimMat, iSimMat, DiDrAMat):
    with ThreadPoolExecutor(2) as pool:
        # model_dataset = pool.submit(get_model_dataset, train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg)
        user_dataset = pool.submit(get_user_dataset, train, uSimMat, DiDrAMat)
        item_dataset = pool.submit(get_item_dataset, train, iSimMat, DiDrAMat)

        # model_dataset = model_dataset.result()
        user_dataset = user_dataset.result()
        item_dataset = item_dataset.result()

    # return model_dataset, user_dataset, item_dataset
    return user_dataset, item_dataset


def get_model_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg):
    global neg_sample
    neg_sample = neg

    start_time = time()
    # print(f'Start generating model dataset')
    model_dataset = tf.data.Dataset.from_generator(
        get_model_generator,
        output_signature=(
            {
                "u_input": tf.TensorSpec(shape=None, dtype=tf.float32),
                "u_input_c": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_input": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_input_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            },
            tf.TensorSpec(shape=(), dtype=tf.int32)
        ),
        args=(train, num_negatives, uSimMat, iSimMat, DiDrAMat)
    ).shuffle(4096).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    # print(f'End generating model dataset | TOTAL:{time()-start_time:.2f}s')

    return model_dataset


def get_user_dataset(train, uSimMat, DiDrAMat):
    start_time = time()
    # print(f'Start generating user dataset')
    user_dataset = tf.data.Dataset.from_generator(
        get_user_generator,
        output_signature=(
            {
                "u_input": tf.TensorSpec(shape=None, dtype=tf.float32),
                "u_input_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            }, {
                "u_output": tf.TensorSpec(shape=None, dtype=tf.float32),
                "u_output_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            }
        ),
        args=(train, uSimMat, DiDrAMat)
    ).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    # print(f'End generating user dataset | TOTAL:{time()-start_time:.2f}s')

    return user_dataset


def get_item_dataset(train, iSimMat, DiDrAMat):
    start_time = time()
    # print(f'Start generating item dataset')
    item_dataset = tf.data.Dataset.from_generator(
        get_item_generator,
        output_signature=(
            {
                "i_input": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_input_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            }, {
                "i_output": tf.TensorSpec(shape=None, dtype=tf.float32),
                "i_output_c": tf.TensorSpec(shape=None, dtype=tf.float32)
            }
        ),
        args=(train, iSimMat, DiDrAMat)
    ).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    # print(f'End generating item dataset | TOTAL:{time()-start_time:.2f}s')

    return item_dataset


def get_model_generator(train, num_negatives, uSimMat, iSimMat, DiDrAMat):
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
            }, 1

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
                }, 0


def get_user_generator(train, uSimMat, DiDrAMat):
    indices = np.arange(len(train))
    np.random.shuffle(indices)
    for index in indices:
        instance = train[index]
        u = int(instance[0])
        user_input = np.array(uSimMat[u])
        user_input_c = np.array(DiDrAMat[:, u])
        yield \
            {"u_input": user_input, "u_input_c": user_input_c}, \
            {"u_output": user_input, "u_output_c": user_input_c}


def get_item_generator(train, iSimMat, DiDrAMat):
    indices = np.arange(len(train))
    np.random.shuffle(indices)
    for index in indices:
        instance = train[index]
        i = int(instance[1])
        item_input = np.array(iSimMat[i])
        item_input_c = np.array(DiDrAMat[i])
        yield \
            {"i_input": item_input, "i_input_c": item_input_c}, \
            {"i_output": item_input, "i_output_c": item_input_c}