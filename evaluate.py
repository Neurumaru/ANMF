from time import time
from progress import progress, progressEnd
import numpy as np
import tensorflow as tf


def predict_model(model, test, uSimMat, iSimMat, DiDrAMat, batch_size=1024, verbose=10):
    dataset = {
        "u_input": [],
        "u_input_c": [],
        "i_input": [],
        "i_input_c": []
    }
    result = list()

    start_time = time()
    print(f'Predict using model')
    for idx, (u, i, r) in enumerate(test):
        if verbose != 0 and idx % verbose == 0:
            progress(idx, len(test), start_time, (u, i))

        user_input = np.array(uSimMat[u])
        item_input = np.array(iSimMat[i])
        user_input_c = np.array(DiDrAMat[:, u])
        item_input_c = np.array(DiDrAMat[i])

        dataset['u_input'].append(user_input)
        dataset['u_input_c'].append(item_input)
        dataset['i_input'].append(user_input_c)
        dataset['i_input_c'].append(item_input_c)

        if (idx+1) % batch_size == 0:
            progress(idx, len(test), start_time, 'Predicting')

            tmp = tf.data.Dataset.from_tensors(dataset)
            predict = model.predict(tmp)
            result.extend(predict)

            dataset['u_input'].clear()
            dataset['u_input_c'].clear()
            dataset['i_input'].clear()
            dataset['i_input_c'].clear()

    if len(dataset['u_input']) > 0:
        progress(len(test), len(test), start_time, 'Predicting')
        tmp = tf.data.Dataset.from_tensors(dataset)
        predict = model.predict(tmp)
        result.extend(predict)

    progressEnd(len(test), start_time)
    u, i, r = zip(*test)
    return list(zip(u, i, np.array(result).reshape(-1)))
