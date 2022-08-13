import numpy as np
import tensorflow as tf

from time import time
from progress import progress


def predict_model(model, test, uSimMat, iSimMat, DiDrAMat, batch_size=1024):
    dataset = {
        "u_input": [],
        "u_input_c": [],
        "i_input": [],
        "i_input_c": []
    }
    result = list()
    start_time = time()
    for idx, (u, i, _) in enumerate(test):
        if idx % 10 == 0:
            progress(idx, len(test), start_time)

        user_input = np.array(uSimMat[u])
        item_input = np.array(iSimMat[i])
        user_input_c = np.array(DiDrAMat[:, u])
        item_input_c = np.array(DiDrAMat[i])

        dataset['u_input'].append(user_input)
        dataset['u_input_c'].append(user_input_c)
        dataset['i_input'].append(item_input)
        dataset['i_input_c'].append(item_input_c)

        if (idx+1) % batch_size == 0:
            tmp = tf.data.Dataset.from_tensors(dataset)
            predict = model.predict(tmp)
            result.extend(predict)

            dataset['u_input'].clear()
            dataset['u_input_c'].clear()
            dataset['i_input'].clear()
            dataset['i_input_c'].clear()

    if len(dataset['u_input']) > 0:
        tmp = tf.data.Dataset.from_tensors(dataset)
        predict = model.predict(tmp)
        result.extend(predict)

    u, i, r = zip(*test)
    return list(zip(u, i, np.array(result).reshape(-1)))


def calculate_AUC(y_true, y_pred):
    Pos, Neg = set(), set()

    for user, item, score in y_true:
        if score == 1:
            Pos.add((user, item))
        else:
            Neg.add((user, item))

    y_pred.sort(key=lambda x: -x[2])

    TP, FP = 0, 0
    TP_sum = 0
    for user, item, score in y_pred:
        if (user, item) in Pos:
            TP += 1
        elif (user, item) in Neg:
            FP += 1
            TP_sum += TP
    AUC = TP_sum / (TP * FP)

    return AUC