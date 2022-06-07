# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:35:04 2018

@author: Administrator
"""
import gc
import tensorflow as tf
from multiprocessing.pool import Pool
from time import time
from DataLoader import load_c_matrix, load_matrix, load_rating_file_as_list, load_item_sim_file, load_user_sim_file, \
    load_negative_file
from Model import get_model, compile_model, fit_model_one_epoch
from Dataset import get_dataset

num_factors = 256
num_negatives = 10
learner = 'adam'
learning_rate = 0.001
epochs = 50
batch_size = 64
verbose = 1

drug = 11219
disease = 6322


def load_train():
    st = time()
    print(f'Start loading train.rating')
    result = load_rating_file_as_list('Data\\train.rating')
    print(f'End loading train.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_negative():
    st = time()
    print(f'Start loading negative.rating')
    result = load_negative_file('Data\\negative.rating', drug)
    print(f'End loading negative.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_drug_sim():
    st = time()
    print(f'Start loading DrugSim.txt')
    result = load_user_sim_file('Data\\DrugSim.txt')
    print(f'End loading DrugSim.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_disease_sim():
    st = time()
    print(f'Start loading DiseaseSim.txt')
    result = load_item_sim_file('Data\\DiseaseSim.txt')
    print(f'End loading DiseaseSim.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_didra():
    st = time()
    print(f'Start loading DiDrA.txt')
    result = load_c_matrix('Data\\DiDrA.txt')
    print(f'End loading DiDrA.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_gt():
    st = time()
    print(f'Start loading GT.txt')
    result = load_matrix('Data\\GT.txt')
    print(f'End loading GT.txt | TOTAL:{time() - st:.2f}s')
    return result


if __name__ == '__main__':
    print(f'==================== Dataset ====================')
    print()

    pool = Pool(6)
    train = pool.apply_async(load_train)
    neg_sample = pool.apply_async(load_negative)
    uSimMat = pool.apply_async(load_drug_sim)
    iSimMat = pool.apply_async(load_disease_sim)
    DiDrAMat = pool.apply_async(load_didra)
    GT = pool.apply_async(load_gt)

    train = train.get()
    neg_sample = neg_sample.get()
    uSimMat = uSimMat.get()
    iSimMat = iSimMat.get()
    DiDrAMat = DiDrAMat.get()
    GT = GT.get()

    pool.close()
    pool.join()

    print()
    print()
    print(f'==================== Summary ====================')
    print()
    print(f'train: {len(train)}')
    print(f'negative: {len(neg_sample)}')
    print(f'DrugSim: {len(uSimMat)}')
    print(f'DiseaseSim: {len(iSimMat)}')
    print(f'DiDrA: {DiDrAMat.shape}')
    print(f'GT: {GT.shape}')
    print()
    print()
    print(f'==================== Model ====================')
    print()
    print(f'Building model')
    model, u_autoencoder, i_autoencoder = get_model(drug, disease, num_factors)
    print(f'Compiling model')
    compile_model(model, u_autoencoder, i_autoencoder, learner, learning_rate)
    print()
    print()
    print(f'==================== Train ====================')
    print()
    start_time = time()
    for epoch in range(epochs):
        epoch_time = time()
        print(f'========== Epoch {epoch} ==========')
        print()
        model_dataset, user_dataset, item_dataset = get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg_sample)
        print()
        fit_model_one_epoch(model, u_autoencoder, i_autoencoder, model_dataset, user_dataset, item_dataset, batch_size)
        print()
        print(f'TOTAL:{time() - epoch_time:.2f}s')
        print()
        print()
        gc.collect()
        tf.keras.backend.clear_session()

