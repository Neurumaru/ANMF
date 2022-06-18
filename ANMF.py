# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:35:04 2018

@author: Administrator
"""
import gc
import os
import pickle
import tensorflow as tf
from multiprocessing.pool import Pool
from time import time
from DataLoader import load_c_matrix, load_matrix, load_rating_file_as_list, load_item_sim_file, load_user_sim_file, \
    load_negative_file
from Model import get_model, compile_model, fit_model_one_epoch
from Dataset import get_dataset
from evaluate import predict_model
from progress import progress, progressEnd

num_factors = 1024
num_negatives = 10
learner = 'adam'
learning_rate = 0.001
epochs = 50
batch_size = 64
verbose = 1

drug = 11219
disease = 6322


def load_train(data_folder):
    st = time()
    print(f'Start loading train.rating')
    result = load_rating_file_as_list(f'Data\\{data_folder}\\train.rating')
    print(f'End loading train.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_test(data_folder):
    st = time()
    print(f'Start loading test.rating')
    result = load_rating_file_as_list(f'Data\\{data_folder}\\test.rating')
    print(f'End loading test.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_negative(data_folder):
    st = time()
    print(f'Start loading negative.rating')
    result = load_negative_file(f'Data\\{data_folder}\\negative.rating', drug)
    print(f'End loading negative.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_drug_sim(data_folder):
    st = time()
    print(f'Start loading DrugSim.txt')
    result = load_user_sim_file(f'Data\\{data_folder}\\DrugSim.txt')
    print(f'End loading DrugSim.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_disease_sim(data_folder):
    st = time()
    print(f'Start loading DiseaseSim.txt')
    result = load_item_sim_file(f'Data\\{data_folder}\\DiseaseSim.txt')
    print(f'End loading DiseaseSim.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_didra(data_folder):
    st = time()
    print(f'Start loading DiDrA.txt')
    result = load_c_matrix(f'Data\\{data_folder}\\DiDrA.txt')
    print(f'End loading DiDrA.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_dictionary():
    st = time()
    print(f'Start loading drug_I2S.pickle and disease_I2S.pickle')
    with open('Data/drug_I2S.pickle', 'rb') as f:
        drug_I2S = pickle.load(f)
    with open('Data/disease_I2S.pickle', 'rb') as f:
        disease_I2S = pickle.load(f)
    print(f'End loading drug_I2S.pickle and disease_I2S.pickle | TOTAL:{time() - st:.2f}s')
    return drug_I2S, disease_I2S


def main(data_folder):
    print(f'==================== Dataset {data_folder} ====================')
    print()

    pool = Pool(7)
    train = pool.apply_async(load_train, args=[data_folder])
    test = pool.apply_async(load_test, args=[data_folder])
    neg_sample = pool.apply_async(load_negative, args=[data_folder])
    uSimMat = pool.apply_async(load_drug_sim, args=[data_folder])
    iSimMat = pool.apply_async(load_disease_sim, args=[data_folder])
    DiDrAMat = pool.apply_async(load_didra, args=[data_folder])
    I2S = pool.apply_async(load_dictionary)

    train = train.get()
    test = test.get()
    neg_sample = neg_sample.get()
    uSimMat = uSimMat.get()
    iSimMat = iSimMat.get()
    DiDrAMat = DiDrAMat.get()
    drug_I2S, disease_I2S = I2S.get()

    pool.close()
    pool.join()

    print()
    print()
    print(f'==================== Summary {data_folder} ====================')
    print()
    print(f'train: {len(train)}')
    print(f'test: {len(test)}')
    print(f'negative: {len(neg_sample)}')
    print(f'DrugSim: {len(uSimMat)}')
    print(f'DiseaseSim: {len(iSimMat)}')
    print(f'DiDrA: {DiDrAMat.shape}')
    print()
    print()
    print(f'==================== Model {data_folder} ====================')
    print()
    print(f'Building model')
    model, u_autoencoder, i_autoencoder = get_model(drug, disease, num_factors)
    print(f'Compiling model')
    compile_model(model, u_autoencoder, i_autoencoder, learner, learning_rate)
    print()
    print()
    print(f'==================== Train {data_folder} ====================')
    print()
    start_time = time()
    for epoch in range(epochs):
        epoch_time = time()
        print(f'========== Epoch {epoch} {data_folder} ==========')
        print()
        model_dataset, user_dataset, item_dataset = get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg_sample)
        print()
        fit_model_one_epoch(model, u_autoencoder, i_autoencoder, model_dataset, user_dataset, item_dataset, batch_size)
        print()
        print(f'EPOCH:{time() - epoch_time:.2f}s')
        print()
        print()
        gc.collect()
        tf.keras.backend.clear_session()
    print(f'TOTAL:{time() - start_time:.2f}s')
    print()
    print()
    print(f'==================== Evaluate {data_folder} ====================')
    predict = predict_model(model, test, uSimMat, iSimMat, DiDrAMat)
    print()
    start_time = time()
    print(f'Saving to predict.txt')
    os.makedirs(f'outputs\\{data_folder}', exist_ok=True)
    with open(f'outputs\\{data_folder}\\predict.txt', 'w') as f:
        for idx, (u, i, pred) in enumerate(predict):
            if verbose != 0 and idx % verbose == 0:
                progress(idx, len(predict), start_time, f'{drug_I2S[u]}\t{disease_I2S[i]}\t{pred:.4f}')
            f.write(f'{drug_I2S[u]}\t{disease_I2S[i]}\t{pred}\n')
        progressEnd(len(predict), start_time)


if __name__ == '__main__':
    #for i in range(9, 10):
    #    main(f'Disease{i}')
    for i in range(3, 10):
        main(f'Drug{i}')

