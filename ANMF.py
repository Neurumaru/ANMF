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
from DataLoader import load_c_matrix, load_rating_file_as_list, load_item_sim_file, load_user_sim_file, load_negative_file
from Model import get_model, compile_model, fit_model_one_epoch
from Dataset import get_dataset, get_model_dataset
from evaluate import predict_model
from progress import progress, progressEnd

num_negatives = 10
learner = 'adam'
learning_rate = 0.001
batch_size = 64
verbose = 1


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


def load_negative(data_folder, drug):
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


def ANMF(data_folder, drug, disease, num_factors, epochs = 50, original_dataset=False, original_evaluate=False):
    print(f'==================== Dataset ({data_folder}) ====================')
    print()

    pool = Pool(7)
    train = pool.apply_async(load_train, args=[data_folder])
    test = pool.apply_async(load_test, args=[data_folder])
    neg_sample = pool.apply_async(load_negative, args=[data_folder, drug])
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

    if original_dataset:
        from Dataset_original import Dataset
        dataset = Dataset()
        train, test, uSimMat, iSimMat, DiDrAMat, neg_sample \
            = dataset.trainMatrix, dataset.testRatings, dataset.uSimMat, dataset.iSimMat, dataset.DiDrAMat, dataset.Sim_order

    print()
    print(f'==================== Summary ({data_folder}) ====================')
    print()
    print(f'train: {len(train)}')
    print(f'test: {len(test)}')
    print(f'negative: {len(neg_sample)}')
    print(f'DrugSim: {len(uSimMat)}')
    print(f'DiseaseSim: {len(iSimMat)}')
    print(f'DiDrA: {DiDrAMat.shape}')
    print()
    print(f'==================== Model ({data_folder}) ====================')
    print()
    print(f'Building model')
    model, u_autoencoder, i_autoencoder = get_model(drug, disease, num_factors)
    print(f'Compiling model')
    compile_model(model, u_autoencoder, i_autoencoder, learner, learning_rate)
    print()
    print(f'==================== Train ({data_folder}) ====================')
    start_time = time()
    print()
    for epoch in range(epochs):
        print(f'========== Auto-Encoders : Epoch {epoch} ({data_folder}) ==========')
        user_dataset, item_dataset = get_dataset(train, uSimMat, iSimMat, DiDrAMat)
        print()
        fit_model_one_epoch(u_autoencoder, user_dataset, batch_size)
        fit_model_one_epoch(i_autoencoder, item_dataset, batch_size)
        print()
        gc.collect()
        tf.keras.backend.clear_session()
    for epoch in range(epochs):
        epoch_time = time()
        print(f'========== ANMF : Epoch {epoch} ({data_folder}) ==========')
        model_dataset = get_model_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg_sample)
        print()
        fit_model_one_epoch(model, model_dataset, batch_size)
        if original_evaluate:
            from evaluate_original import evaluate_model
            hit, auc, fpr, tpr, _, area_pr = evaluate_model(model, test, uSimMat, iSimMat, DiDrAMat, 10, 1, train)
            print()
            print('area_pr: ' + str(area_pr))
            print('auc: ' + str(auc))
            print('hit: ' + str(hit))
            print()
        print(f'EPOCH:{time() - epoch_time:.2f}s')
        print()
        gc.collect()
        tf.keras.backend.clear_session()
    print(f'TOTAL:{time() - start_time:.2f}s')
    print()
    print(f'==================== Evaluate ({data_folder}) ====================')
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
    # ANMF(f'master', drug=593, disease=313, num_factors=256, original_evaluate=True)
    # ANMF(f'master_original', drug=593, disease=313, num_factors=256, original_dataset=True, original_evaluate=True)
    for i in range(10):
        ANMF(f'Disease{i}', drug=11219, disease=6322, num_factors=512, epochs=50)
    # for i in range(10):
    #     ANMF(f'Drug{i}', drug=11219, disease=6322, num_factors=512, epochs=50)
