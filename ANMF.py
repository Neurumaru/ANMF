# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:35:04 2018

@author: Administrator
"""
import gc
import os
import sys
import pickle
import tensorflow as tf
from multiprocessing.pool import Pool

from time import time
from DataLoader import load_matrix, load_rating_file_as_list, load_negative_file
from Model import get_model, compile_model, fit_model_one_epoch
from Dataset import get_dataset
from evaluate import predict_model
from progress import progress




def load_train(data_folder, reverse):
    result = load_rating_file_as_list(f'inputs/{data_folder}/train.rating', reverse)
    return result


def load_test(data_folder, reverse):
    result = load_rating_file_as_list(f'inputs/{data_folder}/test.rating', reverse)
    return result


def load_negative(data_folder, drug, reverse):
    result = load_negative_file(f'inputs/{data_folder}/negative.rating', drug, reverse)
    return result


def load_drug_sim(data_folder):
    result = load_matrix(f'inputs/{data_folder}/DrugSim.txt')
    return result


def load_disease_sim(data_folder):
    result = load_matrix(f'inputs/{data_folder}/DiseaseSim.txt')
    return result


def load_didra(data_folder):
    result = load_matrix(f'inputs/{data_folder}/DiDrA.txt')
    return result


def ANMF(
        data_folder, drug, disease, reverse=False,
        num_factors=256, epochs=50, num_negatives=10,
        noise=0.3, alpha=0.5, beta=0.5, ld=0.5, delta=0.5, phi=0.5, psi=0.5, 
        return_AUC=False, save_predict=True, 
        learner='adam', learning_rate=0.001, batch_size=1024, verbose=1
    ):
    if reverse:
        user, item = disease, drug
    else:
        user, item = drug, disease

    pool = Pool(6)

    train = pool.apply_async(load_train, args=[data_folder, reverse])
    test = pool.apply_async(load_test, args=[data_folder, reverse])
    neg_sample = pool.apply_async(load_negative, args=[data_folder, user, reverse])
    uSimMat = pool.apply_async(load_drug_sim, args=[data_folder])
    iSimMat = pool.apply_async(load_disease_sim, args=[data_folder])
    DiDrAMat = pool.apply_async(load_didra, args=[data_folder])

    train = train.get()
    test = test.get()
    neg_sample = neg_sample.get()
    uSimMat = uSimMat.get()
    iSimMat = iSimMat.get()
    DiDrAMat = DiDrAMat.get()

    pool.close()
    pool.join()
    
    if reverse:
        uSimMat, iSimMat, DiDrAMat = iSimMat, uSimMat, DiDrAMat.T

    model, prediction_model = get_model(user, item, num_factors, noise, ld, delta)
    compile_model(model, learner, learning_rate, alpha, beta, phi, psi)

    start_time = time()
    for epoch in range(epochs):
        progress(epoch, epochs, start_time)

        dataset = get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg_sample, batch_size)
        fit_model_one_epoch(model, dataset, batch_size, verbose=verbose)

        gc.collect()
        tf.keras.backend.clear_session()

    predict = predict_model(prediction_model, test, uSimMat, iSimMat, DiDrAMat)
    if save_predict:
        os.makedirs(f'outputs/{data_folder}', exist_ok=True)
        with open(f'outputs/{data_folder}/predict.txt', 'w') as f:
            for u, i, pred in predict:
                if reverse:
                    f.write(f'{i}\t{u}\t{pred}\n')
                else:
                    f.write(f'{u}\t{i}\t{pred}\n')

    if return_AUC:
        Pos, Neg = set(), set()

        for user, item, score in test:
            if score == 1:
                Pos.add((user, item))
            else:
                Neg.add((user, item))

        predict.sort(key=lambda x: -x[2])

        TP, FP = 0, 0
        TP_sum = 0
        for user, item, score in predict:
            if (user, item) in Pos:
                TP += 1
            elif (user, item) in Neg:
                FP += 1
                TP_sum += TP
        AUC = TP_sum / (TP * FP)

        return AUC
    else:
        return


def grid_search():
    epoch = [30]
    noise = [0.1, 0.2]
    alpha_beta = [0.8]
    ld_delta = [1e-04]
    phi_psi = [0.5, 1, 2]
    num_factors = [512]
    num_negatives = [10]

    max_AUC, max_epoch, max_noise, max_alpha_beta, max_ld_delta, max_phi_psi, max_num_factors, max_num_negatives = 0, 0, 0, 0, 0, 0, 0, 0

    try:
        with open('outputs/results.pickle', 'rb') as f:
            results = pickle.load(f)
    except:
        results = dict()

    print(f'=========================================================================================================')
    print(f'{"AUC":9s}{"EPOCH":9s}{"NOISE":9s}{"ALPAH":9s}{"BETA":9s}{"LAMDA":9s}{"DELTA":9s}{"PHI":9s}{"PSI":9s}{"FACTORS":12s}{"NAGATIVES":12s}')
    for result in results:
        AUC = results[result]
        (e, n, ab, ld, pp, nf, nn) = result
        print(f'{AUC:<9.4f}{e:<9d}{n:<9.2f}{ab:<9.2f}{ab:<9.2f}{ld:<9.1e}{ld:<9.1e}{pp:<9.2f}{pp:<9.2f}{nf:<12d}{nn:<12d}')
        if AUC > max_AUC:
            max_AUC = AUC
            max_epoch = e
            max_noise = n
            max_alpha_beta = ab
            max_ld_delta = ld
            max_phi_psi = pp
            max_num_factors = nf
            max_num_negatives = nn
    print(f'=========================================================================================================')
    print(f'{max_AUC:<9.4f}{max_epoch:<9d}{max_noise:<9.2f}{max_alpha_beta:<9.2f}{max_alpha_beta:<9.2f}{max_ld_delta:<9.1e}{max_ld_delta:<9.1e}{max_phi_psi:<9.2f}{max_phi_psi:<9.2f}{max_num_factors:<12d}{max_num_negatives:<12d}')
    print(f'=========================================================================================================')
    print(f'{"AUC":9s}{"EPOCH":9s}{"NOISE":9s}{"ALPAH":9s}{"BETA":9s}{"LAMDA":9s}{"DELTA":9s}{"PHI":9s}{"PSI":9s}{"FACTORS":12s}{"NAGATIVES":12s}')
    for e in epoch:
        for n in noise:
            for ab in alpha_beta:
                for ld in ld_delta:
                    for pp in phi_psi:
                        for nf in num_factors:
                            for nn in num_negatives:
                                if (e, n, ab, ld, pp, nf, nn) in results:
                                    continue

                                AUC = ANMF(
                                    f'Drug0', drug=3245, disease=6322, epochs=e,
                                    num_factors=nf, noise=n, num_negatives=nn,
                                    alpha=ab, beta=ab, ld=ld, delta=ld, phi=pp, psi=pp, 
                                    return_AUC=True, save_predict=True
                                )
                                results[(e, n, ab, ld, pp, nf, nn)] = AUC
                                if AUC > max_AUC:
                                    max_AUC = AUC
                                    max_epoch = e
                                    max_noise = n
                                    max_alpha_beta = ab
                                    max_ld_delta = ld
                                    max_phi_psi = pp
                                    max_num_factors = nf
                                
                                with open('outputs/results.pickle', 'wb') as f:
                                    pickle.dump(results, f)
                                                        
                                print(f'\r{AUC:<9.4f}{e:<9d}{n:<9.2f}{ab:<9.2f}{ab:<9.2f}{ld:<9.1e}{ld:<9.1e}{pp:<9.2f}{pp:<9.2f}{nf:<12d}{nn:<12d}')
    print(f'=========================================================================================================')
    print(f'{max_AUC:<9.4f}{max_epoch:<9d}{max_noise:<9.2f}{max_alpha_beta:<9.2f}{max_alpha_beta:<9.2f}{max_ld_delta:<9.1e}{max_ld_delta:<9.1e}{max_phi_psi:<9.2f}{max_phi_psi:<9.2f}{max_num_factors:<12d}{max_num_negatives:<12d}')
    print(f'=========================================================================================================')

if __name__ == '__main__':
    # grid_search()

    file_path = sys.argv[1]
    
    if len(sys.argv) != 3:
        print('ANMF')
        print()
        print('usage: python ANMF.py [Disease | Drug] [0-9]')
        exit()

    fold = sys.argv[1]
    i = sys.argv[2]

    AUC = ANMF(
        f'{fold}{i}', drug=3245, disease=6322, epochs=50,
        num_factors=512, noise=0.1, num_negatives=10,
        alpha=0.8, beta=0.8, ld=1e-4, delta=1e-4, phi=1, psi=1, 
        return_AUC=True, save_predict=True, verbose=0, reverse=True
    )
    print(f'\{fold}{i}: {AUC}')
