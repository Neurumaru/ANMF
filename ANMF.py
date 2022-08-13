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
from Model import get_model, compile_model, fit_model_one_epoch, load_weights, save_weights
from Dataset import get_dataset
from evaluate import predict_model, calculate_AUC
from progress import progress


def load_train(data_folder, reverse):
    result = load_rating_file_as_list(f'inputs/{data_folder}/train.rating', reverse)
    return result


def load_test(data_folder, reverse):
    result = load_rating_file_as_list(f'inputs/{data_folder}/test.rating', reverse)
    return result


def load_negative(data_folder, user, item, reverse):
    result = load_negative_file(f'inputs/{data_folder}/negative.rating', user, item, reverse)
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


def grid_search(datafolder, drug, disease, epoch, noise, alpha_beta, ld_delta, phi_psi, num_factors, num_negatives, reverse):
    max_AUC, max_epoch, max_noise, max_alpha_beta, max_ld_delta, max_phi_psi, max_num_factors, max_num_negatives = 0, 0, 0, 0, 0, 0, 0, 0

    try:
        with open('outputs/results.pickle', 'rb') as f:
            results = pickle.load(f)
    except:
        results = dict()

    print('=' * 105)
    print(
        f'{"AUC":9s}'
        f'{"EPOCH":9s}'
        f'{"NOISE":9s}'
        f'{"ALPAH":9s}'
        f'{"BETA":9s}'
        f'{"LAMDA":9s}'
        f'{"DELTA":9s}'
        f'{"PHI":9s}'
        f'{"PSI":9s}'
        f'{"FACTORS":12s}'
        f'{"NAGATIVES":12s}')
    for result in results:
        AUC = results[result]
        (e, n, ab, ld, pp, nf, nn) = result
        print(
            f'{AUC:<9.4f}'
            f'{e:<9d}'
            f'{n:<9.2f}'
            f'{ab:<9.2f}'
            f'{ab:<9.2f}'
            f'{ld:<9.1e}'
            f'{ld:<9.1e}'
            f'{pp:<9.2f}'
            f'{pp:<9.2f}'
            f'{nf:<12d}'
            f'{nn:<12d}')
        if AUC > max_AUC:
            max_AUC = AUC
            max_epoch = e
            max_noise = n
            max_alpha_beta = ab
            max_ld_delta = ld
            max_phi_psi = pp
            max_num_factors = nf
            max_num_negatives = nn
    print('=' * 105)
    print(
        f'{max_AUC:<9.4f}'
        f'{max_epoch:<9d}'
        f'{max_noise:<9.2f}'
        f'{max_alpha_beta:<9.2f}'
        f'{max_alpha_beta:<9.2f}'
        f'{max_ld_delta:<9.1e}'
        f'{max_ld_delta:<9.1e}'
        f'{max_phi_psi:<9.2f}'
        f'{max_phi_psi:<9.2f}'
        f'{max_num_factors:<12d}'
        f'{max_num_negatives:<12d}')
    print('=' * 105)
    print(
        f'{"AUC":9s}'
        f'{"EPOCH":9s}'
        f'{"NOISE":9s}'
        f'{"ALPAH":9s}'
        f'{"BETA":9s}'
        f'{"LAMDA":9s}'
        f'{"DELTA":9s}'
        f'{"PHI":9s}'
        f'{"PSI":9s}'
        f'{"FACTORS":12s}'
        f'{"NAGATIVES":12s}')
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
                                    f'{datafolder}', drug=drug, disease=disease, epochs=e,
                                    num_factors=nf, noise=n, num_negatives=nn,
                                    alpha=ab, beta=ab, ld=ld, delta=ld, phi=pp, psi=pp, 
                                    verbose=0, return_AUC=True, save_predict=True, reverse=reverse
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
                                                        
                                print(
                                    f'\r{AUC:<9.4f}'
                                    f'{e:<9d}'
                                    f'{n:<9.2f}'
                                    f'{ab:<9.2f}'
                                    f'{ab:<9.2f}'
                                    f'{ld:<9.1e}'
                                    f'{ld:<9.1e}'
                                    f'{pp:<9.2f}'
                                    f'{pp:<9.2f}'
                                    f'{nf:<12d}'
                                    f'{nn:<12d}')
    print('=' * 105)
    print(
        f'{max_AUC:<9.4f}'
        f'{max_epoch:<9d}'
        f'{max_noise:<9.2f}'
        f'{max_alpha_beta:<9.2f}'
        f'{max_alpha_beta:<9.2f}'
        f'{max_ld_delta:<9.1e}'
        f'{max_ld_delta:<9.1e}'
        f'{max_phi_psi:<9.2f}'
        f'{max_phi_psi:<9.2f}'
        f'{max_num_factors:<12d}'
        f'{max_num_negatives:<12d}')
    print('=' * 105)


def ANMF(
        data_folder, drug, disease, reverse=False,
        num_factors=256, epochs=50, num_negatives=10,
        noise=0.3, alpha=0.5, beta=0.5, ld=0.5, delta=0.5, phi=0.5, psi=0.5, 
        return_AUC=False, save_predict=True, 
        learner='adam', learning_rate=0.001, batch_size=1024, verbose=1, checkpoint=False
    ):
    if reverse:
        user, item = disease, drug
    else:
        user, item = drug, disease

    pool = Pool(6)

    train = pool.apply_async(load_train, args=[data_folder, reverse])
    test = pool.apply_async(load_test, args=[data_folder, reverse])
    neg_sample = pool.apply_async(load_negative, args=[data_folder, user, item, reverse])
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

    meta = None
    if checkpoint:
        model, prediction_model, meta = load_weights(model, prediction_model, meta, f'checkpoints/{data_folder}')

    if meta == None:
        start_epoch = 0
        start_time = time()
        meta = dict()
    else:
        start_epoch = meta['epoch']
        start_time = time() - meta['time']

    for epoch in range(start_epoch, epochs):
        progress(epoch, epochs, start_time)

        dataset = get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg_sample, batch_size)
        fit_model_one_epoch(model, dataset, batch_size, verbose=verbose)

        if checkpoint:
            meta['epoch'] = epoch+1
            meta['time'] = time() - start_time
            save_weights(model, prediction_model, meta, f'checkpoints/{data_folder}')

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
        return calculate_AUC(test, predict)
    else:
        return
    

def usage():
    print('ANMF')
    print()
    print(
        'Usage: python ANMF.py [type]\n'
        '         [dataset] [datafolder] [reverse] [epochs] [noise]\n'
        '         [alpha & beta] [lamda & delta] [phi & psi]\n'
        '         [num_factors] [num_negatives]')
    print()
    print(' - type : evaluete | grid_search')
    print(' - dataset : atc-code | chemical | original')
    print(' - datafolder : Disease | Drug')
    print(' - reverse : 0 | 1')
    print()
    print(
        '  ex) python ANMF.py grid_search\n'
        '        atc-code Drug0 30,50 0.1\n'
        '        0.8 1e-04 1\n'
        '        256,512 10,20')
    exit()


def get_argv(index):
    if len(sys.argv) < index + 1:
        usage()
    return sys.argv[index]


if __name__ == '__main__':
    exetype = get_argv(1)
    dataset = get_argv(2)
    datafolder = get_argv(3)
    reverse = get_argv(4)

    if dataset == 'atc-code':
        drug = 3245
        disease = 6322
    elif dataset == 'chemical':
        drug = 11219
        disease = 6322
    elif dataset == 'original':
        drug = 593
        disease = 313
    else:
        usage()

    if reverse == '0':
        reverse = False
    elif reverse == '1':
        reverse = True
    else:
        usage()

    if exetype == 'evaluate':
        epoch = int (get_argv(5))
        noise = float(get_argv(6))
        alpha_beta = float(get_argv(7))
        ld_delta = float(get_argv(8))
        phi_psi = float(get_argv(9))
        num_factors = int(get_argv(10))
        num_negatives = int(get_argv(11))

        AUC = ANMF(
            f'{datafolder}', drug=drug, disease=disease, epochs=epoch,
            num_factors=num_factors, noise=noise, num_negatives=num_negatives,
            alpha=alpha_beta, beta=alpha_beta, ld=ld_delta, delta=ld_delta, phi=phi_psi, psi=phi_psi, 
            return_AUC=True, save_predict=True, verbose=0, reverse=reverse, checkpoint=True
        )
        print(f'\r{datafolder}: {AUC}')
    elif exetype == 'grid_search':
        epoch = list(map(int, get_argv(5).split(',')))
        noise = list(map(float, get_argv(6).split(',')))
        alpha_beta = list(map(float, get_argv(7).split(',')))
        ld_delta = list(map(float, get_argv(8).split(',')))
        phi_psi = list(map(float, get_argv(9).split(',')))
        num_factors = list(map(int, get_argv(10).split(',')))
        num_negatives = list(map(int, get_argv(11).split(',')))

        grid_search(datafolder, drug, disease, epoch, noise, alpha_beta, ld_delta, phi_psi, num_factors, num_negatives, reverse)
    else:
        usage()
        
