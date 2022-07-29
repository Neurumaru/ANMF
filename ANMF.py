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
from Dataset import get_dataset
from evaluate import predict_model
from progress import progress, progressEnd
from evaluate_original import evaluate_model

learner = 'adam'
learning_rate = 0.001
batch_size = 64
verbose = 1


def load_train(data_folder):
    st = time()
    # print(f'Start loading train.rating')
    result = load_rating_file_as_list(f'inputs\\{data_folder}\\train.rating')
    # print(f'End loading train.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_test(data_folder):
    st = time()
    # print(f'Start loading test.rating')
    result = load_rating_file_as_list(f'inputs\\{data_folder}\\test.rating')
    # print(f'End loading test.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_negative(data_folder, drug):
    st = time()
    # print(f'Start loading negative.rating')
    result = load_negative_file(f'inputs\\{data_folder}\\negative.rating', drug)
    # print(f'End loading negative.rating | TOTAL:{time() - st:.2f}s')
    return result


def load_drug_sim(data_folder):
    st = time()
    # print(f'Start loading DrugSim.txt')
    result = load_user_sim_file(f'inputs\\{data_folder}\\DrugSim.txt')
    # print(f'End loading DrugSim.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_disease_sim(data_folder):
    st = time()
    # print(f'Start loading DiseaseSim.txt')
    result = load_item_sim_file(f'inputs\\{data_folder}\\DiseaseSim.txt')
    # print(f'End loading DiseaseSim.txt | TOTAL:{time() - st:.2f}s')
    return result


def load_didra(data_folder):
    st = time()
    # print(f'Start loading DiDrA.txt')
    result = load_c_matrix(f'inputs\\{data_folder}\\DiDrA.txt')
    # print(f'End loading DiDrA.txt | TOTAL:{time() - st:.2f}s')
    return result


# def load_dictionary():
#     st = time()
#     # print(f'Start loading drug_I2S.pickle and disease_I2S.pickle')
#     with open('inputs\\drug_I2S.pickle', 'rb') as f:
#         drug_I2S = pickle.load(f)
#     with open('inputs\\disease_I2S.pickle', 'rb') as f:
#         disease_I2S = pickle.load(f)
#     # print(f'End loading drug_I2S.pickle and disease_I2S.pickle | TOTAL:{time() - st:.2f}s')
#     return drug_I2S, disease_I2S


def ANMF(
    data_folder, drug, disease, 
    num_factors=256, epochs=50, num_negatives=10,
    noise=0.3, alpha=0.5, beta=0.5, ld=0.5, delta=0.5, phi=0.5, psi=0.5, 
    original_dataset=False, original_evaluate=False, return_AUC=False, save_predict=True
):
    # print(f'==================== Dataset ({data_folder}) ====================')
    # print()

    pool = Pool(6)
    train = pool.apply_async(load_train, args=[data_folder])
    test = pool.apply_async(load_test, args=[data_folder])
    neg_sample = pool.apply_async(load_negative, args=[data_folder, drug])
    uSimMat = pool.apply_async(load_drug_sim, args=[data_folder])
    iSimMat = pool.apply_async(load_disease_sim, args=[data_folder])
    DiDrAMat = pool.apply_async(load_didra, args=[data_folder])
    # I2S = pool.apply_async(load_dictionary)

    train = train.get()
    test = test.get()
    neg_sample = neg_sample.get()
    uSimMat = uSimMat.get()
    iSimMat = iSimMat.get()
    DiDrAMat = DiDrAMat.get()
    # drug_I2S, disease_I2S = I2S.get()

    pool.close()
    pool.join()

    if original_dataset:
        from Dataset_original import Dataset
        dataset = Dataset()
        train, test, uSimMat, iSimMat, DiDrAMat, neg_sample \
            = dataset.trainMatrix, dataset.testRatings, dataset.uSimMat, dataset.iSimMat, dataset.DiDrAMat, dataset.Sim_order

    # print()
    # print(f'==================== Summary ({data_folder}) ====================')
    # print()
    # print(f'train: {len(train)}')
    # print(f'test: {len(test)}')
    # print(f'negative: {len(neg_sample)}')
    # print(f'DrugSim: {len(uSimMat)}')
    # print(f'DiseaseSim: {len(iSimMat)}')
    # print(f'DiDrA: {DiDrAMat.shape}')
    # print()
    # print(f'==================== Model ({data_folder}) ====================')
    # print()
    # print(f'Building model')
    model, prediction_model = get_model(drug, disease, num_factors, noise, ld, delta)
    # print(f'Compiling model')
    compile_model(model, learner, learning_rate, alpha, beta, phi, psi)
    # print()
    # print(f'==================== Train ({data_folder}) ====================')
    start_time = time()
    # print()
    for epoch in range(epochs):
        # epoch_time = time()
        # print(f'========== ANMF : Epoch {epoch} ({data_folder}) ==========')
        dataset = get_dataset(train, num_negatives, uSimMat, iSimMat, DiDrAMat, neg_sample)
        # print()
        fit_model_one_epoch(model, dataset, batch_size)
        if original_evaluate:
            hit, auc, _, _, _, area_pr = evaluate_model(prediction_model, test, uSimMat, iSimMat, DiDrAMat, 10, 1, train)
            # print()
            # print('area_pr: ' + str(area_pr))
            # print('auc: ' + str(auc))
            # print('hit: ' + str(hit))
            # print()
        # print(f'EPOCH:{time() - epoch_time:.2f}s')
        # print()
        gc.collect()
        tf.keras.backend.clear_session()
    # print(f'TOTAL:{time() - start_time:.2f}s')
    # print()
    # print(f'==================== Evaluate ({data_folder}) ====================')
    predict = predict_model(prediction_model, test, uSimMat, iSimMat, DiDrAMat)
    # print()
    if save_predict:
        # start_time = time()
        # print(f'Saving to predict.txt')
        os.makedirs(f'outputs\\{data_folder}', exist_ok=True)
        with open(f'outputs\\{data_folder}\\predict.txt', 'w') as f:
            for idx, (u, i, pred) in enumerate(predict):
                # if verbose != 0 and idx % verbose == 0:
                #     progress(idx, len(predict), start_time)
                f.write(f'{u}\t{i}\t{pred}\n')
            # progressEnd(len(predict), start_time)

    if return_AUC:
        if original_evaluate:
            return auc
        else:
            Pos, Neg = set(), set()

            for drug, disease, score in test:
                if score == 1:
                    Pos.add((drug, disease))
                else:
                    Neg.add((drug, disease))

            predict.sort(key=lambda x: -x[2])

            TP, FP = 0, 0
            TP_sum = 0
            for drug, disease, score in predict:
                if (drug, disease) in Pos:
                    TP += 1
                elif (drug, disease) in Neg:
                    FP += 1
                    TP_sum += TP
            AUC = TP_sum / (TP * FP)

            return AUC
    else:
        return


if __name__ == '__main__':
    # ANMF(f'master', drug=593, disease=313, num_factors=256, epochs=50, original_evaluate=True)
    # ANMF(f'master_original', drug=593, disease=313, num_factors=256, epochs=50, original_dataset=True, original_evaluate=True)
    # for i in range(10):
    #     ANMF(f'Disease{i}', drug=11219, disease=6322, num_factors=512, epochs=50)
    # for i in range(10):
    #     ANMF(f'Drug{i}', drug=11219, disease=6322, num_factors=512, epochs=50)

    epoch = [50]
    noise = [0.1, 0.2]
    alpha_beta = [0.2, 0.5, 0.8]
    ld_delta = [1e-03, 1e-04]
    phi_psi = [1]
    num_factors = [256, 512]
    num_negatives = [10]

    max_AUC, max_epoch, max_noise, max_alpha_beta, max_ld_delta, max_phi_psi, max_num_factors, max_num_negatives = 0, 0, 0, 0, 0, 0, 0, 0

    try:
        with open('outputs\\results.pickle', 'rb') as f:
            results = pickle.load(f)
    except:
        results = dict()

    for result in results:
        if results[result] > max_AUC:
            (e, n, ab, ld, pp, nf, nn) = result
            max_AUC = results[result]
            max_epoch = e
            max_noise = n
            max_alpha_beta = ab
            max_ld_delta = ld
            max_phi_psi = pp
            max_num_factors = nf
            max_num_negatives = nn

    for e in epoch:
        for n in noise:
            for ab in alpha_beta:
                for ld in ld_delta:
                    for pp in phi_psi:
                        for nf in num_factors:
                            for nn in num_negatives:
                                start_time = time()
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
                                
                                with open('outputs\\results.pickle', 'wb') as f:
                                    pickle.dump(results, f)
                                                        
                                print(f'========================================')
                                print(f'AUC: {AUC}')
                                print(f'EPOCH: {e}')
                                print(f'NOISE: {n}')
                                print(f'ALPHA: {ab}')
                                print(f'BETA: {ab}')
                                print(f'LAMBDA: {ld}')
                                print(f'DELTA: {ld}')
                                print(f'PHI: {pp}')
                                print(f'PSI: {pp}')
                                print(f'NUM_FACTORS: {nf}')
                                print(f'NUM_NAGATIVES: {nn}')
                                print(f'TOTAL:{time() - start_time:.2f}s')
                                print(f'========================================')

    print(f'========================================')
    print(f'BEST AUC: {max_AUC}')
    print(f'BEST EPOCH: {max_epoch}')
    print(f'BEST NOISE: {max_noise}')
    print(f'BEST ALPHA: {max_alpha_beta}')
    print(f'BEST BETA: {max_alpha_beta}')
    print(f'BEST LAMBDA: {max_ld_delta}')
    print(f'BEST DELTA: {max_ld_delta}')
    print(f'BEST PHI: {max_phi_psi}')
    print(f'BEST PSI: {max_phi_psi}')
    print(f'BEST NUM_FACTORS: {max_num_factors}')
    print(f'BEST NUM_NAGATIVES: {max_num_negatives}')
    print(f'========================================')
