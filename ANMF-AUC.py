import os
import numpy as np
from time import time
from matplotlib import pyplot as plt


verbose = 1000


# def load_dictionary():
#     st = time()
#     # print(f'Start loading drug_I2S.pickle and disease_I2S.pickle')
#     with open('inputs/drug_I2S.pickle', 'rb') as f:
#         drug_I2S = pickle.load(f)
#     with open('inputs/disease_I2S.pickle', 'rb') as f:
#         disease_I2S = pickle.load(f)
#     # print(f'End loading drug_I2S.pickle and disease_I2S.pickle | TOTAL:{time() - st:.2f}s')
#     return drug_I2S, disease_I2S


# drug_I2S, disease_I2S = load_dictionary()

for file in os.listdir(f'outputs'):
    if file == 'results.pickle':
        continue
    
    # print(f'Reading Data\\{file}\\test.rating')
    Pos, Neg = set(), set()
    with open(f'inputs\\{file}\\test.rating', 'r') as f:
        lines = f.readlines()
    # print()
    # print(f'Parsing Data\\{file}\\test.rating')
    start_time = time()
    for idx, line in enumerate(lines):
        # if verbose != 0 and idx % verbose == 0:
        #     progress(idx, len(lines), start_time, line.strip())
        drug, disease, score = line.strip().split('\t')
        if score == '1':
            Pos.add((int(drug), int(disease)))
        else:
            Neg.add((int(drug), int(disease)))
    # progressEnd(len(lines), start_time)
    # print()
    # print(f'Reading outputs\\{file}\\predict.txt')
    with open(f'outputs\\{file}\\predict.txt', 'r') as f:
        lines = f.readlines()
    # print()
    # print(f'Parsing outputs\\{file}\\predict.txt')
    predict = list()
    start_time = time()
    for idx, line in enumerate(lines):
        # if verbose != 0 and idx % verbose == 0:
        #     progress(idx, len(lines), start_time, line.strip())
        drug, disease, score = line.strip().split('\t')
        predict.append((int(drug), int(disease), float(score)))
    # progressEnd(len(lines), start_time)
    # print()
    # print(f'Sorting outputs\\{file}\\predict.txt')
    predict.sort(key=lambda x: x[2], reverse=True)

    TP, FP = 0, 0
    TP_sum = 0
    TPs = []
    FPs = []
    for drug, disease, score in predict:
        if (drug, disease) in Pos:
            TP += 1
        elif (drug, disease) in Neg:
            FP += 1
            TP_sum += TP
        else:
            print(f'ERROR:\t{drug}\t{disease}\t{score}')
        TPs.append(TP)
        FPs.append(FP)
    TPs = np.array(TPs) / (np.max(TPs) + 1e-10)
    FPs = np.array(FPs) / (np.max(FPs) + 1e-10)
    plt.plot(FPs, TPs)
    plt.show()
    # print()
    AUC = TP_sum / (TP * FP)
    print(f'{file}: {AUC}')
    # print()