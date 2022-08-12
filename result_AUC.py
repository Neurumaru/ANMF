import os
from evaluate import calculate_AUC

for file in os.listdir(f'outputs'):
    if file == 'results.pickle':
        continue
    if file == 'ANMF_disease.txt':
        continue
    if file == 'ANMF_drug.txt':
        continue

    y_true = []
    with open(f'inputs\\{file}\\test.rating', 'r') as f:
        for line in f:
            drug, disease, score = line.strip().split('\t')
            y_true.append((int(drug), int(disease), float(score)))

    predict = []
    with open(f'outputs\\{file}\\predict.txt', 'r') as f:
        for  line in f:
            drug, disease, score = line.strip().split('\t')
            predict.append((int(drug), int(disease), float(score)))

    AUC = calculate_AUC(y_true, predict)
    print(f'{file}: {AUC}')