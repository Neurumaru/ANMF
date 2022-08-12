import pickle


def load_dictionary():
    with open('inputs/drug_I2S.pickle', 'rb') as f:
        drug_I2S = pickle.load(f)
    with open('inputs/disease_I2S.pickle', 'rb') as f:
        disease_I2S = pickle.load(f)
    return drug_I2S, disease_I2S

def integrate(cv):
    drug_I2S, disease_I2S = load_dictionary()

    predicts = []
    for i in range(10):
        file = f'{cv}{i}'
        with open(f'outputs\\{file}\\predict.txt', 'r') as f:
            for line in f:
                drug, disease, score = line.strip().split('\t')
                predicts.append((int(drug), int(disease), float(score)))
    predicts.sort(key=lambda x:x[2], reverse=True)

    with open(f'outputs/ANMF_{cv}.txt', 'w') as f:
        for drug, disease, score in predicts:
            f.write(f'{drug_I2S[drug]}\t{disease_I2S[disease]}\t{score}')

integrate('disease')
integrate('drug')