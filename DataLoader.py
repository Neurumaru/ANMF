import numpy as np


def load_matrix(filename):
    DiDrMat = np.loadtxt(filename)
    return DiDrMat

def load_rating_file_as_list(filename, reverse):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            line = line.strip('\n')
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
            if reverse:
                ratingList.append([item, user, rating])
            else:
                ratingList.append([user, item, rating])
            line = f.readline()
    return ratingList

def load_sim_file(filename):
    SimMat = []
    with open(filename, "r") as f:
        for line in f.readlines():
            temp = []
            line = line.strip('\n')
            arr = line.split()
            for item in arr:
                temp.append(float(item))
            SimMat.append(temp)
    return SimMat

def load_negative_file(filename, drug, reverse):
    negativeList = [[] for _ in range(drug)]
    with open(filename) as f:
        line = f.readline()
        while line is not None and line != "":
            line = line.strip('\n')
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            if reverse:
                negativeList[item].append(user)
            else:
                negativeList[user].append(item)
            line = f.readline()
    return negativeList
