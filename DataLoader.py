import numpy as np


def load_matrix(filename):
    DiDrMat = np.loadtxt(filename, dtype=np.float32)
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

def load_negative_file(filename, user, item, reverse):
    negativeList = [[] for _ in range(user)]
    with open(filename) as f:
        line = f.readline()
        while line is not None and line != "":
            line = line.strip('\n')
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            if reverse:
                negativeList[i].append(u)
            else:
                negativeList[u].append(i)
            line = f.readline()
    for u in range(user):
        if item < 32768:
            negativeList[u] = np.array(negativeList[u], dtype=np.int16)
        elif item < 1073741824:
            negativeList[u] = np.array(negativeList[u], dtype=np.int32)
        else:
            negativeList[u] = np.array(negativeList[u], dtype=np.int64)
    return negativeList
