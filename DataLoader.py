'''
Created on Aug 8, 2016
Processing datasets.

@author: XinXing Yang
'''
import numpy as np


def load_c_matrix(filename):
    DiDrMat = np.loadtxt(filename)
    DiDrMat = DiDrMat + np.random.normal(0, 1.0, DiDrMat.shape)
    return DiDrMat


def load_matrix(filename):
    DiDrMat = np.loadtxt(filename)
    return DiDrMat


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            line = line.strip('\n')
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
            ratingList.append([user, item, rating])
            line = f.readline()
    return ratingList


def load_user_sim_file(filename):
    uSimMat = []
    with open(filename, "r") as f:
        for line in f.readlines():
            temp = []
            line = line.strip('\n')
            arr = line.split()
            for item in arr:
                temp.append(float(item))
            uSimMat.append(temp)
    return uSimMat


def load_item_sim_file(filename):
    iSimMat = []
    with open(filename, "r") as f:
        for line in f.readlines():
            temp = []
            line = line.strip('\n')
            arr = line.split()
            for item in arr:
                temp.append(float(item))
            iSimMat.append(temp)
    return iSimMat


def load_negative_file(filename, drug):
    negativeList = [[] for _ in range(drug)]
    with open(filename) as f:
        line = f.readline()
        while line is not None and line != "":
            line = line.strip('\n')
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            negativeList[user].append(item)
            line = f.readline()
    return negativeList


