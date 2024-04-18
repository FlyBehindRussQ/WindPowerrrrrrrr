import matplotlib.pyplot as plt
import random
import numpy as np
import math
from sklearn import datasets
import pandas as pd


def dist(t1, t2):
    dis = math.sqrt((np.power((t1[0]-t2[0]),2) + np.power((t1[1]-t2[1]),2)))
    # print("两点之间的距离为："+str(dis))
    return dis

def dbscan(Data_to_filter, eps, MinPts):
    num = len(Data_to_filter)
    unvisited = [i for i in range(num)]
    visited = []
    labels = [-1 for i in range(num)]
    k = -1

    while len(unvisited) > 0:
        point = random.choice(unvisited)
        unvisited.remove(point)
        visited.append(point)

        neighbour = []
        for i in range(num):
            if dist(Data_to_filter[i], Data_to_filter[point]) <= eps:
                neighbour.append(i)

        if len(neighbour)>=MinPts:
            k = k + 1
            labels[point] = k
            for p in neighbour:
                if p in unvisited:
                    unvisited.remove(p)
                    visited.append(p)
                    neigh = []
                    for j in range(num):
                        if dist(Data_to_filter[j], Data_to_filter[p]) <= eps:
                            neigh.append(j)
                    if len(neigh)>=MinPts:
                        for t in neigh:
                            if t not in neighbour:
                                neighbour.append(t)
                if labels[p]==-1:
                    labels[p] = k
        else:
            labels[point] = -1
    return labels

data = pd.read_csv("data_after_filter.csv")
speed = data.iloc[:, 2]
power = data.iloc[:, -1]
data = pd.concat([speed,power],axis=1)
dbscan(data,0.5,300)
