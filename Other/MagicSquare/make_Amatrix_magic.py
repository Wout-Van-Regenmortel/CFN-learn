import csv
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pickle
import json


def Amatrix(data, m_vec, dim_vec, balance, c):
    """
    data: the sufficient statistic data
    m_vec: array containing the number of nodes for each type of distribution (plus initial value 1)
    dim_vec: array containing the dimension for each type of distribution (plus initial value 1)
    balance = [1, balance1, balance2, balance3, balance4]
    c: binary vector that indicates the nodes that are discrete random variables
    """

    num_sam, d_num = np.shape(data)

    balance_vec = np.repeat(balance, m_vec*dim_vec)
    balance_mat = np.tile(balance_vec, (num_sam, 1))

    data_2 = np.concatenate((np.ones([num_sam, 1]), data), axis=1)
    new_data = np.divide(data_2, balance_mat)

    M = np.dot(new_data.T, new_data)/num_sam

    A = M + np.diag(c)

    return A

# read latin squares data file
path = '/home/wout/CFN-learn/Other/MagicSquare/Instances/Tests/5x5/instance.json'
    # path = '/home/wout/CFN-learn/Other/LatinSquare/Instances/Real'

with open(path, "r") as datafile:
    dict_data = json.load(datafile)

sq_dim = len(dict_data['solutions'][0]['square'][0])
list_data = list()
for square_dict in dict_data['solutions']:
    square_data = [nb for row in square_dict['square'] for nb in row]

    list_data.append(square_data)


np_data = np.asarray(list_data)
np_data = [tuple(row) for row in np_data]
np_data = np.unique(np_data, axis= 0) #delete dublicates
print(len(np_data))
# print(np_data)

# ONE HOT 3 DIMENSIONS 100 010 001 

dom = [*range(1, sq_dim**2 + 1)]
all_domains = [np.asarray(dom) for _ in range(sq_dim*sq_dim)]

onehotencoder = OneHotEncoder(categories=all_domains)
data = onehotencoder.fit_transform(list_data).toarray()
data = np.array(data, dtype=int)
# print(data[0])

num_sample_init = len(dict_data['solutions'])
num_nodes = sq_dim**2

def option1():
    m = np.ones(int(num_nodes+1), dtype=int)  # number of nodes for each type of distribution (plus initial value 1)
    list_dim = [1]
    for k in range(0, sq_dim**2):
        list_dim.append(sq_dim**2)
    dim = np.array(list_dim)  # dimension for each type of distribution (plus initial value 1)

    # vector that indicates the nodes that are discrete random variables
    c = np.zeros(int(sum(m*dim)))
    c[1:] = 1

    d = np.sum(np.multiply(m, dim))-1

    balance = np.ones(len(dim))
    return m, dim, c, balance



m, dim, c, balance = option1()
A = Amatrix(data, m, dim, balance, c)

pathout = '/home/wout/CFN-learn/Other/MagicSquare/Amatrices/Amatrix.csv'
with open(pathout, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(A)

print(A)

    # m = vector met lengte aantal variabelen +1, allemaal 1
    # dim = vector met lengte aantal variabelen +1, alle waarden = aantal mogelijke opties bij latin van 9x9 dus allemaal 9 buiten eerste, die is 1
    # balance = vector met lengte aantal variabelen +1, allemaal 1.0
    # c = vector met lengte som van vector product m en dim, eerste = 0 al de rest = 1
