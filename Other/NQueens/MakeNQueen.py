import itertools
import random
import pandas as pd
import os
import numpy as np
import json
from cpmpy import * 
from cpmpy.solvers import CPM_ortools
from random import randrange
import sys



def nqueen_sq(square):
    iSize = len(square[0])

    #Diagonals Part
    result1 = 0
    for i in range(0,iSize):
        result1 +=square[i][i]
    
    result2 = 0
    for i in range(iSize-1,-1,-1):
        result2 +=square[i][i]

    diags = [square[::-1,:].diagonal(i) for i in range(-iSize+1,iSize)]
    diags.extend(square.diagonal(i) for i in range(iSize-1,-iSize,-1))

    return [[sum(row) == 1 for row in square],
            [sum(col) == 1 for col in square.T],
            [sum(dia) <= 1 for dia in diags]]



def model_nqueen_sq(N):
    square = intvar(0,1, shape=(N,N))
    return square, Model(nqueen_sq(square))

def make_inst_nqueen(N, pos):
    binData = dict()  
    binData['solutions'] = []
    binData['shortSolutions'] = []

    
    (square,m) = model_nqueen_sq(N)
    s = SolverLookup.get("ortools", m)

    while len(binData['solutions']) < pos:

        # get random assignment
        rand = np.random.randint(0,2, size=(N,N))
        s.solution_hint(square.flatten(), rand.flatten())
        s.solve()

        joined = []
        for row in square.value().tolist():
            for index in range(0, len(row)):
                row[index] += 1
            joined += row

        incremented = []
        for row in square.value().tolist():
            for index in range(0, len(row)):
                row[index] += 1
            incremented.append(row)

        binData['solutions'].append(dict([('board', incremented)]))
        binData['shortSolutions'].append(''.join(map(str,joined)))

        s += any(square != rand)
    
    return binData

if len(sys.argv) != 3:
    n = 3
    sys.argv = [None] * n
    sys.argv[1] = 8
    sys.argv[2] = 20

N = int(sys.argv[1])
trainInstanceAmnt = int(sys.argv[2])
testInstanceAmnt = int(trainInstanceAmnt * 0.2)
print(N)

print("start finding solutions")
data = dict()
data['solutions'] = []
data['shortSolutions'] = []
step = 20
counter = 0
while len(data['solutions']) < (trainInstanceAmnt + testInstanceAmnt):
    print("currently on " + str(len(data['solutions'])), end='\r')
    extratraindata = make_inst_nqueen(N, step)
    counter += step
    for i in range(0, len(extratraindata['solutions'])):
        extraSol = extratraindata['solutions'][i]
        extraShort = extratraindata['shortSolutions'][i]
        if extraSol not in data['solutions']:
            data['solutions'].append(extraSol)
            data['shortSolutions'].append(extraShort)
print("done finding solutions")


outtraindata = dict()
outtraindata['solutions'] = data['solutions'][0:trainInstanceAmnt]
outtraindata['shortSolutions'] = data['shortSolutions'][0:trainInstanceAmnt]

outtestset = dict()
outtestset['solutions'] = data['solutions'][trainInstanceAmnt:trainInstanceAmnt + testInstanceAmnt]
outtestset['shortSolutions'] = data['shortSolutions'][trainInstanceAmnt:trainInstanceAmnt + testInstanceAmnt]

for train in outtraindata['solutions']:
    for test in outtestset['solutions']:
        if train == test:
            print("mag niet geprint worden!!!")

print("all data length is " + str(counter))
print("all unique data length is " + str(len(data['solutions'] )))
print("train length is " + str(len(outtraindata['solutions'] )))
print("test length is " + str(len(outtestset['solutions'] )))


path_data = 'Data/Nqueens/dim' + str(N) 

path_train_data = path_data + '/train'
path_test_data = path_data + '/test'

try:
    os.mkdir(path_data)
except FileExistsError:
    pass

try:
    os.mkdir(path_train_data)
except FileExistsError:
    pass

try:
    os.mkdir(path_test_data)
except FileExistsError:
    pass

json.dump(outtraindata, fp=open(f"{path_train_data}/instance_{trainInstanceAmnt}.json", 'w'))
json.dump(outtestset, fp=open(f"{path_test_data}/instance_{trainInstanceAmnt}.json", 'w'))
