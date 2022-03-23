import itertools
import random
import pandas as pd
import os
import numpy as np
import json
from cpmpy import * # pip3 install cpmpy
from cpmpy.solvers import CPM_ortools
from random import randrange
import sys



# latin square has rows/cols permutations (alldifferent)
def nqueen_sq(square):
    iSize = len(square[0])

    #Diagonals Part
    result1 = 0
    for i in range(0,iSize):
        result1 +=square[i][i]
    
    result2 = 0
    for i in range(iSize-1,-1,-1):
        result2 +=square[i][i]

    # iSize = 4
    # square = np.array(
    #      [[-2,  5,  3,  2],
    #       [ 9, -6,  5,  1],
    #       [ 3,  2,  7,  3],
    #       [-1,  8, -4,  8]])

    # diags = [matrix[::-1,:].diagonal(i) for i in range(-3,4)]
    # diags.extend(matrix.diagonal(i) for i in range(3,-4,-1))
    # print([n.tolist() for n in diags])

    diags = [square[::-1,:].diagonal(i) for i in range(-iSize+1,iSize)]
    diags.extend(square.diagonal(i) for i in range(iSize-1,-iSize,-1))
    # print([n.tolist() for n in diags])

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
    binData['shortHints'] = []

    
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
            # print(row)

        incremented = []
        for row in square.value().tolist():
            for index in range(0, len(row)):
                row[index] += 1
            incremented.append(row)
            # print(row)


        # print(incremented)
        # print(square.value().tolist())


        binData['solutions'].append(dict([('board', incremented)]))

        binData['shortSolutions'].append(''.join(map(str,joined)))

        hints = joined[:2]
        hints += [0]* (N**2 -2)
        hints = ''.join(map(str,hints))
        binData['shortHints'].append(hints)

        s += any(square != rand)
    
    return binData

# a = make_inst_nqueen(4, 1)
N = int(sys.argv[1])
instanceAmnt = int(sys.argv[2])

# outdata = make_inst(N, i, N*50, N*50)
print("start finding solutions")
outtraindata = dict()
outtraindata['solutions'] = []
outtraindata['shortSolutions'] = []
outtraindata['shortHints'] = []
for i in range(0, instanceAmnt, 20):
    print("currently on " + str(i), end='\r')
    extratraindata = make_inst_nqueen(N, 20)
    outtraindata['solutions'].extend(extratraindata['solutions'])
    outtraindata['shortSolutions'].extend(extratraindata['shortSolutions'])
    outtraindata['shortHints'].extend(extratraindata['shortHints'])

outtestset = make_inst_nqueen(N, int(instanceAmnt * 0.2))
print("done finding solutions")

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

json.dump(outtraindata, fp=open(f"{path_train_data}/instance_{instanceAmnt}.json", 'w'))
json.dump(outtestset, fp=open(f"{path_test_data}/instance_{instanceAmnt}.json", 'w'))




    
