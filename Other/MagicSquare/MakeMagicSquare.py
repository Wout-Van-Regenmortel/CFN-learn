import random
import pandas as pd
import os
import numpy as np
import json
from cpmpy import * # pip3 install cpmpy
from cpmpy.solvers import CPM_ortools
from random import randrange

# Notebook to make magic square problems/instance

# latin square has rows/cols permutations (alldifferent)
def magic_sq(square):
    iSize = len(square[0])
    magic_num = sum(square[0])    

    #Diagonals Part
    result1 = 0
    for i in range(0,iSize):
        result1 +=square[i][i]
    
    result2 = 0
    for i in range(iSize-1,-1,-1):
        result2 +=square[i][i]

    a = np.ravel(square)

    return [[sum(row) == magic_num for row in square],
            [sum(col) == magic_num for col in square.T],
            result1 == magic_num,
            result2 == magic_num,
            AllDifferent(np.ravel(square))]



def model_magic_sq(N, max_int):
    square = intvar(1,max_int, shape=(N,N))
    return square, Model(magic_sq(square))

def make_inst_magic(N, max_int, pos):
    binData = dict()  
    binData['solutions'] = []
    binData['shortSolutions'] = []
    binData['shortHints'] = []

    
    (square,m) = model_magic_sq(N, max_int)
    s = SolverLookup.get("ortools", m)

    while len(binData['solutions']) < pos:

        # get random assignment
        rand = np.random.randint(1,max_int+1, size=(N,N))
        s.solution_hint(square.flatten(), rand.flatten())
        s.solve()

        joined = []
        for i in square.value().tolist():
            joined += i

        binData['solutions'].append(dict([('square', square.value().tolist())]))

        binData['shortSolutions'].append(''.join(map(str,joined)))

        hints = joined[:2]
        hints += [0]* (N**2 -2)
        hints = ''.join(map(str,hints))
        binData['shortHints'].append(hints)

        s += any(square != rand)
    
    return binData


N = 7
max_int = N**2
print(N)

# outdata = make_inst(N, i, N*50, N*50)
print("start finding solutions")
outtraindata = dict()
outtraindata['solutions'] = []
outtraindata['shortSolutions'] = []
outtraindata['shortHints'] = []
for i in range(0, 100, 1):
    print("currently on " + str(i), end='\r')
    extratraindata = make_inst_magic(N, max_int, 1)
    outtraindata['solutions'].extend(extratraindata['solutions'])
    outtraindata['shortSolutions'].extend(extratraindata['shortSolutions'])
    outtraindata['shortHints'].extend(extratraindata['shortHints'])

outtestset = make_inst_magic(N, max_int, 10)
print("done finding solutions")

directory = input("Enter directoryName: ")

parent_dir_traindata = '/home/wout/CFN-learn/Other/MagicSquare/Instances/Tests'
parent_dir_testset = '/home/wout/CFN-learn/Other/MagicSquare/TestSets'

# parent_dir = '/home/wout/CFN-learn/Other/LatinSquare/Real'
path_traindata = os.path.join(parent_dir_traindata, directory)
path_testset = os.path.join(parent_dir_testset, directory)

os.mkdir(path_traindata)
os.mkdir(path_testset)

print("Directory '% s' created" % directory)


json.dump(outtraindata, fp=open(f"{path_traindata}/instance.json", 'w'))
json.dump(outtestset, fp=open(f"{path_testset}/instance.json", 'w'))


    