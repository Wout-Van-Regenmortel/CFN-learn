from genericpath import exists
from importlib.resources import path
import pandas as pd
import os
import numpy as np
import json
from cpmpy import * 
from cpmpy.solvers import CPM_ortools
from random import randrange
import sys


# latin square has rows/cols permutations (alldifferent)
def latin_sq(square):
        return [[AllDifferent(row) for row in square],
                [AllDifferent(col) for col in square.T]]

def model_latin_sq(N):
    square = intvar(1,N, shape=(N,N))
    return square, Model(latin_sq(square))


def make_inst_latin(N, pos=10):
    binData = dict()
    binData['solutions'] = []
    binData['shortSolutions'] = []
    binData['shortHints'] = []

    
    (square,m) = model_latin_sq(N)
    s = SolverLookup.get("ortools", m)

    while len(binData['solutions']) < pos:

        # get random assignment
        rand = np.random.randint(1,N+1, size=(N,N))
        s.solution_hint(square.flatten(), rand.flatten())
        s.solve()

        joined = []
        for i in square.value().tolist():
            joined += i

        binData['solutions'].append(dict([('square', square.value().tolist())]))
        binData['shortSolutions'].append(''.join(map(str,joined)))
        s += any(square != rand)
    
    return binData

if len(sys.argv) != 3:
    n = 3
    sys.argv = [None] * n
    sys.argv[1] = 4
    sys.argv[2] = 30

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
    extratraindata = make_inst_latin(N, step)
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



path_data = 'Data/LatinSquare/dim' + str(N) 

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
