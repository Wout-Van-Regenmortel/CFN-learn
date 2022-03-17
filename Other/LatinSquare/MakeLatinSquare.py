import pandas as pd
import os
import numpy as np
import json
from cpmpy import * # pip3 install cpmpy
from cpmpy.solvers import CPM_ortools
from random import randrange

# Notebook to make magic square problems/instance

# latin square has rows/cols permutations (alldifferent)
def latin_sq(square):
        return [[AllDifferent(row) for row in square],
                [AllDifferent(col) for col in square.T]]

def model_latin_sq(N):
    square = intvar(1,N, shape=(N,N))
    return square, Model(latin_sq(square))
(square,m) = model_latin_sq(4)

# m
# m.solve()
# print(square.value())
# m.solveAll()
# N=4
# np.random.randint(1,N+1, size=(N,N))

def make_inst(N, i, pos=200, neg=200):
    binData = dict()
    binData['problemType'] = 'type01'
    binData['instance'] = i
    binData['size'] = N
    binData['formatTemplate'] = dict()
    # N*N boolvars
    binData['formatTemplate']['square'] = [[dict([('high',N), ('low',1), ('type','dvar')])]*N ]*N
    binData['tests'] = []
    
    binData['solutions'] = []
    binData['nonSolutions'] = []
    
    # nonSolutions, uniformly
    (square,m) = model_latin_sq(N)
    while len(binData['nonSolutions']) < neg:
        s = SolverLookup.get("ortools", m)
        # get random assignment
        rand = np.random.randint(1,N+1, size=(N,N))
        s += [square == rand]
        # I make the same error here, that they need not be unique...
        if s.solve():
            binData['solutions'].append(dict([('square', rand.tolist())]))
        else:
            binData['nonSolutions'].append(dict([('square', rand.tolist())]))

    # solutions, uniformly guided, unique
    s = SolverLookup.get("ortools", m)
    while len(binData['solutions']) < pos:
        # get random assignment
        rand = np.random.randint(1,N+1, size=(N,N))
        s.solution_hint(square.flatten(), rand.flatten())
        s.solve()
        binData['solutions'].append(dict([('square', square.value().tolist())]))
        s += any(square != rand)
    
    return binData

def make_inst2(N, i, pos=10, neg=10):
    binData = dict()

    # binData['problemType'] = 'type01'
    # binData['instance'] = i
    # binData['size'] = N
    # binData['formatTemplate'] = dict()
    ## N*N boolvars
    # binData['formatTemplate']['square'] = [[dict([('high',N), ('low',1), ('type','dvar')])]*N ]*N
    # binData['tests'] = []
    
    binData['solutions'] = []
    # binData['nonSolutions'] = []
    binData['shortSolutions'] = []
    binData['shortHints'] = []

    
    # # nonSolutions, uniformly
    (square,m) = model_latin_sq(N)
    # while len(binData['nonSolutions']) < neg:
    #     s = SolverLookup.get("ortools", m)
    #     # get random assignment
    #     rand = np.random.randint(1,N+1, size=(N,N))
    #     s += [square == rand]
    #     # I make the same error here, that they need not be unique...
    #     if s.solve():
    #         binData['solutions'].append(dict([('square', rand.tolist())]))
    #     else:
    #         binData['nonSolutions'].append(dict([('square', rand.tolist())]))

    # solutions, uniformly guided, unique
    s = SolverLookup.get("ortools", m)
    while len(binData['solutions']) < pos:
        # get random assignment
        rand = np.random.randint(1,N+1, size=(N,N))
        s.solution_hint(square.flatten(), rand.flatten())
        s.solve()

        joined = []
        for i in square.value().tolist():
            joined += i

        # print(int(''.join(map(str,joined))))
        binData['solutions'].append(dict([('square', square.value().tolist())]))

        binData['shortSolutions'].append(''.join(map(str,joined)))

        hints = joined[:2]
        hints += [0]* (N**2 -2)
        hints = ''.join(map(str,hints))
        binData['shortHints'].append(hints)

        s += any(square != rand)
    
    return binData


N = 9
i = 0
print(N)

# outdata = make_inst(N, i, N*50, N*50)
outtraindata = make_inst2(N, i, 100, 0)
outtestset = make_inst2(N, i, 10, 0)

directory = input("Enter directoryName: ")

parent_dir_traindata = '/home/wout/CFN-learn/Other/LatinSquare/Instances/Tests'
parent_dir_testset = '/home/wout/CFN-learn/Other/LatinSquare/TestSets'

# parent_dir = '/home/wout/CFN-learn/Other/LatinSquare/Real'
path_traindata = os.path.join(parent_dir_traindata, directory)
path_testset = os.path.join(parent_dir_testset, directory)

os.mkdir(path_traindata)
os.mkdir(path_testset)

print("Directory '% s' created" % directory)


json.dump(outtraindata, fp=open(f"{path_traindata}/instance{i}.json", 'w'))
json.dump(outtestset, fp=open(f"{path_testset}/instance{i}.json", 'w'))


pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)

df_json = pd.read_json(f"{path_traindata}/instance{i}.json", dtype= {'shortSolutions': object})
df_json = pd.read_json(f"{path_testset}/instance{i}.json", dtype= {'shortSolutions': object})

print(df_json['shortSolutions'])
a = df_json['shortSolutions']

# df_json['solutions'] = df_json['solutions'].astype('float64')
# df_json['shortSolutions'] = df_json['shortSolutions'].astype('int64')

df_json.to_csv(f"{path_testset}/instance{i}.csv")

# exce = pd.read_csv(f"{path}/instance{i}.csv", dtype={'shortSolutions':np.object_})
# print(exce['shortSolutions'])
# print(exce.dtypes)
    