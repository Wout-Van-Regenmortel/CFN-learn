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

def connected_constraints(square):
    constraints = []
    cur_row = square[0]
    for col_index in range(0, len(square[0])-1):
        sum = cur_row[col_index]
        for other_row_index in range(0, len(square)):
            cur_other_row = square[other_row_index]
            if other_row_index != 0:
                sum = sum + cur_row[other_row_index]*cur_other_row[col_index]
        constraints.append(sum>0)
    return constraints

def diff_color_constraints(square):
    constraints = []
    color_index = len(square)
    for row_index in range(0, len(square)):
        cur_row = square[row_index]


        for other_row_index in range(0, len(square)):
            cur_other_row = square[other_row_index]
            if other_row_index != row_index:
                constraints.append(cur_row[color_index] != cur_other_row[row_index]*cur_other_row[color_index])


    return constraints


def path_constr(square):
    constraints = []
    for row_index in range(0, len(square)):
        cur_row = square[row_index]
        for col_index in range(0, len(square[0])-1):
            constraints.append(cur_row[col_index]<2)
    return constraints

def color_constr(square):
    constraints = []
    for row_index in range(0, len(square)):
        cur_row = square[row_index]
        constraints.append(cur_row[len(square)]>=2)
    return constraints

def nmb_dir_con_constr(square, nmb_colors):
    constraints = []
    for row_index in range(0, len(square)):
        cur_row = square[row_index]
        sum = 0
        for col_index in range(0, len(square[0])-1):
            sum += cur_row[col_index]
        constraints.append(sum<=nmb_colors-1)
    return constraints

def sym_constr(square):
    constraints = []
    for row_index in range(0, len(square)):
        cur_row = square[row_index]
        for col_index in range(0, len(square[0])-1):
            constraints.append(cur_row[col_index] == square[col_index][row_index])

    return constraints


def graph_sq(square, num_colors):
    con_constr = connected_constraints(square)

    return   [con_constr, 
                path_constr(square), 
                color_constr(square), 
                nmb_dir_con_constr(square, num_colors), 
                sym_constr(square),
                diff_color_constraints(square)]
    
    


def model_graph_sq(N, max_colors):
    square = intvar(0,max_colors+1, shape=(N,N+1))
    return square, Model(graph_sq(square, max_colors))

# def model_graph_sq(N, max_colors):                            #als kleuren 1-4
#     square = intvar(0,max_colors-1, shape=(N,N+1))
#     return square, Model(graph_sq(square, max_colors))

def make_inst_graph(N, max_colors, pos):
    binData = dict()  
    binData['solutions'] = []
    binData['shortSolutions'] = []
    binData['shortHints'] = []

    
    (square,m) = model_graph_sq(N, max_colors)
    s = SolverLookup.get("ortools", m)

    while len(binData['solutions']) < pos:

        # get random assignment
        rand = np.random.randint(0,1+1, size=(N,N+1))
        for i in range(0, len(rand)):
            row = rand[i]
            color = np.random.randint(0, max_colors+1)
            # color = np.random.randint(0, max_colors-1)  #als kleuren 1-4
            row[N] = color
        s.solution_hint(square.flatten(), rand.flatten())
        s.solve()

        joined = []
        for i in square.value().tolist():
            joined += i
        
        incremented = []
        for row in square.value().tolist():
            for index in range(0, len(row)):
                row[index] += 1
            incremented.append(row)

        binData['solutions'].append(dict([('graph', incremented)]))

        binData['shortSolutions'].append(''.join(map(str,joined)))

        s += any(square != rand)
    
    return binData

if len(sys.argv) != 4:
    n = 4
    sys.argv = [None] * n
    sys.argv[1] = 10
    sys.argv[2] = 6
    sys.argv[3] = 100


N = int(sys.argv[1])
maxColors = int(sys.argv[2])
trainInstanceAmnt = int(sys.argv[3])
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
    extratraindata = make_inst_graph(N, maxColors, step)
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


path_data = 'Data/GraphColoring/nodes_' + str(N) + '_colors_' + str(maxColors) 

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


    
