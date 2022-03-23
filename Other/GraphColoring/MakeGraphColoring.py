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


def inside(new_row, found_rows):
    for found_row in found_rows:
        if (new_row == found_row).all():
            return True
    return False

def indirect_connected(square, found_rows, not_fully_searched): 
    new_found_rows = found_rows.copy()
    new_not_fully_searched = not_fully_searched.copy()
    for row in not_fully_searched:

        direct_connections = get_dir_con(square, row)
        for new_row in direct_connections:

            if not (inside(new_row, found_rows)):
                new_found_rows.append(new_row)
                new_not_fully_searched.append(new_row)
        new_not_fully_searched.remove(row)

    if len(new_not_fully_searched) == 0:
        return new_found_rows
    else:
        return indirect_connected(square, new_found_rows, new_not_fully_searched)



def get_dir_con(square, row):
    direct_connections = []
    for j in range(0, len(row)-1):
        path = row[j]
        if path == 1:
            direct_connections.append(square[j])
    return direct_connections

def connected(square):
    row = square[0]
    connections = []

    direct_connections = get_dir_con(square, row)
    connections.extend(direct_connections)

    indirect_paths = indirect_connected(square, connections, direct_connections)
    connections = indirect_paths
    # connections = [tuple(row) for row in connections]
    # connections = np.unique(connections, axis= 0) #delete dublicates
    if len(connections) != len(square):
        return False
    else:
        return True

def connected_constraints2(square):
    constraints = []
    for row_index in range(0, len(square)):
        cur_row = square[row_index]
        for col_index in range(0, len(square[0])-1):
            sum = cur_row[col_index]
            for other_row_index in range(0, len(square)):
                cur_other_row = square[other_row_index]
                if other_row_index != row_index:
                    sum + cur_row[other_row_index]*cur_other_row[col_index]
            constraints.append(sum>0)


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


# square = np.array(
#         [[0,  1,  1,  0,  0,  1],
#         [ 1,  0,  0,  0,  0,  2],
#         [ 1,  0,  0,  1,  0,  3],
#         [ 0,  0,  1,  0,  1,  4],
#         [ 0,  0,  0,  1,  0,  5]])   

# print(connected(square))

def get_col(row):
    return row[len(row)-1]

def neighbours_have_diff_color(square):
    for row in square:
        dir_cons = get_dir_con(square, row)
        for con in dir_cons:
            if get_col(row) == get_col(con):
                return 0
    return 1

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
        # rand = np.random.randint(1,nmb_colors-1)
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


# latin square has rows/cols permutations (alldifferent)
def graph_sq(square, num_colors):
    con_constr = connected_constraints(square)

    return   [con_constr, path_constr(square), 
                color_constr(square), 
                nmb_dir_con_constr(square, num_colors), 
                sym_constr(square),
                diff_color_constraints(square)]
    
    


def model_graph_sq(N, max_colors):
    square = intvar(0,max_colors+1, shape=(N,N+1))
    # for i in range(0, len(square)):
    #     row = square[i]
    #     color = intvar(2, max_colors+1)
    #     row[N] = color
    return square, Model(graph_sq(square, max_colors))

# def model_graph_sq(N, max_colors):                            #als kleuren 1-4
#     square = intvar(0,max_colors-1, shape=(N,N+1))
#     # for i in range(0, len(square)):
#     #     row = square[i]
#     #     color = intvar(2, max_colors+1)
#     #     row[N] = color
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
            # color = np.random.randint(0, max_colors-1)
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

        hints = joined[:2]
        hints += [0]* (N**2 -2)
        hints = ''.join(map(str,hints))
        binData['shortHints'].append(hints) 

        s += any(square != rand)
    
    return binData

# a = make_inst_graph(5, 4, 1)
N = int(sys.argv[1])
maxColors = int(sys.argv[2])
instanceAmnt = int(sys.argv[3])

# outdata = make_inst(N, i, N*50, N*50)
print("start finding solutions")
outtraindata = dict()
outtraindata['solutions'] = []
outtraindata['shortSolutions'] = []
outtraindata['shortHints'] = []
for i in range(0, instanceAmnt, 100):
    print("currently on " + str(i), end='\r')
    extratraindata = make_inst_graph(N, maxColors, 100)
    outtraindata['solutions'].extend(extratraindata['solutions'])
    outtraindata['shortSolutions'].extend(extratraindata['shortSolutions'])
    outtraindata['shortHints'].extend(extratraindata['shortHints'])

outtestset = make_inst_graph(N, maxColors, int(instanceAmnt * 0.2))
print("done finding solutions")

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

json.dump(outtraindata, fp=open(f"{path_train_data}/instance_{instanceAmnt}.json", 'w'))
json.dump(outtestset, fp=open(f"{path_test_data}/instance_{instanceAmnt}.json", 'w'))


    
