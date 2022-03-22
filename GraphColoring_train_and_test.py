#!/usr/bin/env python3
import csv
import lzma
from random import randrange
import random
import numpy as np
import math,time,sys,os
from numpy import linalg as la, sqrt
from PEMRF import *
import pandas as pd
from colorama import Fore, Style
import pickle

# computes a "best" solution of the CFN from exact hints. We try a
# heuristic ub and if no solution is found relax it.
def find_best_sol0(CFNdata,  hints):
    tb2_time = 0
    # we assign the hints and ask for a less than zero cost solution
    cfn = toCFN(*CFNdata, assign = hints,  btLimit = btlimit)
    # cfn.UpdateUB(1e-6)
    ctime = time.process_time()            
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime
    del cfn
    if (not sol):
        cfn = toCFN(*CFNdata, assign = hint, btLimit = btlimit)
        ctime = time.process_time()            
        sol = cfn.Solve()
        tb2_time += time.process_time()-ctime
        del cfn
    return sol, tb2_time

def make_list_of_list(list, width):
    listAmnt = width
    listoflists = [[] for _ in range(listAmnt-1)]

    i = 0
    j = 0

    while i < len(list):
        listoflists[j] = list[i:i + listAmnt]
        j += 1
        i += listAmnt
    return listoflists

def get_dir_con(square, row, current_index):
    direct_connections = []
    for j in range(0, len(row)-1):
        path = row[j]
        if path == 1 and j!=current_index:
            direct_connections.append(square[j])
    return direct_connections

def get_col(row):
    return row[len(row)-1]

def neighbours_have_diff_color(sol, width):
    square = make_list_of_list(sol, width)
    for i in range(0, len(square)):
        row = square[i]
        dir_cons = get_dir_con(square, row, i)
        for con in dir_cons:
            if get_col(row) == get_col(con):
                return False
    return True

# print the true solution with hints, its LeNet decoded variant (if given, mode
# 1/2) and the predicted solution (if given/found)
def pgridLatin(mode,lt,lh, width, lp=None,ld=None):
    print()
    print("   S O L U T I O N            ",end='')
    if (ld): print("  D E C O D E D             ",end='')
    if (lp): print("P R E D I C T E D",end='')
    print('\n')
    dim = width
    num_blocks_per_row = 1
    size_block = width
    for i in range(dim):
        print(end='       ')
        for j in range(num_blocks_per_row):
            print(" ".join([Fore.WHITE+str(a+1) if a==b else Style.RESET_ALL+str(a+1) 
            for a,b in zip(lt[i*dim+j*size_block:i*dim+j*size_block+size_block],lh[i*dim+j*size_block:i*dim+j*size_block+size_block])]),end='   ')
            #for a, b in zip(lt[i*dim+j*size_block:i*dim+j*size_block+size_block],lh[i*dim+j*size_block:i*dim+j*size_block+size_block]):
            #   print(" ".join(Fore.WHITE+str(a+1) if a==b else Style.RESET_ALL+str(a+1)) ,end='   ')
        print(end='                   ')
        if (lp):
            for j in range(num_blocks_per_row):
                print(" ".join([Fore.WHITE+str(b+1) if c==b else Fore.GREEN+str(min(dim,a+1)) if neighbours_have_diff_color(lp, width) else Fore.RED+str(min(dim,a+1)) 
                for a,b,c in zip(lp[i*dim+j*size_block:i*dim+j*size_block+size_block],lt[i*dim+j*size_block:i*dim+j*size_block+size_block],lh[i*dim+j*size_block:i*dim+j*size_block+size_block])]),end='   ')                    
        print()
    print(Style.RESET_ALL)



def make_hint(sol, width):
    # print(dropped)
    for index in range(0, len(sol)):
        if ((index % width) == (width -1)):
            sol[index] = 0
    return sol

# Main section

Norms = ["l1","l2","l1_l2"]

if (len(sys.argv) not in [4,5]):
    n = 4
    sys.argv = [None] * n
    sys.argv[1] = "0"
    sys.argv[2] = "13000"
    sys.argv[3] = "satnet-test"

    
btlimit = 50000
if (len(sys.argv) == 5):
    btlimit = int(sys.argv[4])


mode = int(sys.argv[1])
norm_type = Norms[0]
training_size= int(sys.argv[2])

testset_path = os.path.join("Other/GraphColoring/TestSets/graph4",os.path.join("instance"+".json"))

with open(testset_path, "r") as testset_file:
    dict_testset = json.load(testset_file)

list_data = list()
for square_dict in dict_testset['solutions']:
    square_data = [nb for row in square_dict['square'] for nb in row]

    list_data.append(square_data)

length = len(dict_testset['solutions'][0]['square'])
width = len(dict_testset['solutions'][0]['square'][0])
num_nodes = length* width


testset_data = np.asarray(list_data)
testset_data = [tuple(row) for row in testset_data]
testset_data = np.unique(testset_data, axis= 0) #delete dublicates
num_sample = len(testset_data)

test_sols = list()
test_hints = list()
for numList in testset_data:
    hint = make_hint(numList.copy(), width)
    sol_num = int(''.join(map(str,numList)))
    hint_num = int(''.join(map(str,hint)))
    hint_str = str(hint_num)
    test_sols.append(str(sol_num))
    test_hints.append(hint_str)



dim_test_sols = len(test_sols)


num_colors = 4
def create_list_domains(i, width_dim):
    if (i % width_dim) == 5:
        return [*range(3, 3+ num_colors)]
    else:
        return [*range(1, 2 + 1)]


list_dim = [1]
for i in range(0, length*width):
    list_dim.append(len(create_list_domains(i, width)))
dim = np.array(list_dim)  # dimension for each type of distribution (plus initial value 1)

# m = np.array([1, num_nodes])    # number of nodes for each type of distribution (plus initial value 1)
m = np.ones(int(num_nodes+1), dtype=int)
d = np.sum(np.multiply(m, dim))-1

# Load the precomputed A matrix
A_matrix_path = '/home/wout/CFN-learn/Other/GraphColoring/Amatrices/Amatrix.csv'
with open(A_matrix_path) as filename:
    A = np.loadtxt(filename, delimiter=",")

print(A)
# A = lzma.open(A_matrix_path,"rb")

with open(os.path.join("Sudoku","lambdas/lambda-"+sys.argv[1]+"-"+sys.argv[2]),'r') as f:
    lamb = float(f.read())
    # lamb = 5.424690937011328
    lamb = 0.5
print(Fore.CYAN + "Lambda is",lamb)

Z_init = np.ones([d+1,d+1])*0.2
U_init = np.zeros([d+1, d+1])
ctime = time.process_time()
CFNdata = ADMM(A, Z_init, U_init, lamb, training_size, m, dim, norm_type, 4)
ADMM_time = time.process_time()-ctime

func_count,exact = CFNcount(*CFNdata)
print("The CFN has",func_count,"binary functions")
print("The CFN has only (soft) differences: ",exact,Style.RESET_ALL)
        
ndiff = 0
bad = 0 
total_tb2_time = 0

for s,hint in enumerate(test_hints):
    ltruth = [int(v)-1 for v in test_sols[s].strip()]
    lhint = [int(v)-1 for v in hint]
    sol = []
    tb2_time = 0

    sol, tb2_time = find_best_sol0(CFNdata, lhint.copy())
    total_tb2_time += tb2_time
    
    if (sol):
        pgridLatin(mode,ltruth,lhint, width, list(sol[0]))
        diff = sum(a != b for a, b in zip(list(sol[0]), ltruth)) 

        ndiff += diff
        if (diff > 0):
            print(Fore.RED,"Best solution has score:",diff," Sample",s+1,"/",num_sample,Style.RESET_ALL)
            bad += 1
        else:
            print(Fore.GREEN,"Zero score solution found. Sample",s+1,"/",num_sample,Style.RESET_ALL)
        # no solution found in the backtrack budget, this is bad, all cell predictions are bad too
    else:
        print(Fore.RED,"No solution found. Sample",s+1,"/",num_sample,Style.RESET_ALL)
        bad += 1
        ndiff += 81*(20 if mode == 2 else 1)
probe = (lamb, num_sample, bad, ndiff/(num_sample*81), ADMM_time, total_tb2_time, func_count, exact)

print(Fore.CYAN +"======================================")
print("=====> Lambda:", lamb)
print("=====> Number of incorrect solutions :", bad,"/", num_sample)
print("=====> Ratio of wrongly guessed cells:", probe[3])
print("=====> ADMM cpu-time                 :", probe[4])
print("=====> Toulbar2 average cpu-time     :", probe[5]/num_sample)
print("=====> Number of functions in model  :", probe[6])
print("=====> Exact model                   :", probe[7])
print("======================================", Style.RESET_ALL)

filename = "test-"+sys.argv[1]+"-"+sys.argv[2]+"-"+sys.argv[3] 

with open(filename, 'w') as file:
    file.write("training_size,correct_grid_ratio,correct_cell_ratio,ADMM_time,total_toulbar2_time, funcnumber, exact\n")   
    file.write(str(training_size)+", "+str((num_sample-bad)/num_sample)+", "+str(1.0-probe[3])+", "+str(probe[4])+", "+str(probe[5])+", "+str(probe[6])+", "+str(int(probe[7])))
    file.write("\n")

