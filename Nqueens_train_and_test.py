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

# computes a best solution of the CFN knowing only the hints as images
def find_best_sol12(CFNdata, fuzz_hints):
    tb2_time = 0
    # we add the unary cost functions that represent the confidence
    # scores of LeNet and use the heurisric ub
    cfn = toCFN(*CFNdata, weight = fuzz_hints,  btLimit = btlimit)
    cfn.UpdateUB(1e-6)
    ctime = time.process_time()                
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime    
    del cfn
    # else we relax our bound
    if (not sol):
        cfn = toCFN(*CFNdata, weight = fuzz_hints, btLimit = btlimit)
        ctime = time.process_time()            
        sol = cfn.Solve()
        tb2_time += time.process_time()-ctime
        del cfn
    return sol, tb2_time

def MNIST_transform(sample, sample_idx):
    lw = [[] for i in sample]
    for idx,c in enumerate(sample):
        if (c != '0'):
            img_idx = hash(c+str(idx)+str(sample_idx)) % exp_logits_len[int(c)]
            lwi = list(map(lambda x: max(0,-math.log(x)), exp_logits[int(c)][img_idx][1:]))
            minlwi = min(lwi)
            lw[idx] = list(map(lambda x: (x - minlwi),lwi))
    return lw

# decodes a grid (hint or solution) assuming it needs to be Lenet decoded
def MNIST_decode(sample, sample_idx):
    dec = []
    for idx,c in enumerate(sample):
        if (c != '0'):
            img_idx = hash(c+str(idx)+str(sample_idx)) % exp_logits_len[int(c)]
            dec.append(exp_logits[int(c)][img_idx][1:].argmax()+1)
        else:
            dec.append(0)
    return dec

# print the true solution with hints, its LeNet decoded variant (if given, mode
# 1/2) and the predicted solution (if given/found)
def pgrid(mode,lt,lh,lp=None,ld=None):
    print()
    print("   S O L U T I O N            ",end='')
    if (ld): print("  D E C O D E D             ",end='')
    if (lp): print("P R E D I C T E D",end='')
    print('\n')
    for i in range(9):
        for j in range(3):
            print(" ".join([Fore.WHITE+str(a+1) if a==b else Style.RESET_ALL+str(a+1) for a,b in zip(lt[i*9+j*3:i*9+j*3+3],lh[i*9+j*3:i*9+j*3+3])]),end='   ')
        print(end='    ')
        if (ld and mode == 2):
            for j in range(3):
                print(" ".join([Fore.GREEN+str(a) if (a-1)==b else Fore.RED+str(a) for a,b in zip(ld[i*9+j*3:i*9+j*3+3],lt[i*9+j*3:i*9+j*3+3])]),end='   ')
            print(end='    ')
        if (ld and mode == 1):
            for j in range(3):
                print(" ".join([Style.RESET_ALL+"-" if b<0 else Fore.GREEN+str(a) if (a-1)==b else Fore.RED+str(a) for a,b in zip(ld[i*9+j*3:i*9+j*3+3],lh[i*9+j*3:i*9+j*3+3])]),end='   ')
            print(end='    ')      
        if (lp):
            if (mode > 0):
                for j in range(3):
                    print(" ".join([Fore.GREEN+str(min(9,a+1)) if a==b else Fore.RED+str(min(9,a+1)) for a,b in zip(lp[i*9+j*3:i*9+j*3+3],lt[i*9+j*3:i*9+j*3+3])]),end='   ')
            else:
                for j in range(3):
                    print(" ".join([Fore.WHITE+str(b+1) if c==b else Fore.GREEN+str(min(9,a+1)) if a==b else Fore.RED+str(min(9,a+1)) for a,b,c in zip(lp[i*9+j*3:i*9+j*3+3],lt[i*9+j*3:i*9+j*3+3],lh[i*9+j*3:i*9+j*3+3])]),end='   ')                    
        print()
        if (i%3 == 2): print()
    print(Style.RESET_ALL)

# print the true solution with hints, its LeNet decoded variant (if given, mode
# 1/2) and the predicted solution (if given/found)
def pgridLatin(mode,lt,lh, num_nodes, size_block, lp=None,ld=None):
    print()
    print("   S O L U T I O N            ",end='')
    if (ld): print("  D E C O D E D             ",end='')
    if (lp): print("P R E D I C T E D",end='')
    print('\n')
    dim = int(sqrt(num_nodes))
    num_blocks_per_row = 1
    for i in range(dim):
        print(end='   ')
        for j in range(num_blocks_per_row):
            print(" ".join([Fore.WHITE+str(a+1) if a==b else Style.RESET_ALL+str(a+1) 
            for a,b in zip(lt[i*dim+j*size_block:i*dim+j*size_block+size_block],lh[i*dim+j*size_block:i*dim+j*size_block+size_block])]),end='   ')
            #for a, b in zip(lt[i*dim+j*size_block:i*dim+j*size_block+size_block],lh[i*dim+j*size_block:i*dim+j*size_block+size_block]):
            #   print(" ".join(Fore.WHITE+str(a+1) if a==b else Style.RESET_ALL+str(a+1)) ,end='   ')
        print(end='          ')
        if (lp):
            for j in range(num_blocks_per_row):
                print(" ".join([Fore.WHITE+str(b+1) if c==b else Fore.GREEN+str(min(dim,a+1)) if a==b else Fore.RED+str(min(dim,a+1)) 
                for a,b,c in zip(lp[i*dim+j*size_block:i*dim+j*size_block+size_block],lt[i*dim+j*size_block:i*dim+j*size_block+size_block],lh[i*dim+j*size_block:i*dim+j*size_block+size_block])]),end='   ')                    
        print()
    print(Style.RESET_ALL)


# checks wether all items in a list are different 
def all_diff(list:list):
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if list[i] == list[j]:
                return False
    return True

# checks wether a list containt numbers below a certain maximum number
def all_below(list:list, nb:int):
    for i in list:
        if i > nb:
            return False
    return True

# Check wether square is correct latin square
def check_correct_latin_sq(square):
    columns:list[list] = [[] for _ in range(len(square[0]))]

    for row in square:
        if not all_diff(row) or not all_below(row, len(row)):
            return False
        i = 0

        for col in row:
            columns[i].append(col)
            i += 1
    
    for col in columns:
        if not all_diff(col) or not all_below(col, len(col)):
            return False

    return True

def make_list_of_list(list):
    import math
    listAmnt = int(math.sqrt(len(list)))
    listoflists = [[] for _ in range(listAmnt)]

    i = 0
    j = 0

    while i < len(list):
        listoflists[j] = list[i:i + listAmnt]
        j += 1
        i += listAmnt
    return listoflists

def LatinPrinter(sol):
    for index, item in enumerate(sol, start=1):
        print(item, end=' ' if index % 9 else '\n')


def make_hint(sol, nmb_left_out_queens, nmb_left_out_nodes, nmb_nodes):
    nmb_rows = int(np.sqrt(nmb_nodes))
    nmb_col = nmb_rows
    dropped_queens= random.sample(range(0, nmb_rows), nmb_left_out_queens)
    dropped_nodes = random.sample(range(0, nmb_nodes), nmb_left_out_nodes)
    # print(dropped)
    for index in dropped_nodes:
        sol[index] = 0

    for row_index in dropped_queens:
        for i in range(0, nmb_col):
            if sol[row_index*nmb_col + i] == 2:
                sol[row_index*nmb_col + i] = 0
    return sol

# Main section

Norms = ["l1","l2","l1_l2"]

if (len(sys.argv) not in [4,5]):
    n = 4
    sys.argv = [None] * n
    sys.argv[1] = "0"
    sys.argv[2] = "13000"
    sys.argv[3] = "satnet-test"

    #print("Bad number of arguments!")
    #print("mode [0-2] training sample size [1000-180000] test set [satnet-test/rrn-test-??] {btlimit}")
    #exit()
    
btlimit = 50000
if (len(sys.argv) == 5):
    btlimit = int(sys.argv[4])
num_sample = 30 #komt van 1000

num_nodes = 64
num_values = 2
mode = int(sys.argv[1])
norm_type = Norms[0]
training_size= int(sys.argv[2])

testset_path = os.path.join("Other/NQueens/TestSets/8x8",os.path.join("instance"+".json"))

with open(testset_path, "r") as testset_file:
    dict_testset = json.load(testset_file)

list_data = list()
for square_dict in dict_testset['solutions']:
    square_data = [nb for row in square_dict['square'] for nb in row]

    list_data.append(square_data)

testset_data = np.asarray(list_data)
testset_data = [tuple(row) for row in testset_data]
testset_data = np.unique(testset_data, axis= 0) #delete dublicates
test_sols = list()
test_hints = list()
nmb_left_out_queens = 2
nmb_left_out_nodes = 3
for numList in testset_data:
    hint = make_hint(numList.copy(), nmb_left_out_queens, nmb_left_out_nodes, num_nodes)
    sol_num = int(''.join(map(str,numList)))
    hint_num = int(''.join(map(str,hint)))
    hint_str = str(hint_num)
    while (len(hint_str) < num_nodes):
        hint_str = '0' + hint_str
    test_sols.append(str(sol_num))
    test_hints.append(hint_str)



dim_test_sols = len(test_sols)

# test_hints = ['0'] * dim_test_sols

# test_CSV = pd.read_csv(test_set,sep=",",nrows=num_sample,header=None).values
# test_hints = test_CSV[:][:,3]
# test_sols = test_CSV[:][:,2]

# for noisy hints
exp_logits = pickle.load(lzma.open(os.path.join("Sudoku","LeNet-outputs/MNIST_test_marginal.xz"), "rb"))
exp_logits_len = list(map(lambda x: len(x), exp_logits))


m = np.array([1, num_nodes])    # number of nodes for each type of distribution (plus initial value 1)
dim = np.array([1, num_values]) # dimension for each type of distribution (plus initial value 1)
d = np.sum(np.multiply(m, dim))-1

# Load the precomputed A matrix
A_matrix_path = '/home/wout/CFN-learn/Other/NQueens/Amatrices/Amatrix.csv'
with open(A_matrix_path) as filename:
    A = np.loadtxt(filename, delimiter=",")

print(A)
# A = lzma.open(A_matrix_path,"rb")

with open(os.path.join("Sudoku","lambdas/lambda-"+sys.argv[1]+"-"+sys.argv[2]),'r') as f:
    lamb = float(f.read())
    lamb = 0.3050527890267027 
    # lamb = 0.5
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
    if (mode == 0):
        # mode 0: ltruth and hints are available as digits
        sol, tb2_time = find_best_sol0(CFNdata, lhint.copy())
    else:
        sol, tb2_time = find_best_sol12(CFNdata, MNIST_transform(hint,s))
    total_tb2_time += tb2_time
    
    if (sol):
        #deze lijn hier onder doet het printen
        sol2 = make_list_of_list(sol[0])
        print(check_correct_latin_sq(sol2))
        # LatinPrinter(lhint)
        # print('\n')
        # LatinPrinter(sol[0])
        # print('\n')
        # LatinPrinter(ltruth)
        # print('\n')
        size_block = int(np.sqrt(num_nodes))
        pgridLatin(mode,ltruth,lhint,num_nodes, size_block, list(sol[0]), MNIST_decode(test_sols[s].strip(), s) if mode > 0 else None)
        if (mode < 2):
            # exact solution known, we can count the number of
            # wrong cells
            diff = sum(a != b for a, b in zip(list(sol[0]), ltruth))
        else:
            # the solution is only available as an image. We
            # compute the LeNet scores of the predicted digit
            # in the soft_max output of LeNet on the
            # handwritten digit used.
            lread = MNIST_transform(test_sols[s].strip(),s)
            diff = sum([lread[i][sol[0][i]] for i in range(num_nodes)])
        ndiff += diff
        if (diff > 0):
            print(Fore.RED,"Best solution has score:",diff," Sample",s+1,"/",num_sample,Style.RESET_ALL)
            bad += 1
        else:
            print(Fore.GREEN,"Zero score solution found. Sample",s+1,"/",num_sample,Style.RESET_ALL)
        # no solution found in the backtrack budget, this is bad, all cell predictions are bad too
    else:
        pgrid(mode,ltruth,lhint,None, MNIST_decode(test_sols[s].strip(), s) if mode > 0 else None)
        print(Fore.RED,"No solution found. Sample",s+1,"/",num_sample,Style.RESET_ALL)
        bad += 1
        ndiff += 81*(20 if mode == 2 else 1)
    aaaa = 1
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

