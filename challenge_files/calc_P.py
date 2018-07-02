import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import scipy.sparse as sps
import Models
import json
import utils
import utils_matrix
import random
import math


s2p = utils.jload("../data/training/s2p.json")

print "building R...", 
utils.fl()
data = []
row = []
col = []

lengths = np.zeros(len(s2p))
for s in s2p:
	p_set = set(s2p[s])
	lengths[int(s)] = 1.0/len(p_set)
	for p in p_set:
		data.append(1.0)
		row.append(p)
		col.append(int(s))

print("done"); utils.fl()
lengths = sps.csr_matrix(np.array(lengths))
'''
print "building P csc...",
utils.fl()
RT = sps.csc_matrix((data, (col, row)))
R =  sps.csc_matrix((data, (row, col)))
P = (RT.dot(R)).multiply(lengths)

print("done")
utils.fl()

print "saving P csc...",
utils.fl()
sps.save_npz("../data/training/item_based/P_csc.npz", P)
print "done"
utils.fl()
'''
print "building P...",
RT = sps.csr_matrix((data, (col, row)))
R =  sps.csr_matrix((data, (row, col)))
P = (RT.dot(R)).multiply(lengths)

print("done")
utils.fl()

print "saving P...",
utils.fl()
sps.save_npz("../data/training/item_based/P.npz", P)
print "done"
utils.fl()
