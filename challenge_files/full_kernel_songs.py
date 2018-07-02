import sys
import numpy as np
import os
import math
import scipy.sparse as sps
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils

'''
File used to build the Kernel between songs: songs are represented with playlists they belongs to. In each
element of the kernel matrix we will have the cosine similarity between representation of two songs
'''

#s2p = utils.jload("../data/validation/s2p.json")
s2p = utils.jload("../data/test/s2p.json")
for s in s2p:
	s2p[s] = set(s2p[s])

print "Creating s2d...", 
utils.fl()
s2d = {}
for ss in s2p:
	s = int(ss)
	if s not in s2d:
		s2d[s] = {}
	for p in s2p[ss]:
		if p not in s2d[s]:
			s2d[s][p] = 1.
		else:
			s2d[s][p] += 1.
print "done!"
utils.fl()

print "Creating norms...", 
utils.fl()
norms = [0.0]*len(s2p)
for s in s2d:
	for p in s2d[s]:
		norms[s] += s2d[s][p]**2
        if norms[s]==0:
                print 'NORM 0 !!'
                print s
                utils.fl()
	norms[s] = math.sqrt(norms[s])
print "done!"
utils.fl()

print "Creating X...", 
utils.fl()

data = []
row = []
col = []
for s in s2d:
	for p in s2d[s]:
		row.append(s)
		col.append(p)
		data.append(s2d[s][p]/norms[s])
print "done!"

print len(row), len(col), len(data)
utils.fl()

print "Saving X...",
utils.fl()
np.save("../data/test/kernels/data_X.npy", data)
np.save("../data/test/kernels/col_X.npy", col)
np.save("../data/test/kernels/row_X.npy", row)
print "done!"
utils.fl()

print "Computing X...",
utils.fl()
X = sps.csr_matrix((data, (row, col)))
print "done!"
utils.fl()

print "Computing X.T...",
utils.fl()
XT = sps.csr_matrix((data, (col, row)))
print "done!"
utils.fl()

print "Computing K...",
utils.fl()
K = X.dot(XT)
print "done!"
utils.fl()

print "Saving K...",
utils.fl()
#np.save("../data/training/kernels/K.npy", K)
sps.save_npz("../data/test/kernels/K_set", K)
print "done!",
utils.fl()

np.save("../data/test/kernels/q_set.npy",  np.mean(K, axis=0)) 
