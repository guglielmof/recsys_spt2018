'''
Copyright 2018 IN3PD

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
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
s2p = utils.jload("./s2p.json")
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
np.save("./data_X.npy", data)
np.save("./col_X.npy", col)
np.save("./row_X.npy", row)
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
sps.save_npz("./K.npz", K)
print "done!",
utils.fl()

np.save("./q.npy",  np.mean(K, axis=0)) 
