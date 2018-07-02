import sys
import os
import numpy as np
import utils
import utils_matrix
import time
import scipy.sparse as sps
import math


#the script implements a user based recommendation technique based on the output of build_Ku.py.
#it loads some needed files, calculates the global popularity to fill recommendations that are not
#long enought, calculates the rating matrix for the training set and calculates Ku*R. by using 
#it takes then the first 500 higher elements for each row and uses them as recommendation

test_seed = '1'
q = 1


s_id2spt = utils.jload("./s_id2spt.json")
####--------------- TEST ---------------####
s2p = utils.jload("./s2p.json")
p2s = utils.jload("./p2s.json")
p2s_test = utils_matrix.load_test_set("./", "track")
Ku = sps.load_npz("./1_0.5.npz")
####--------------- TEST ---------------####



global_popularity = np.zeros(len(s2p))
for s in s2p:
	global_popularity[int(s)] = len(s2p[s])
global_popularity = np.argsort(-global_popularity)


utr_str = "noutr"
utr_id2utr = None
utr2utr_id = {p: i for i,p in enumerate(sorted(list(map(int, p2s.keys()))))}

itr_str = "noitr"
itr_id2itr = None
itr2itr_id = {int(s): i for i,s in enumerate(sorted(list(map(int, s2p.keys()))))}


u_id2u = sorted(p2s_test[test_seed].keys())
u2u_id = {u: j for j,u in enumerate(u_id2u)}

data = []
row = []
col = []

for user in utr2utr_id:
	data_row = []
	for song in set(p2s[str(user)]):
		if itr_id2itr==None or song in itr_id2itr:
			data_row.append(1.0)
			row.append(utr2utr_id[user])
			col.append(itr2itr_id[song])
	data += data_row
R = sps.csr_matrix((data, (row, col)))
print("R built with shape: "+str(R.shape))
utils.fl()

itr_id2itr  = {itr2itr_id[s]:s for s in itr2itr_id}


Rhat = (Ku.power(q).dot(R))
print(Rhat.shape)
recommendation = {}
for user in u2u_id:
	print(user)
	scores = Rhat[u2u_id[user]].toarray()[0]
	idx = utils.scores_sorter(scores, 500)
	recommendation[str(user)] = [itr_id2itr[i] for i in idx if scores[i]>0] 

	recommendation[str(user)] = [s_id2spt[i] for i in recommendation[str(user)] if i not in p2s_test[test_seed][user][:int(test_seed)]][:500]
	k = 0
	while len(recommendation[str(user)])<500:
		track = s_id2spt[global_popularity[k]]
		if track not in recommendation[str(user)] and global_popularity[k] not in p2s_test[test_seed][user][:int(test_seed)]:
			recommendation[str(user)].append(track)
		k += 1

utils.save_recommendation("./1.csv", recommendation)
