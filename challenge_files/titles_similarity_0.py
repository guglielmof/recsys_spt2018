
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils
import numpy as np
import json
import scipy.sparse as sps
import utils_matrix
import math



#code used to build the recommendation in case of seed O.
s_id2spt = utils.jload("../data/s_id2spt.json")


q = 10

p2s_test = utils.jload("../data/test/p2s.json")
p2s_test = utils_matrix.load_test_set("../files", "track")
p2t = utils.jload("../data/words/p2t_c.json")
w2p = utils.jload("../data/test/words/t2p.json")


s2p = utils.jload("../data/validation/s2p.json")
s2p = {s: set(p) for s, p in s2p.items()}


t_id2t = utils.jload("../data/t_id2t.json")
t2t_id = {t: i for i, t in enumerate(t_id2t)}


S = np.load("../data/words/similarity_matrices/S_titles.npy")



# code use
w2pop = {}
for i, w in enumerate(t2t_id):
	w2pop[w] = {}
	for pl in w2p[w]:
		for song in set(p2s_train[str(pl)]):
			if(song in w2pop[w]):
				w2pop[w][song]+=1
			else:
				w2pop[w][song]=1

data = []
col = []
row =[]
for t in w2pop:
	for s in w2pop[t]:
		data.append(w2pop[t][s]) 
		col.append(int(s))
		row.append(t2t_id[t])
P = sps.csr_matrix((data, (col, row))) #the matrix needs to be transposed

recommendation = {}
for user in p2t:
	row = S[t2t_id[p2t[user]], :]**q
	recommendation[user] = [s_id2spt[s] for s in utils.scores_sorter(P.dot(row))][:500]

#here you should specify the directory where to save the recommendation
utils.save_recommendation("../sub_21_06/"+str(bucket)+".csv", recommendation)
