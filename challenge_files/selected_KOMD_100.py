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


#this file first builds a selection for each playlist: a list of songs that are more likely to belongs to the
#final recommendation; in particular this selection is built according to popularity of each song for each playlist
#with the same title (after some preprocessing) as the paylist we want to build the recommendation for. then, songs
#are ordere with KOMD and a recommendation is built.
#in input, you should specify the seed of the test set you want to consider (in final submission we used this method 
#only with seed = 100) and the number of songs to select (50000)

test_seed = sys.argv[1]
k = int(sys.argv[2])

def train_mod(test_set, bucket, lp, s_id2spt, selection, K, q):
	modello = Models.CF_KOMD_selected(q, lp = lp, K = K, items_list =s_id2spt)
	modello.train(test_set, bucket, selection)
	recommendation = {}
	for i in test_set:
		recommendation[i] = modello.get_recommendation(i, bucket)
	#here you should specify the directory where to save the recommendation
	utils.save_recommendation("../sub_21_06/"+str(bucket)+".csv", recommendation)
	return r,n,c


def train_mod_comb(test_set, bucket, lp, s_id2spt, selection, K, q):
	modello = Models.CF_KOMD_selected(q, lp = lp, K = K, items_list =s_id2spt)
	modello.train(test_set, bucket, selection)
	recommendation = {}
	for i in test_set:
		recommendation[i] = modello.get_recommendation(i, bucket)
	
	spt2id = utils.jload("../data/s_spt2id.json")
	recommendation = {i:[spt2id[s] for s in recommendation[i]] for i in recommendation}
	return recommendation

val_set_keys = set()

##------------------TEST------------------###
K = sps.load_npz("./K.npz")
print("K loaded")
utils.fl()
q = np.load("./q.npy")
p2s_test = utils_matrix.load_test_set("./", "track")
p2s_train = utils.jload("./p2s.json")
p2t = utils.jload("./p2t_c.json")
s2p = utils.jload("./s2p.json")
##----------------END TEST----------------###


s_id2spt = utils.jload("./s_id2spt.json")

for s in s2p:
	s2p[s] = set(s2p[s])


###---------------BUILD SELECTION---------------###
t2p = utils.jload("./t2p_filt.json")


t_id2t = utils.jload("./t_id2t.json")
t2t_id = {}
for i, t in enumerate(t_id2t):
	t2t_id[t] = i

S = np.load("./S_titles.npy")


w2pop = {}
for i, t in enumerate(t2t_id):
	w2pop[t] = {}
	for pl in t2p[t]:
		if str(pl) not in val_set_keys:
			for song in set(p2s_train[str(pl)]):
				if(song in w2pop[t]):
					w2pop[t][song]+=1
				else:
					w2pop[t][song]=1

print("w2pop built")
utils.fl()

data = []
col = []
row =[]
for t in w2pop:
	for s in w2pop[t]:
		data.append(1.0*w2pop[t][s]) 
		col.append(int(s))
		row.append(t2t_id[t])
P = sps.csr_matrix((data, (col, row))) #the matrix needs to be transposed

print("P built")
utils.fl()

lengths =np.zeros(len(t2t_id))
for title in t2t_id:
	lengths[t2t_id[title]]= len(t2p[title])




global_popularity = np.zeros(len(s2p))
for s in s2p:
	global_popularity[int(s)] = len(s2p[s])
global_popularity = np.argsort(-global_popularity)

power = 10

def build_selection(k, p2s_test, global_popularity, t2t_id, S, p2t, lengths, test_seed, P, power):
	selection = {}

	for user in p2s_test[test_seed]:
		if(p2t[str(user)]=="" or p2t[str(user)] not in t2t_id):
			selection[user] = global_popularity[:k]
		else:
			row = S[t2t_id[p2t[str(user)]], :]**power/lengths
			selection[user] = utils.scores_sorter(P.dot(row), k)
	print("selection built")
	utils.fl()
	return selection


selection = build_selection(k, p2s_test, global_popularity, t2t_id, S, p2t, lengths, test_seed, P, power)
train_mod(p2s_test[test_seed], int(test_seed), 0.01, s_id2spt, selection, K, q)
