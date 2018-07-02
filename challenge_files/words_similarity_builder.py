import sys
import numpy as np
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import utils
import scipy.sparse as sps
import random


#Script used to build the similarity matrix for similarity between titles represented as the songs in playlists having the 
#said title.

def titles_similarity(t2rep, t2t_id, n_songs):

	A = np.zeros((len(t2t_id), r2r_id))
	for title in t2t_id:
		for rep in t2rep[title]:
			A[t2t_id[title], int(rep)] = 1.0

 	K = np.dot(A, A.T)
 	for i in range(K.shape[0]):
 		if K[i,i]==0:
 			K[i,i] = 1.0
 	n = K.shape[0]
	d = np.array([[K[i,i] for i in range(n)]])
	Kn = K / np.sqrt(np.dot(d.T,d))

	return Kn


t_id2t = utils.jload("./t_id2t.json")
t2t_id = {t:i for i, t in enumerate(t_id2t.items())}


t2rep = utils.jload("./t2s.json")
t2rep = {t:set(rep) for t, rep in enumerate(t2rep.items())}
rep_size = len(utils.jload("./s2p.json"))


for t in t2rep:
	t2rep[t] = set(t2rep[t]) 

sim = titles_similarity(t2rep, t2t_id, rep_size)
np.save("./S_titles.npy", sim)
