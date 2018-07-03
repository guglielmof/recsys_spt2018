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
import os
import utils
import numpy as np
import json
import scipy.sparse as sps
import utils_matrix
import math


bucket = '0'
#code used to build the recommendation in case of seed O.
s_id2spt = utils.jload("./s_id2spt.json")


q = 10

p2s_test = utils.jload("./p2s.json")
p2s_test = utils_matrix.load_test_set("./", "track")
p2t = utils.jload("./p2t_c.json")
w2p = utils.jload("./t2p_filt.json")


s2p = utils.jload("./s2p.json")
s2p = {s: set(p) for s, p in s2p.items()}


t_id2t = utils.jload("./t_id2t.json")
t2t_id = {t: i for i, t in enumerate(t_id2t)}


S = np.load("./S_titles.npy")



# code use
w2pop = {}
for i, w in enumerate(t2t_id):
	w2pop[w] = {}
	for pl in w2p[w]:
		for song in set(p2s[str(pl)]):
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
utils.save_recommendation("./"+str(bucket)+".csv", recommendation, intestation=True)
