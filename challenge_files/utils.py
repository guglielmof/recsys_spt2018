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
import math
import numpy as np
import sys
import json

def fl():
	sys.stdout.flush()
# Calculates the R-precision metric
def r_precision(ord_pred, target_set):

	return len(set(ord_pred[0:len(target_set)]) & target_set) / float(len(target_set))


# Calculates the normalized discounted cumulative gain at k
def ndcg_k(ord_pred, target_set):
	#k = set(ord_pred) & target_set
	idcg = idcg_k(len(target_set))
	dcg_k = sum([int(ord_pred[i] in target_set) / math.log(i+2, 2) for i in range(len(ord_pred))])
	return dcg_k / idcg


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
	res = sum([1.0/math.log(i+2, 2) for i in range(k)])
	if not res: return 1.0
	else: return res

def click(ord_pred, target_set):
	k = 0
	for i in ord_pred:
		k+=1
		if i in target_set:
			return math.floor((k-1)/10)
	return 51

def validation(recommendation, test_set, list_of_tracks, bucket, verbose=False):
	rp = []
	ndcg = []
	click_vals = []
	bad_pls = []
	for u in recommendation:	
		r = recommendation[u]
		ground =  set([list_of_tracks[int(t)] for t in test_set[u][bucket:] if t not in set(test_set[u][:bucket])])
		rp.append(r_precision(r, ground))
		#print(rp[-1])
		ndcg.append(ndcg_k(r, ground))
		#print(ndcg[-1])
		click_vals.append(click(r, ground))
		#print(click_vals[-1])
		if(click_vals[-1]>40):
			bad_pls.append(u)
	print("r-precision: " + str(np.mean(rp)))
	print("ndcg: " + str(np.mean(ndcg)))
	print("click mean: " +str(np.mean(click_vals)))
	if(verbose):
		print("bad playlists: ")
		print(bad_pls)
	fl()
	return np.mean(rp), np.mean(ndcg), np.mean(click_vals)



def scores_sorter(scores, n = 600):
	s = np.argpartition(-scores, n)[:n]

	s = s[np.argsort(-scores[s])]
	return s



def reduced_binom2(N, k, d=2):
	return np.prod([float(k-i)/(N-i) for i in range(d)])
	
def reduced_binom(N, k, d=2):
	b = 1.
	for i in range(d):
		b *= float(k-i)/(N-i)
	return b

def reduced_disjunctive(N, x, z, xintz, d=2):
	a = reduced_binom(N, N-x, d)
	b = reduced_binom(N, N-z, d)
	c = reduced_binom(N, N-x-z+xintz, d)
	return 1. - a - b + c

def identity_disjunctive(N, x, d=2):
	return 1. - reduced_binom(N, N-x, d)

def imputation_single(x, z, N, d=2):
	if d == 2:
    		sx = math.sqrt(identity_disjunctive(N, x))
    		sz = math.sqrt(identity_disjunctive(N, z))
    		return (2. / (sx*sz)) * (x / N) * (z / (N-1.))
	else:
		return reduced_disjunctive(N, x, z, 0, d)
  

def jload(path):
	loaded_file = json.load(open(path, "r"))
	print(path+" loaded")
	fl()
	return 	loaded_file

def split_test_set(pos, size, bucket):
	idx = int(pos)*size

	tot_test = jload("../files/validation/validation_playlists_pids.txt")
	pl_test_list = tot_test[bucket][idx:idx+size]
	return pl_test_list

def save_recommendation(path, recommendation, intestation = False):
	with open(path, "w") as F:
		if intestation:
			F.write("team_info,main,IN3PD,guglielmo.faggioli@studenti.unipd.it\n")
		for r in recommendation:
			F.write(str(r)+","+",".join(recommendation[r])+"\n")
