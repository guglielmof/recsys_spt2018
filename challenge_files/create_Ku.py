import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import utils
import utils_matrix
import time
import scipy.sparse as sps
import time
import math
t=time.time()

#Script used to build the similarity matrix between users; it needs in input the number of the seed i want to use to
# build the matrix (0, 1, 5, 10, 25, 100) and alpha, used to combine two matrices: U_tr to Songs and (U_test to Song) transposed.
# our submission's results are obtained with alpha = 0.5


####--------------- TEST ---------------####
s2p = utils.jload("../data/test/s2p.json")
p2s = utils.jload("../data/test/p2s.json")
p2s_test = utils_matrix.load_test_set("../files", "track")
####--------------- TEST ---------------####

# ####------------ VALIDATION ------------####
# s2p = utils.jload("../data/validation/s2p.json")
# p2s = utils.jload("../data/validation/p2s.json")
# p2s_test = utils.jload("../data/validation/test/p2s.json")
# ####------------ VALIDATION ------------####

test_seed = sys.argv[1]
alpha = float(sys.argv[2])

if (len(sys.argv)>3):
	for i in range(3, len(sys.argv),2):
		if(sys.argv[i]=='-utr'):
				utr_str = "selu"
				utr_id2utr = utils.jload(sys.argv[i+1])
				utr2utr_id = {p: j for j,p in enumerate(utr_id2utr)}

		elif(sys.argv[i]=='-itr'):
			if(len(sys.argv)>i):
				itr_str = "seli"
				itr_id2itr = utils.jload(sys.argv[i+1])
				itr2itr_id = {s: j for j,s in enumerate(itr_id2itr)}


#items_training = utils.jload("")
#items_test = utils.jload("")

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
#build a matrix where each row represent a test user and each column represent an item;
#this matrix need to be normalized by row
for user in p2s_test[test_seed]:
	data_row = []
	for song in set(p2s_test[test_seed][user][:int(test_seed)]):
		if itr_id2itr==None or song in itr_id2itr:
			data_row.append(1.0)
			row.append(u2u_id[user])
			col.append(itr2itr_id[song])
	sq = len(data_row)**alpha
	data += [i/sq for i in data_row]

Rute = sps.csr_matrix((data,(row, col)), shape=(len(p2s_test[test_seed]), len(itr2itr_id)))
print("Rute built with shape: "+str(Rute.shape))
utils.fl()

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
	sq = len(data_row)**(1-alpha)
	data+=[i/sq for i in data_row]
RutrT = sps.csr_matrix((data, (col, row)))
print("RutrT built with shape: "+str(RutrT.shape))
utils.fl()

# in Ku we have a row for each test playlist an a column for each train playlist: the element Ku[i, j] corresponds to the intersection
# between songs in playlist i and playlist u devided by number of song in playlist i to the power of alpha times the number of songs in
# playlist j to the power of (1-alpha)
Ku = Rute.dot(RutrT)
print("Ku built with shape: "+str(Ku.shape))
utils.fl()

sps.save_npz("../data/training/user_based_matrices/%s_%.1f"%(test_seed, alpha, utr_str, itr_str), Ku)

print("done! time needed: "+str(time.time()-t))
