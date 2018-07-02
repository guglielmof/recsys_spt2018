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


#this script is used to build the recommendation for seed 5, 10 and 25. it takes in input alpha (we used 0.7)
#q (we used 0.4) and the seed we want to build the recommendation for


alpha = float(sys.argv[1])
q = float(sys.argv[2])
test_seed = sys.argv[3]

#p2s_test = utils.jload("../data/validation/test/p2s.json")
p2s_test = utils_matrix.load_test_set("./")
s_id2spt = utils.jload("./s_id2spt.json")
print "loading P...",
utils.fl()
P = sps.load_npz("./P.npz")
print "done"
utils.fl()

print "loading P csc...",
utils.fl()
P_csc = sps.load_npz("./P_csc.npz")
print "done"
utils.fl()


print("prediction with alpha=%f, q=%f started"%(alpha, q))

recommendation = {}
for u in sorted(p2s_test[test_seed].keys()):
	Iu = p2s_test[test_seed][u][:int(test_seed)]
	Wu = (P_csc[:, Iu].power(alpha).multiply(P[Iu, :].power(1-alpha).transpose())).power(q)
	scores = np.array(np.sum(Wu, axis=1)).flatten()
	recommendation[u] = [s_id2spt[i] for i in utils.scores_sorter(scores) if i not in Iu][:500]
	print(u)
	utils.fl()

#here you should specify the directory where to save the recommendation
utils.save_recommendation("./%s.csv"%(test_seed), recommendation)
