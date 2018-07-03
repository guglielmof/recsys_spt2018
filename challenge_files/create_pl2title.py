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
#!/usr/bin/python
import sys, os
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import utils
import json
import pickle
import string
import re
import numpy as np
from string import maketrans   # Required to call maketrans function.

from gensim import parsing

"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
"""
def damerau_levenshtein_distance(s1, s2):
	d = {}
	lenstr1 = len(s1)
	lenstr2 = len(s2)
	for i in xrange(-1,lenstr1+1):
		d[(i,-1)] = i+1
	for j in xrange(-1,lenstr2+1):
		d[(-1,j)] = j+1
 
	for i in xrange(lenstr1):
		for j in xrange(lenstr2):
			if s1[i] == s2[j]:
				cost = 0
			else:
				cost = 1
			d[(i,j)] = min(
						   d[(i-1,j)] + 1, # deletion
						   d[(i,j-1)] + 1, # insertion
						   d[(i-1,j-1)] + cost, # substitution
						  )
			if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
				d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
 
	return float(d[lenstr1-1,lenstr2-1])


t2pl = {}
for i in range(0,1000000,1000):
	fname = "mpd.slice.%d-%d.json" %(i, i+999)
	print "Processing file %s..." %fname,
	
	data = json.load(open("../mpd.v1/data/%s" %fname, "r+"))
	playlists = data["playlists"]
	for pl in playlists:
		t = pl["name"].lower()
		if t not in t2pl:
			t2pl[t] = []
		t2pl[t].append(pl["pid"])
	print " Done"
	
pickle.dump(t2pl, open("./structures/t2pl.pickle", "w"))
print "Saved"


def check_punct(s):
	for c in string.punctuation:
		if c in s: return True
	return False

def get_nums(s):
	return re.findall(r'\d+', s)

t2pl = pickle.load(open("./t2pl.pickle", "r"))



pl2t = {p:k for k,v in t2pl.items() for p in  v}
#json.dump(pl2t, open("./structures/pl2t.json", "w"), indent=4)
#json.dump(t2pl, open("./structures/t2p.json", "w"), indent=4)
'''
print len(t2pl)
def tclusters(titles, thr=.8):
	clusters = {}
	for i,t in enumerate(titles):
		print i
		clusters[t] = []
		for tc in titles:
			if damerau_levenshtein_distance(t, tc)/max(len(t), len(tc)) <= .2:
				clusters[t].append(tc)
		print t, clusters[t] 
	return clusters

print tclusters(t2pl.keys())
'''

punctuation = "?!,.-;()/\|^:_#'"

t2pl_strip = {}
for k,v in t2pl.items():
	t = ''.join(c if c not in punctuation else " " for c in k)
	t = re.sub(' +',' ', t)
	t = t.strip()
	t = parsing.stem_text(t)
	if t not in t2pl_strip:
		t2pl_strip[t] = []
	t2pl_strip[t] += v
	
# OCCHIO!!!!
t2pl = t2pl_strip
pl2t = {p:k for k,v in t2pl.items() for p in  v}

##if "stereo" in t2pl:
##	print "stereo", len(t2pl["stereo"])
##if "universal" in t2pl:
##	print "universal", len(t2pl["universal"])
	
json.dump(t2pl, open("./t2p_filt.json", "w"), indent=4)
json.dump(pl2t, open("./p2t_filt.json", "w"), indent=4)

print "TRAINING SET"
print "Total number of titles",
print len(t2pl)
print

t2pl_len = {k:len(p) for k,p in t2pl.items()}
t2pl_len_sorted = sorted(t2pl_len.items(), key=lambda (k, v): v, reverse=True)
print "20 most popular titles"
print t2pl_len_sorted[:100]
print

pun = {k:v for k,v in t2pl_len.items() if check_punct(k)}
print "Number of unique titles with punctuation", len(pun)
print "Number of playlits with punctuation in the title", sum(pun.values())

nums, tot = [], 0
for k,v in t2pl_len.items():
	n = get_nums(k)
	nums += n
	if n: tot += v

print "Number of numerical values in titles", tot
nums = set(map(int, nums))
print "List of uniqe numerical values in titles"
print sorted(nums)
print "Number of unique numerical values in titles", len(nums)
print

print "Percentile number of playlists per title"
T = [t[1] for t in t2pl_len_sorted]
for i in range(5, 101, 5):
	print "%d %d" %(i, np.percentile(T, i))
print


print "CHALLENGE SET"

chset = json.load(open("../CHALLENGE_SET/challenge_set.json", "r"))
chplaylists = chset["playlists"]
count = 0
skipped = 0
ch_t = []

p2t_c = {}

for i,pl in enumerate(chplaylists):
	if "name" in pl:
		#t = pl["name"].lower()
		t = pl["name"].lower()
		t = ''.join(c if c not in punctuation else " " for c in t)
		t = re.sub(' +',' ', t)
		t = t.strip()
		t = parsing.stem_text(t)
		ch_t.append(t)
		count += t not in t2pl
		p2t_c[pl['pid']] = t
	else:
		skipped += 1
		p2t_c[pl['pid']] = ''

json.dump(p2t_c, open("./p2t_c.json", "w"), indent=4)
titles_list = [t for t in t2pl if t!=""]
json.dump(titles_list, open("./t_id2t.json", "w"))
print "Number of titles not in the training", count
print "Number of empty titles", skipped

ch_t = set(ch_t)

f = open("chset_title_dist.txt", "w")
ch_list = []
for t in ch_t:
	if t in t2pl:
		s = "%s\t%d" %(t.encode('utf-8'), len(t2pl[t]))
		ch_list.append((len(t2pl[t]), t.encode('utf-8')))
	else:
		s = "%s\t0" %t.encode('utf-8')
		ch_list.append((0, t.encode('utf-8')))
	

ch_list = sorted(ch_list, reverse=True)
f.writelines(["%s\t%d\n" %(v,k) for k,v in ch_list])
f.close()


p2s = utils.jload("./p2s.json")
t2s = {}
for t in t2pl:
	t2s[t] = []
	for p in t2pl:
		t2s[t]+=p2s[str(p)]
json.dump(t2s, open("./t2s.json", "w"))


