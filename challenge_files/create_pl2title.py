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

t2pl = {}
for i in range(0,1000000,1000):
	fname = "mpd.slice.%d-%d.json" %(i, i+999)
	print "Processing file %s..." %fname,
	
	data = json.load(open("./mpd/data/%s" %fname, "r+"))
	playlists = data["playlists"]
	for pl in playlists:
		t = pl["name"].lower()
		if t not in t2pl:
			t2pl[t] = []
		t2pl[t].append(pl["pid"])
	print " Done"

def get_nums(s):
	return re.findall(r'\d+', s)

pl2t = {p:k for k,v in t2pl.items() for p in  v}

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
	
json.dump(t2pl, open("./t2p_filt.json", "w"), indent=4)
json.dump(pl2t, open("./p2t_filt.json", "w"), indent=4)

print "CHALLENGE SET"

chset = json.load(open("./challenge_set.json", "r"))
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

p2s = utils.jload("./p2s.json")
t2s = {}
for t in t2pl:
	t2s[t] = []
	for p in t2pl:
		t2s[t]+=p2s[str(p)]
json.dump(t2s, open("./t2s.json", "w"))


