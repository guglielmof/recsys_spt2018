import os
import numpy as np
import scipy.sparse as sps
import json
import threading

def load_matrix(width, bands_height, dir_name):
	'''
	use this function in order to load a sparse matrix that is kept in different small files
	files need to be devided in 3 directories, one containing data, one row indexes and one col indexes
	they have to have sortable names
	'''
	mtrs = []
	for i, fname in enumerate(sorted(os.listdir(dir_name+"/data"))):
	    data = np.load(dir_name+"/data/"+fname)
	    row = np.load(dir_name+"/row/"+fname)
	    col = np.load(dir_name+"/col/"+fname)
	    mtrs.append(sps.csr_matrix((data, (row, col)), shape =(min(bands_height, abs(bands_height*i-width)), width)))
	    print fname

	return(sps.vstack([i for i in mtrs]))

def load_matrix_power(width, bands_height, power, dir_name):
	mtrs = []
	for i, fname in enumerate(sorted(os.listdir(dir_name+"/data"))):
	    data = np.load(dir_name+"/data/"+fname) ** power
	    row = np.load(dir_name+"/row/"+fname)
	    col = np.load(dir_name+"/col/"+fname)
	    mtrs.append(sps.csr_matrix((data, (row, col)), shape =(min(bands_height, abs(bands_height*i-width)), width)))
	    print fname

	return(sps.vstack([i for i in mtrs]))


def load_matrix_T(width, bands_height, dir_name):
	mtrs = {}
	threads = []
	Lock = threading.Lock()
	flist = sorted(os.listdir(dir_name+"/data"))
	n_bands  = len(flist)
	for i, fname in enumerate(flist):
		threads.append(load_matrix_worker(width, bands_height, fname, dir_name, Lock, mtrs, i))
	for i in threads:
		i.start()
	for i in threads:
		i.join()
	return(sps.vstack([mtrs[i] for i in range(n_bands)]))


class load_matrix_worker(threading.Thread):
	def __init__(self, width, bands_height, fname, dir_name, Lock, mtrs, i):
		threading.Thread.__init__(self)
		self.width = width
		self.bands_height = bands_height
		self.fname = fname
		self.Lock = Lock
		self.dir_name = dir_name
		self.i = i
		self.mtrs = mtrs
	def run(self):
		data = np.load(self.dir_name+"/data/"+self.fname)
		row = np.load(self.dir_name+"/row/"+self.fname)
		col = np.load(self.dir_name+"/col/"+self.fname)
		m = sps.csr_matrix((data, (row, col)), shape =(min(self.bands_height, abs(self.bands_height*self.i-self.width)), self.width))
		self.Lock.acquire()
		self.mtrs[self.i]=m
		self.Lock.release()

def load_test_set(dir_name, item_class="track"):
	if(item_class == "track"):
		file_name = "s_id2spt.json"
	elif(item_class == "artist"):
		file_name = "/artists/index_of_artist.txt"	
	with open(dir_name+file_name, "r") as F:
	    item_indexes = json.load(F)
	test_set = {'0':{}, '5':{}, '10':{}, '25':{}, '100':{}, '1':{}}
	with open(dir_name+"challenge_set.json") as F:
	    challenge_struc = json.load(F)
	    for p in challenge_struc['playlists']:
		test_set[str(p['num_samples'])][p['pid']]=[]
		for t in p['tracks']:
		    test_set[str(p['num_samples'])][p['pid']].append(item_indexes[t[item_class+'_uri']])

	return(test_set)

def load_test_set_with_titles(dir_name, item_class):
	names = {}
	if(item_class == "track"):
		file_name = "/tracks/index_of_track_complete.txt"
	elif(item_class == "artist"):
		file_name = "/artists/index_of_artist.txt"	
	with open(dir_name+file_name, "r") as F:
	    item_indexes = json.load(F)
	test_set = {'0':{}, '5':{}, '10':{}, '25':{}, '100':{}, '1':{}}
	names = {'0':{}, '5':{}, '10':{}, '25':{}, '100':{}, '1':{}}
	with open(dir_name+"/ch/challenge_set.json") as F:
	    challenge_struc = json.load(F)
	    for p in challenge_struc['playlists']:
		if('name' in p):
			names[str(p['num_samples'])][p['pid']] = p['name']
		else:
			names[str(p['num_samples'])][p['pid']] = None
		test_set[str(p['num_samples'])][p['pid']]=[]
		for t in p['tracks']:
		    test_set[str(p['num_samples'])][p['pid']].append(item_indexes[t[item_class+'_uri']])


	return(names, test_set)


def load_dense_matrix_T(dir_name):
	K = {}
	threads = []
	flist = sorted(os.listdir(dir_name+"/data"))
	n_bands  = len(flist)
	Lock = threading.Lock()
	for i, file_name in enumerate(flist):
		threads.append(load_dense_matrix_worker(file_name, dir_name, Lock, K, i))
	for i in threads:
		i.start()
	for i in threads:
		i.join()

	ret = []
	for i in range(n_bands):
		ret.append(K[i])
	return ret



class load_dense_matrix_worker(threading.Thread):
	def __init__(self, file_name, dir_name, Lock, K, i):
		threading.Thread.__init__(self)
		self.file_name = file_name
		self.Lock = Lock
		self.dir_name = dir_name
		self.i = i
		self.K = K
	def run(self):
		data = np.load(self.dir_name+"/data/"+self.file_name)
		self.Lock.acquire()
		self.K[self.i]=m
		self.Lock.release()
