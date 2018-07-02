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
import os
import json
import numpy as np
import sys
#import sklearn
#import numpy.random as npr
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
import random
from Engine import RecEngine
import utilscvx as utc
import cvxopt as co
import cvxopt.solvers as solver
#from nltk.stem.porter import *
import string
import math
import utils
import time
import scipy.sparse as sps

class Random(RecEngine):
    def __init__(self, track_list, track_indexes, playlists_containing_track, tracks_in_playlist):
        super(self.__class__, self).__init__(track_list, track_indexes, playlists_containing_track, tracks_in_playlist)
        self.rand = None

    def train(self, test_playlists):
        self.rand = {}

        for key in test_playlists:
            self.rand[key]=[]
            #generate 750 all different values between 0 and the number of tracks
            generated_tracks = random.sample(range(self.n_items), 750)
            #count first 500 values that are not in the seed
            for j in generated_tracks:
                if j not in test_playlists[key]:
                    self.rand[key].append(j)
                if(len(self.rand[key])==500):
                    break
            
        return self

    def get_recommendation(self, pid):
        return self.rand[pid]


class Popular(RecEngine):
    def __init__(self, track_list, track_indexes, playlists_containing_track, tracks_in_playlist):
        super(self.__class__, self).__init__(track_list, track_indexes, playlists_containing_track, tracks_in_playlist)
        self.popular = None

    def train(self, test_playlists):
        #calculate the first 750 popular songs
        popularity = np.zeros(self.n_items)
        #popularity = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for x in range(len(self.items_list)):
            popularity[x] = len(self.users_to_item[str(x)])

        
        pop_sorted  = sorted(range(len(popularity)), key=lambda k: -popularity[k])
        pop_sorted = pop_sorted[0:750]
        
        self.popular = {}
        for key in test_playlists:
            self.popular[key]=[]
            for j in pop_sorted:
                if j not in test_playlists[key]:
                    self.popular[key].append(j)
                if(len(self.popular[key])==500):
                    break
        
        return self

    def get_recommendation(self, pid):
        return self.popular[pid]
    
class KNN(RecEngine):
    #items_list contains the list of tracks, to find the track_uri associated with a specific integer
    #items_index is the opposite of items_list and it's a dictionary where the item uri (track) is linked to an int
    #users_to_item contains the lists of users (playlists) assiciated with each item (track)
    #items_to_user contains the lists of items (tracks) associated with each user (playlist) (tracks/artists contained in each playlist)
    def __init__(self, items_list, items_index, users_to_item, items_to_user):
        super(self.__class__, self).__init__(items_list, items_index, users_to_item, items_to_user)
        self.KNN = None
        #popularity is necessary when i'm not able to recommend enough tracks
        popularity = np.zeros(len(items_list))
        for x in range(len(items_list)):
            popularity[x] = len(users_to_item[str(x)])		
        self.pop_sorted  = sorted(range(len(popularity)), key=lambda k: -popularity[k])[0:750]
        print("Model initialized")

    def train(self, test_playlists, K, directory, len_seed):
        #directory is the path to the directory where similarity files are; each file should be a csv where the first column
        #contains the id of the test playlist, meanwhile the remaining columns contain a tuple for each of the 2000 most similar playlist
        #with (id, similarity) already ordered by similarity
        
        most_similar_playlists = {}
        self.knn = {}
        
        k = 0
        #first of all, is necessary to build the list of K nearest neighbors 
        for file_name in os.listdir(directory):
            with  open(directory+"/"+file_name, 'r') as F:
                #read a single line
                for line in F:
                    row = line.rstrip().split(';')
                    pid = row.pop(0)
                    most_similar_playlists[pid] =[]
                    if len(row)>1:
                        
                        most_similar_playlists[pid] = list(map(eval, row[0:K]))
                    k+=1
                print("file "+file_name+" processed")

        tot_time = len(most_similar_playlists.keys())
        #now, for each playlist, is necessary to calculate score
        for act_time, key in enumerate(most_similar_playlists.keys()): #in key, there will be all pids of test playlist
            sys.stdout.write("\r{0}>".format(str(act_time*100/tot_time)+"%"))
            sys.stdout.flush()
            self.knn[key] = []
            scores = np.zeros(len(self.items_list))
            #t1 = time.time()
            for playlist, score in most_similar_playlists[key]:
                    scores[self.items_to_user[str(playlist)]]+=score
            #print(time.time() -t1)
            #get top 750 scores
            s = np.argpartition(-scores, 750)[:750]
            #t1 = time.time()
            for i in s:
                if scores[i] == 0: #if scores from now on are 0, it means the song wasn't in any similar playlist, so 
                                    # we are going to use popularity for the last tracks.
                    for j in self.pop_sorted:
                        if j not in test_playlists[key][0:len_seed] and j not in self.knn[key]:
                            self.knn[key].append(j)
                            if(len(self.knn[key])==500):
                                break
                    break;
                elif i not in test_playlists[key][0:len_seed]:
                    self.knn[key].append(i)
                    if(len(self.knn[key])==500):
                        break
            #print(time.time() -t1)
        return self

    def get_recommendation(self, pid):
        return self.knn[pid]

class CF_KOMD:
    def __init__(self, q, users_to_item=None, K=None, K_indexing = None, inverse_indexing = None,
                lp=0.1, spr=False, verbose=True, items_list = None):
        #super(CF_KOMD, self).__init__(data, verbose)
        self.lambda_p = lp
        self.q_=co.matrix(q, (len(K_indexing), 1))
        self.verbose = verbose
        self.items_list = items_list
        self.inverse_indexing = inverse_indexing

        if K is not None:
            self.K = K
            self.K_indexing  = K_indexing
        else:
            self.users_to_item = users_to_item
            for i in self.users_to_item:
                users_to_item[i] = set(users_to_item[i])
        '''
        self.K = K
        self.q_ = co.matrix(0.0, (self.n_items, 1))
        for i in xrange(self.n_items):
            self.q_[i,0] = sum(self.K[i,:]) / float(self.n_items) #-1
        '''
        self.model = None
        self.test_users = None
        #self.sol = {} #TODO

    def train(self, test_users=None, b=None):
        self.model = {}
        self.test_users = test_users
        
        if test_users is None:
            test_users = range(self.n_users)
        
        tu={}
        if b is not None:
            for u in test_users:
                tu[u]= set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u]= set(test_users[u])
        
        for i, u in enumerate(tu):
            if self.verbose and (i+1) % 100 == 0:
                print("%d/%d" %(i+1, len(test_users)))

            #Xp = list(self.data.get_items(u)) #get list of songs 
            Xp = [int(self.K_indexing[str(idx)]) for idx in tu[u]]
            
            npos = len(Xp)
            
            #for each element in Xp, you could either calculate the similarity or take it from the matrix
            '''
            directly calculate them
            kp = np.zeros((len(Xp), len(Xp))
            for i in Xp:
                for j in Xp:
                    kp[i, j] = (playlists_containing_track[i] & playlists_containing_track[j] / 
                    math.sqrt(len(playlists_containing_track[i])*len(playlist_containing_track[j])))
            '''
            '''
                take them from the matrix
                
            '''
            kp = np.zeros((npos, npos))
        
            for j in range(npos):
                #print(self.K[Xp[i], Xp[i]])
                #print(self.K[Xp[i],Xp])
                kp[j] = self.K[Xp[j],Xp].todense()
        
            kp = co.matrix(kp)
            
            kn = self.q_[Xp, :]
                        
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
            
            self.model[u] = (self.K[Xp,:].T).dot(sol['x']) - self.q_
            #print self.model[u]

        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):
        i = -self.model[u].T[0]
        s = np.argpartition(i, 600)[:600]
        s = s[np.argsort(i[s])]
        j = 0
        rac = []
        while(len(rac)<500):
            if(int(self.inverse_indexing[s[j]]) not in self.test_users[u][:b]):
                	rac.append(self.items_list[int(self.inverse_indexing[s[j]])])
            j+=1
        return rac

class Title_based:

    def __init__(self, W, words_list, word2items, items_list, users_to_item):
        self.W = W
        self.words_list = words_list
        self.items_list = items_list
        self.word2items = word2items
        self.model = {}
        self.stemmer = PorterStemmer()
        
        popularity = np.zeros(len(items_list))
        for x in range(len(items_list)):
            popularity[x] = len(users_to_item[str(x)])		
        pop_sorted  = sorted(range(len(popularity)), key=lambda k: -popularity[k])[0:600]
        self.default = [items_list[i] for i in pop_sorted]

        self.inverse_dic ={}
        k = 0
        for i in self.words_list:
            self.inverse_dic[i] = 0
            k+=1
        
        print("model initialized")
        
    def train (self, test_titles, test_users, b, name):
        unable_to_train = 0
        for u in test_titles:
            K = False
            scores = np.zeros(len(self.items_list))
            for t in test_titles[u].split(" "):
                if t:
                    term = self.stemmer.stem(''.join([i for i in t.lower() if i not in string.punctuation]))
                    print(term in self.words_list)
                    
                try:
                    title_pos = self.inverse_dic[term]
                except:
                    title_pos = -1
                if title_pos>-1:
                    K = True
                    for w in range(len(self.words_list)):
                        scores[self.word2items[term]] += self.W[title_pos, w]
            if K:
                s = np.argpartition(-scores, 600)[:600]
                s = s[np.argsort(-scores[s])]
                self.model[u] = [self.items_list[i] for i in s if self.items_list[i] not in test_users[u][:b]][0:500]
                
            else:
                self.model[u] = [i for i in self.default if i not in test_users[u][:b]][0:500] 
                unable_to_train += 1
            
        if unable_to_train>0:
            print("unable to find "+str(unable_to_train)+" titles")
        
        print(self.model)
        with open("../files/word_rec_"+str(name)+".txt", "w") as F:
            json.dump(self.model, F)
        return self
                
    def get_recommendation(self, pid):
            return self.model[pid]	

class Word_popularity_borda(RecEngine):
    #items_list contains the list of tracks, to find the track_uri associated with a specific integer
    #items_index is the opposite of items_list and it's a dictionary where the item uri (track) is linked to an int
    #users_to_item contains the lists of users (playlists) assiciated with each item (track)
    #items_to_user contains the lists of items (tracks) associated with each user (playlist) (tracks/artists contained in each playlist)
    def __init__(self, items_list, items_index, users_to_item, items_to_user, word2items):
        super(self.__class__, self).__init__(items_list, items_index, users_to_item, items_to_user)
        self.recommendations = None
        self.word2items = word2items
        #popularity is necessary when i'm not able to recommend enough tracks
        popularity = np.zeros(len(items_list))
        for x in range(len(items_list)):
            popularity[x] = len(users_to_item[str(x)])		
        pop_sorted  = sorted(range(len(popularity)), key=lambda k: popularity[k])
        self.borda_pop_scores = np.zeros(len(pop_sorted))
        for i, x in enumerate(pop_sorted):
            self.borda_pop_scores[x] = float(i)/float(len(pop_sorted))

        print("Model initialized")

    def train(self, test_users, test_titles, word2users):
        for u in test_users:
            for j in test_titles[u].split(" "):
                term = self.stemmer.stem(''.join([i for i in t.lower() if i not in string.punctuation]))
                pop_limited = np.zeros(len(self.items_list))
                #per ogni playlist t in W[term], per ogni canzone nella playlist t, aggiungi 1 alla posizione della canzone in pop limited
                for pl in word2users[term]:
                    for tr in items_to_user[pl]:
                        pop_limited[tr] += 1
                pop_limited = pop_limited/np.max(pop_limited)
                base += pop_limited
            ordina_base [0:600]

        return self

    def get_recommendation(self, pid):
        return self.recommendations[pid]	

class CF_KOMDisj:
    def __init__(self, q, users_to_item=None, K=None, K_indexing = None, N = None, imp_dic=None, inverse_indexing = None,
                lp=0.1, spr=False, verbose=True, items_list = None):
        self.lambda_p = lp
        self.q_ = co.matrix(q, (len(K_indexing), 1))
        self.verbose = verbose
        self.items_list = items_list
        self.inverse_indexing = inverse_indexing

        self.K = K

        self.K_indexing  = K_indexing
        self.N = float(N)
        
        self.model = None
        self.test_users = None
        self.u2i = users_to_item

        self.imp_dic = imp_dic#{(j,i):self.imputation(j,i,self.N) for i in range(1,251) for j in range(i,251)}
        #self.sol = {} #TODO

    def imputation(self, x, z, N):
        if (x,z) not in self.imp_dic:
            sx = math.sqrt(self.identity_disjunctive(N, x))
            sz = math.sqrt(self.identity_disjunctive(N, z))
            self.imp_dic[(x,z)] = (2. / (sx*sz)) * (x / N) * (z / (N-1.))
        return self.imp_dic[(x,z)]
        
    def reduced_binom(self, N, k, d=2):
        b = 1.
        for i in range(d):
            b *= float(k-i)/(N-i)
        return b
        
    def identity_disjunctive(self, N, x, d=2):
        return 1. - self.reduced_binom(N, N-x, d)
    
    def train(self, test_users=None, b=None):
        t = time.time()
        self.model = {}
        self.test_users = test_users
        
        if test_users is None:
            test_users = range(self.n_users)
        
        tu = {}
        if b is not None:
            for u in test_users:
                tu[u] = set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u] = set(test_users[u])
        
        for i, u in enumerate(tu):
            if self.verbose and (i+1) % 10 == 0:
                print("%d/%d" %(i+1, len(test_users)))
                print(str(time.time()-t))
                utils.fl()

            Xp = [int(self.K_indexing[str(idx)]) for idx in tu[u]]
                
            npos = len(Xp)
                
            kp = np.zeros((npos, npos))
        
            for j in range(npos):
                kp[j] = self.K[Xp[j],Xp].todense()
            
            '''
            for row in range(kp.shape[0]):
                x = len(self.u2i[int(self.inverse_indexing[Xp[row]])])
                sx = math.sqrt(self.identity_disjunctive(self.N, x))
                xN = (x / N)
                for col in range(row+1, kp.shape[0]):
                    if kp[row,col] == 0.:
                        y = len(self.u2i[int(self.inverse_indexing[Xp[col]])])
                        sz = math.sqrt(self.identity_disjunctive(self.N, z))
                        kp[row,col] = (2. / (sx*sz)) * xN * (z / (N-1.))
                        kp[row,col] = kp[col,row]
            '''

            for row in range(kp.shape[0]):
                x = len(self.u2i[self.inverse_indexing[Xp[row]]])
                for col in range(row+1, kp.shape[0]):
                    if kp[row,col] == 0.:
                        z = len(self.u2i[self.inverse_indexing[Xp[col]]])
                        mx,mn = (x,z) if x >= z else (z,x)
                        kp[row,col] = self.imp_dic["%d,%d" %(mx,mn)]
                        kp[col,row] = kp[row,col]

            kp = co.matrix(kp)
        
            kn = self.q_[Xp, :]
            
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)

            K_imp = co.matrix(self.K[Xp,:].todense())
    	    for row in range(K_imp.size[0]):
    	    	x = len(self.u2i[self.inverse_indexing[Xp[row]]])
    		for col in range(K_imp.size[1]):
    			if K_imp[row,col] == 0.:
    				z = len(self.u2i[self.inverse_indexing[col]])
    				mx,mn = (x,z) if x >= z else (z,x)
    				K_imp[row,col] = self.imp_dic["%d,%d" %(mx,mn)]

            
            self.model[u] = (K_imp.T*sol['x']) - self.q_
        #import numpy.random as rnd
        #with open("../files/"+rnd.randint()+".txt") as F:
        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p, "d" : 2}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):

        scores = np.array(-(self.model[u].T))[0]

        s = np.argpartition(scores, 600)[:600]

        s = s[np.argsort(scores[s])]
        j = 0
        rac = []
        while(len(rac) < 500):
            if(int(self.inverse_indexing[s[j]]) not in self.test_users[u][:b]):
                rac.append(self.items_list[int(self.inverse_indexing[s[j]])])
            j+=1
        return rac


class CF_KOMD_full_mtrx:
    def __init__(self, q, users_to_item=None, K=None, K_indexing = None, inverse_indexing = None,
                lp=0.1, spr=False, verbose=True, items_list = None):
        #super(CF_KOMD, self).__init__(data, verbose)
        self.lambda_p = lp
        self.q_=co.matrix(q, (len(K_indexing), 1))
        self.verbose = verbose
        self.items_list = items_list
        self.inverse_indexing = inverse_indexing

        if K is not None:
            self.K = K
            self.K_indexing  = K_indexing
        else:
            self.users_to_item = users_to_item
            for i in self.users_to_item:
                users_to_item[i] = set(users_to_item[i])
        '''
        self.K = K
        self.q_ = co.matrix(0.0, (self.n_items, 1))
        for i in xrange(self.n_items):
            self.q_[i,0] = sum(self.K[i,:]) / float(self.n_items) #-1
        '''
        self.model = None
        self.test_users = None
        #self.sol = {} #TODO

    def train(self, test_users=None, b=None):
        block_size = 5000
        self.model = {}
        self.test_users = test_users
        
        if test_users is None:
            test_users = range(self.n_users)
        
        tu={}
        if b is not None:
            for u in test_users:
                tu[u]= set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u]= set(test_users[u])
        
        for i, u in enumerate(tu):
            if self.verbose and (i+1) % 100 == 0:
                print("%d/%d" %(i+1, len(test_users)))
                utils.fl()

            #Xp = list(self.data.get_items(u)) #get list of songs 
            Xp = [int(self.K_indexing[str(idx)]) for idx in tu[u]]
            mtx_indexes = [a/block_size for a in Xp]
            pos_indexes = [a%block_size for a in Xp]
            
            npos = len(Xp)

            kp = np.zeros((npos, npos))
        
            for j in range(npos):
                #kp[j] = self.K[Xp[j],Xp].todense()
                kp[j] = self.K[mtx_indexes[j]][pos_indexes[j], Xp]
        
            kp = co.matrix(kp)
            
            kn = self.q_[Xp, :]
                        
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
            
            band = np.zeros((npos, len(self.inverse_indexing)))
            for j in range(npos):
                band[j]= self.K[mtx_indexes[j]][pos_indexes[j], :]
            
            self.model[u] = (co.matrix(band.T)*(sol['x'])) - self.q_
            #print self.model[u]

        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return np.array(-(self.model[u].T))[0]

    def get_recommendation (self, u, b):
        i = np.array(-(self.model[u].T))[0]
        s = np.argpartition(i, 600)[:600]
        s = s[np.argsort(i[s])]
        j = 0
        rac = []
        while(len(rac)<500):
            if(int(self.inverse_indexing[s[j]]) not in self.test_users[u][:b]):
                    rac.append(self.items_list[int(self.inverse_indexing[s[j]])])
            j+=1
        return rac
    
    

class CF_KOMD_saa:
    def __init__(self, mu=None, qlist=None, Klist=None, s2k=None, k2s=None, ar2k=None, al2k=None, s2ar=None, s2al=None, lp=None, verbose=True, songs_list=None):
        
        self.lambda_p = lp
        self.q_ = co.matrix(sum([qlist[i] * mu[i] for i in range(len(mu))])).T
        self.verbose = verbose
        
        self.songs_list = songs_list
        
        self.mu = mu
        self.k2s = k2s
        self.s2k = s2k
        self.s2ar = s2ar
        self.s2al = s2al
        self.ar2k = ar2k
        self.al2k = al2k
        self.Klist = Klist
        
        self.k2ar = [self.ar2k[self.s2ar[str(s)]] for s in self.k2s]
        self.k2al = [self.al2k[self.s2al[str(s)]] for s in self.k2s]
        
        self.model = None
        self.test_users = None

    
    def train(self, test_users=None, bucket=None):
        t = time.time()
        self.model = {}
        
        Kar = self.Klist[1].todense()
        Kal = self.Klist[2].todense()
        
        self.test_users = test_users
        if test_users is None:
            self.test_users = range(self.n_users)
        
        if bucket is not None:
            tu = {u:set(self.test_users[u][:bucket]) for u in self.test_users}
        else:
            tu = {u:set(self.test_users[u]) for u in self.test_users}        
        
        for i, u in enumerate(tu):
            if self.verbose and (i+1) % 10 == 0:
                print("%d/%d" %(i+1, len(self.test_users)))
                print(str(time.time()-t))
                utils.fl()

            Xp = [int(self.s2k[idx]) for idx in tu[u]]
            idxar = [self.k2ar[p] for p in Xp]   
            idxal = [self.k2al[p] for p in Xp]
            npos = len(Xp)
                
            kp = np.zeros((npos, npos))
            for j in range(npos):
                kp[j] = self.mu[0] * self.Klist[0][Xp[j], Xp].todense().reshape((npos,))
                kp[j] = kp[j] + self.mu[1] * Kar[idxar[j], idxar].reshape((npos,))
                kp[j] = kp[j] + self.mu[2] * Kal[idxal[j], idxal].reshape((npos,))

            kp = co.matrix(kp)
            kn = self.q_[Xp, :]
            
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
	   
            K = co.matrix(self.mu[0] * self.Klist[0][Xp, :].todense())
            
	    idxar = np.array(idxar).reshape((len(idxar),1))
	    idxal = np.array(idxal).reshape((len(idxal),1))
	        
	    K = K + co.matrix(self.mu[1] * Kar[idxar, self.k2ar])
            K = K + co.matrix(self.mu[2] * Kal[idxal, self.k2al])
                
            self.model[u] = (K.T * sol['x']) - self.q_

        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):

        scores = np.array(-(self.model[u].T))[0]
        s = np.argpartition(scores, 600)[:600]
        s = s[np.argsort(scores[s])]
        j = 0
        rac = []
        while(len(rac) < 500):
            if(int(self.k2s[s[j]]) not in self.test_users[u][:b]):
                rac.append(self.songs_list[int(self.k2s[s[j]])])
            j+=1
	
	return rac

class CF_KOMD_all_songs:
    def __init__(self, q, users_to_item=None, K=None,
                lp=0.1, spr=False, verbose=True, items_list = None):
        #super(CF_KOMD, self).__init__(data, verbose)
        self.lambda_p = lp
        self.q_=co.matrix(q, (K.shape[0], 1))
        self.verbose = verbose
        self.items_list = items_list

        if K is not None:
            self.K = K
        else:
            self.users_to_item = users_to_item
            for i in self.users_to_item:
                users_to_item[i] = set(users_to_item[i])

        self.model = None
        self.test_users = None


    def train(self, test_users=None, b=None):
        self.model = {}
        self.test_users = test_users
        
        if test_users is None:
            test_users = range(self.n_users)
        
        tu={}
        if b is not None:
            for u in test_users:
                tu[u]= set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u]= set(test_users[u])
                
        for i, u in enumerate(tu):
            if self.verbose and (i+1) % 100 == 0:
                print("%d/%d" %(i+1, len(test_users)))
                utils.fl()

            Xp = [idx for idx in tu[u]]
            
            npos = len(Xp)


            kp = np.zeros((npos, npos))
        
            for j in range(npos):
                kp[j] = self.K[Xp[j],Xp].todense()
        
            kp = co.matrix(kp)
            
            kn = self.q_[Xp, :]
                        
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
            
            self.model[u] = (self.K[Xp,:].T).dot(sol['x']) - self.q_
            #print self.model[u]

        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):
        s = utils.scores_sorter(self.model[u].T[0])
        rac = []
        j = 0
        while(len(rac)<500):
            if(s[j] not in self.test_users[u][:b]):
                    rac.append(self.items_list[s[j]])
            j+=1
        return rac

class CF_KOMD_selected:
    def __init__(self, q, users_to_item=None, K=None,
                lp=0.1, spr=False, verbose=True, items_list = None):
        #super(CF_KOMD, self).__init__(data, verbose)
        self.lambda_p = lp
        self.q_=co.matrix(q, (q.shape[0], 1))
        self.verbose = verbose
        self.items_list = items_list

        if K is not None:
            self.K = K
        else:
            self.users_to_item = users_to_item
            for i in self.users_to_item:
                users_to_item[i] = set(users_to_item[i])

        self.model = None
        self.test_users = None


    def train(self, test_users=None, b=None, selection=None):
        self.model = {}
        self.test_users = test_users
        
        if test_users is None:
            test_users = range(self.n_users)
        
        tu={}
        if b is not None:
            for u in test_users:
                tu[u]= set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u]= set(test_users[u])
                
        for i, u in enumerate(tu):
            if self.verbose and (i+1) % 100 == 0:
                print("%d/%d" %(i+1, len(test_users)))
                utils.fl()

            Xp = [idx for idx in tu[u]]
            
            npos = len(Xp)


            kp = np.zeros((npos, npos))
        
            for j in range(npos):
                kp[j] = self.K[Xp[j],Xp].todense()
        
            kp = co.matrix(kp)
            
            kn = self.q_[Xp, :]
                        
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
            #print(selection[u][0:5])
            #utils.fl()
            tmp = (self.K[np.array(Xp)[:,None],selection[u]].T).dot(sol['x']) - self.q_[[a for a in selection[u]]]

            self.model[u] = np.zeros(len(self.items_list))-np.inf
            #print(tmp.shape)
            self.model[u][selection[u]] = tmp[:,0]
            #for i, v in enumerate(selection[u]):
            #    self.model[u][v] = tmp[i]
            #print self.model[u]

        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):
        s = utils.scores_sorter(self.model[u])
        rac = []
        j = 0
        while(len(rac)<500):
            if(s[j] not in self.test_users[u][:b]):
                    rac.append(self.items_list[s[j]])
            j+=1
        return rac

        def get_selection (self, u, b, k):
        	s = utils.scores_sorter(self.model[u], k+b)
        	rac = [i for i in s if i not in self.test_users[u][:b]][:k]
        	return rac

class CF_KOMeD:
    def __init__(self, q, users_to_item=None, K=None,
                lp=0.1, spr=False, verbose=True, items_list = None, Beta=None, Beta_K = None):
        #super(CF_KOMD, self).__init__(data, verbose)
        self.lambda_p = lp
        self.q_=co.matrix(q, (K.shape[0], 1))
        self.verbose = verbose
        self.items_list = items_list
        self.Beta = Beta
        self.Beta_K = Beta_K
        if K is not None:
            self.K = K
        else:
            self.users_to_item = users_to_item
            for i in self.users_to_item:
                users_to_item[i] = set(users_to_item[i])

        self.model = None
        self.test_users = None


    def train(self, test_users=None, b=None, test_titles=None):
        self.model = {}
        self.test_users = test_users

        if test_users is None:
            test_users = range(self.n_users)
        
        tu={}
        if b is not None:
            for u in test_users:
                tu[u]= set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u]= set(test_users[u])
                
        for i, u in enumerate(tu):
            
            beta = self.Beta[test_titles[u]]
            beta_K = self.K.dot(beta.T).todense()
            if self.verbose and (i+1) % 100 == 0:
                print("%d/%d" %(i+1, len(test_users)))
                utils.fl()

            Xp = np.array([idx for idx in tu[u]])

            npos = len(Xp)+1
            kp = np.zeros((npos, npos))
            #print(self.K[Xp[:,None],Xp].todense())
            kp[:npos-1, :npos-1] = self.K[Xp[:,None],Xp].todense()

            kp[npos-1, :] = np.concatenate((self.K[Xp,:].dot(beta.T).todense(), np.ones((1,1)))).flatten()
            kp[:npos-1, npos-1] = kp[npos-1, :npos-1]
            print(kp)
            

            q0 = (co.matrix(beta.todense())*self.q_)[0,0]
            kp = co.matrix(kp)

            kn = co.matrix(np.concatenate((np.array(self.q_[list(Xp), :]), np.array([[q0]]))))
                        
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = -utc.identity(npos)
            h = utc.zeroes_vec(npos)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
            
            self.model[u] = np.array((self.K[Xp,:].T).dot(sol['x'][:-1]) + beta_K*sol['x'][-1]  - self.q_).flatten()
            #print self.model[u]

        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):
        s = utils.scores_sorter(self.model[u])
        rac = []
        j = 0
        while(len(rac)<500):
            if(s[j] not in self.test_users[u][:b]):
                    rac.append(self.items_list[s[j]])
            j+=1
        return rac

class CF_KOMeD_c:
    def __init__(self, q, users_to_item=None, K=None,
                lp=0.1, spr=False, verbose=True, items_list = None, Beta=None, Beta_K = None):
        #super(CF_KOMD, self).__init__(data, verbose)
        self.lambda_p = lp
        self.q_=co.matrix(q, (K.shape[0], 1))
        self.verbose = verbose
        self.items_list = items_list
        self.Beta = Beta
        self.Beta_K = Beta_K
        if K is not None:
            self.K = K
        else:
            self.users_to_item = users_to_item
            for i in self.users_to_item:
                users_to_item[i] = set(users_to_item[i])

        self.model = None
        self.test_users = None


    def train(self, test_users=None, b=None, test_titles=None, c = 0):
        self.model = {}
        self.test_users = test_users
        self.c = c
        if test_users is None:
            test_users = range(self.n_users)
        
        tu={}
        if b is not None:
            for u in test_users:
                tu[u]= set(test_users[u][:b])
        else:
            for u in test_users:
                tu[u]= set(test_users[u])
                
        for i, u in enumerate(tu):
            t = time.time()
            beta = self.Beta[test_titles[u]]
            beta_K = self.K.dot(beta.T).todense()
            if self.verbose and (i+1) % 1 == 0:
                print("%d/%d" %(i+1, len(test_users)))
                utils.fl()

            Xp = np.array([idx for idx in tu[u]])

            npos = len(Xp)+1
            kp = np.zeros((npos, npos))
            #print(self.K[Xp[:,None],Xp].todense())
            kp[:npos-1, :npos-1] = self.K[Xp[:,None],Xp].todense()

            kp[npos-1, :] = np.concatenate((self.K[Xp,:].dot(beta.T).todense(), np.ones((1,1)))).flatten()
            kp[:npos-1, npos-1] = kp[npos-1, :npos-1]
            

            q0 = (co.matrix(beta.todense())*self.q_)[0,0]
            kp = co.matrix(kp)

            kn = co.matrix(np.concatenate((np.array(self.q_[list(Xp), :]), np.array([[q0]]))))
                        
            I = self.lambda_p * utc.identity(npos)
            P = kp + I
            q = -kn
            G = np.vstack((-np.eye(npos), np.zeros(npos)))
            G[npos, -1] = 1
            G = co.matrix(G)
            h = np.zeros(npos+1)
            h[-1] = self.c
            h = co.matrix(h)
            A = utc.ones_vec(npos).T
            b = co.matrix(1.0)

            solver.options['show_progress'] = False
            sol = solver.qp(P, q, G, h, A, b)
            
            self.model[u] = np.array((self.K[Xp,:].T).dot(sol['x'][:-1]) + beta_K*sol['x'][-1]  - self.q_).flatten()
            #print self.model[u]
            print(time.time()-t)
        # endfor
        return self

    def get_params(self):
        return {"lambda_p" : self.lambda_p}

    def get_scores(self, u):
        return self.model[u]

    def get_recommendation (self, u, b):
        s = utils.scores_sorter(self.model[u])
        rac = []
        j = 0
        while(len(rac)<500):
            if(s[j] not in self.test_users[u][:b]):
                    rac.append(self.items_list[s[j]])
            j+=1
        return rac
