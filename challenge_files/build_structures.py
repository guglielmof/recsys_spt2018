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
import matplotlib.pyplot as plt
import numpy.random as npr
import random

slices = os.listdir("../mpd/data")

set_of_tracks = set()
list_of_tracks = []
track_indexes = {}
s2ar ={}
artists_list = []
artists_set =set()
artist_index = {}
playlists_containing_track={}
tracks_in_playlist = {}
val_set = {}
field = "track_uri"
pls_pop = {}

for i, s in enumerate(slices):
    sys.stdout.write("\r{0}>".format(str(i/10)+"%"))
    sys.stdout.flush()
    with open("../mpd/data/"+s) as F:
        slice_struct = json.load(F)
        playlists = slice_struct["playlists"]
        for p in playlists:
            tracks_in_playlist[p['pid']] =[]
            for t in p['tracks']:
                if t[field] not in set_of_tracks:
                    pos = len(set_of_tracks)
                    track_indexes[t[field]]= pos
  
                    set_of_tracks.add(t[field])
                    list_of_tracks.append(t[field])
                    playlists_containing_track[pos] =[]
                tracks_in_playlist[p['pid']].append(track_indexes[t[field]])
                playlists_containing_track[track_indexes[t[field]]].append(p['pid'])



json.dump(tracks_in_playlist, open("./p2s.json", "w"))
json.dump(list_of_tracks, open("./s_id2spt.json", "w"))
json.dump(playlists_containing_track, open("./s2p.json", "w"))
