#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:14:25 2018

@author: anirudhmani
"""

# Exploring data


import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import get_related_artists as from_echo
import re
import pickle


#import utils


# Load metadata and features.
tracks = pd.read_csv('fma_metadata/tracks.csv')
r_tracks = pd.read_csv('fma_metadata/raw_tracks.csv')
genres = pd.read_csv('fma_metadata/genres.csv')
features = pd.read_csv('fma_metadata/features.csv')
echonest = pd.read_csv('fma_metadata/echonest.csv')
r_artists = pd.read_csv('fma_metadata/raw_artists.csv')

#print list(echonest.columns.values)


temp = r_artists['artist_name'].head(n=1000)


# get related artists for every artist
na = 0
artist_rel = {}

for i, elem in enumerate(temp):
    
    artist = re.split(',', elem)[0]
    
    a_id = from_echo.get_artist(artist)
    
    if a_id:
        related_artists = from_echo.get_related(a_id['uri'])
#        print related_artists
        if related_artists:
            artist_rel[artist] = related_artists
        else:
            na += 1
    else:
        na += 1
    

# saving the artist_rel table

with open('artist_rel.pickle', 'wb') as writer:
    pickle.dump(artist_rel, writer, protocol=pickle.HIGHEST_PROTOCOL)
        
            
# loading artist_rel
with open('artist_rel.pickle', 'rb') as handle:
    artist_rel = pickle.load(handle)
    

# get final mapping
artist_all = {}

for key in artist_rel:
    artist_all[key] = []
    
    
for key in artist_rel:
    for elem in artist_rel[key]:
        if elem in artist_rel:
            artist_all[key].append(elem)
            artist_all[elem].append(key)
            



    
    
    


#related = get_related(get_artist(temp[17])['uri'])

#print type(temp[5])



