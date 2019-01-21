# Evaluation Baseline using Mixed Feature Stack

## Import
from sklearn.cluster import KMeans
import collections, numpy
from scipy import spatial
import numpy as np
import librosa
import math
import re
import os, sys
import pandas as pd
import IPython.display as ipd
import pickle


# Load metadata and features.
tracks = pd.read_csv('fma_metadata/tracks.csv')
r_tracks = pd.read_csv('fma_metadata/raw_tracks.csv')
genres = pd.read_csv('fma_metadata/genres.csv')
features = pd.read_csv('fma_metadata/features.csv')
echonest = pd.read_csv('fma_metadata/echonest.csv')
r_artists = pd.read_csv('fma_metadata/raw_artists.csv')



tracks_sm = tracks.loc[(tracks['set.1'] == 'small') & (tracks['set'] == 'training')]


# load artists Full Set
def load_spot_rel_artists(name):
    with open(name, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

artist_rel = load_spot_rel_artists("saved_data/artist_rel_small_training.pickle")

# Creating Condensed Artist Map, and Master 2D Table
artist_fs = {} # artists in Full Set

for key in artist_rel:
    artist_fs[key] = set()

for key in artist_rel:
    for elem in artist_rel[key]:
        if artist_rel.get(elem, None) is not None:
            artist_fs[key].add(elem)
            artist_fs[elem].add(key)

# assert len(artist_all) == len(artist_rel)
# deleting artists with no related artist in set
keys = artist_fs.keys()
for key in keys:
    if not artist_fs[key]:
        del artist_fs[key]
#         pass
    else:
        artist_fs[key] = list(artist_fs[key])

# Artist Index Map
artist_index = {}
for i, a in enumerate(artist_fs.keys()):
    artist_index[a] = i


f1 = features['feature']
f2 = features[['spectral_centroid','spectral_centroid.1', 'spectral_centroid.2', 'spectral_centroid.3', 'spectral_centroid.4', 'spectral_centroid.5', 'spectral_centroid.6']]
f3 = features.iloc[:, 512:]
fs = pd.concat([f1, f2], axis=1)
fs = pd.concat([fs, f3], axis=1)

fs = fs.loc[fs['feature'].isin(tracks_sm['Unnamed: 0'])]

# ipd.display(fs)

overall_fs = fs.values[:, 1:]


def get_bog_model(data, num_clusters):
    # all features in all songs in all artists
    # X = get_data
#     A = 100
#     F = 13
#     all_features = numpy.random.rand(A, F)
    model = KMeans(num_clusters)
    model.fit(data)
    return model

model = get_bog_model(overall_fs, 100) # going for 100 clusters


def save_things(name, table):
    with open('saved_data/%s.pickle'%name, 'wb') as writer:
        pickle.dump(table, writer, protocol=pickle.HIGHEST_PROTOCOL)

save_things('cluster_model', model)
