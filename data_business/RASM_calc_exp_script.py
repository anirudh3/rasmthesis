import os
import sys
# import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
# import get_related_artists as from_echo
import re
import pickle
from IPython.display import display, HTML
import python_utils as utils
import time


# Load metadata and features.
tracks = pd.read_csv('fma_metadata/tracks.csv')
r_tracks = pd.read_csv('fma_metadata/raw_tracks.csv')
genres = pd.read_csv('fma_metadata/genres.csv')
features = pd.read_csv('fma_metadata/features.csv')
echonest = pd.read_csv('fma_metadata/echonest.csv')
r_artists = pd.read_csv('fma_metadata/raw_artists.csv')

tracks.shape, genres.shape, features.shape, echonest.shape

tracks_sm = tracks.loc[(tracks['set.1'] == str(sys.argv[1])) & (tracks['set'] == str(sys.argv[2]))]

def load_spot_rel_artists(name):
    with open(name, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


print "Total number of artists :", len(tracks_sm['artist.12'].unique())


# load artists
artist_rel = load_spot_rel_artists("saved_data/artist_rel_%s_%s.pickle"%(str(sys.argv[1]), str(sys.argv[2])))

print "Artists in Spotify :", len(artist_rel)

# Subsetting Business, Creating Master tables

# Creating Condensed Artist Map, and Master 2D Table
artist_all = {}

for key in artist_rel:
    artist_all[key] = set()

for key in artist_rel:
    for elem in artist_rel[key]:
        if artist_rel.get(elem, None) is not None:
            artist_all[key].add(elem)
            artist_all[elem].add(key)

# assert len(artist_all) == len(artist_rel)
# deleting artists with no related artist in set
keys = artist_all.keys()
for key in keys:
    if not artist_all[key]:
        del artist_all[key]
#         pass
    else:
        artist_all[key] = list(artist_all[key])

# Artist Index Map
artist_index = {}
for i, a in enumerate(artist_all.keys()):
    artist_index[a] = i

artist_rel_master = np.zeros((len(artist_all.keys()), len(artist_all.keys())))

# Master table of artist relation
for i, a in enumerate(artist_all.keys()):
    # going through each artist
    for b in artist_all[a]:
        artist_rel_master[artist_index[a]][artist_index[b]] = 1

# print artist_rel_master[artist_index['Robo']][artist_index['Robo']]


# Master table of artists per genre

# songs per genre
genres_all = {} # maybe redundant
c = 0
artists_set = list(tracks_sm['artist.12'])
genres_artist = {}
i = 0
for gens in tracks_sm['track.9']:
    l = [re.split(']', s) for s in re.split(', |\[', gens)]
    gens = [item for sublist in l for item in sublist]
    for g in gens:
        if g.isdigit():
            if genres_all.get(int(g), -1) != -1:
                pass
            else:
                genres_all[int(g)] = c
                c += 1
            if genres_artist.get(int(g), -1) == -1:
                genres_artist[int(g)] = set()
            if artist_all.get(artists_set[i], None) is not None:
                genres_artist[int(g)].add(artists_set[i])
    i += 1



# creating master table of artists per genre

genre_artist_master = np.zeros((len(genres_all.keys()), len(artist_all.keys())))

for g in genres_all.keys():
#     print genres_artist[g]
    for a in genres_artist[g]:
#         print a
        genre_artist_master[genres_all[g]][artist_index[a]] = 1

# print genre_artist_master

# Number of songs per artist
songs_per_artist = {}
for k in artist_all.keys():
    songs_per_artist[k] = len(tracks_sm[tracks_sm['artist.12'] == k])



# Calculating D

def calc_D(songs_per_artist, num_artists_set):
    # total number of songs in Set
    num_songs_set = 0
    data = []
    for k in songs_per_artist:
        num_songs_set += songs_per_artist[k]
        if songs_per_artist[k] != 0:
            data.append(songs_per_artist[k])

    data = np.array(data)
    mn = np.mean(data, axis=0)
    sd = np.std(data, axis=0)

    D = 1 - (sd/(mn * (np.sqrt(num_artists_set) - 1)))

    return D


# Calculating C

def calc_C(num_artists_set, genres_artist):
    # total number of genres in Set
    sum_gen = genres_artist.sum(axis=1)
    num_gen_set = np.count_nonzero(sum_gen)

    mn = np.mean(sum_gen, axis=0)
    sd = np.std(sum_gen, axis=0)

    C = 1 - (sd/(mn * (np.sqrt(len(sum_gen)) - 1)))

    return C

# Calculating S

def calc_S(artists_in_set, songs_per_artist):

    num_edges = np.count_nonzero(artists_in_set)/2

    # total number of artists in Set
    num_artists_set = len(songs_per_artist.keys())

    # std dev of artist connections per artist
    a_edges = artists_in_set.sum(axis=1)
    mn = np.mean(a_edges, axis=0)
    sd = np.std(a_edges, axis=0)

    S = 1 - (sd/(mn * (np.sqrt(num_artists_set) - 1)))
    return S

print "Full Set :", len(artist_all)

def main_setup(N):
# Main Loop of Calculation
    import copy
    # N is number of Artists in Set

    global artist_all, genre_artist_master, artist_rel_master, artist_index, songs_per_artist

    # N = 200
    artists_in_set = np.random.choice(artist_all.keys(), N, replace=False)
    artists_not_in_set = artist_all.keys()
    for a in artists_in_set:
        artists_not_in_set.remove(a)

    len(artists_in_set), len(artists_not_in_set)

    # songs per artist subset
    s_p_a = {}
    for a in artists_in_set:
        s_p_a[a] = songs_per_artist[a]

    # artists per genre subset
    num_genres, num_artists = genre_artist_master.shape
    g_a = np.zeros((num_genres, N))
    a_rel = np.zeros((N, N))
    index = 0

    ai = [artist_index[k] for k in artists_in_set]

    for i in range(num_genres):
        g_a[i, :] = genre_artist_master[i, ai]

    for a in artists_in_set:
        ind = artist_index[a]
        a_rel[index, :] = artist_rel_master[ind, ai]
        a_rel[:, index] = artist_rel_master[ai, ind]
        index +=1

    return a_rel, g_a, s_p_a, artists_in_set, artists_not_in_set, ai


def main_run(N):

    global g_a, s_p_a, a_rel, artists_in_set, artists_not_in_set, songs_per_artists, genre_artist_master, artist_rel_master, artist_index, ai

    best_artist = []

    C = calc_C(N, g_a)
    D = calc_D(s_p_a, N)
    S = calc_S(a_rel, s_p_a)
    goodness = C * D * S


    g_list = []
    best_set = []
    g_list_local = []
    g_list_local.append(goodness)
    c_list_local = []
    c_list_local.append(C)
    d_list_local = []
    d_list_local.append(D)
    s_list_local = []
    s_list_local.append(S)

    start_time = time.time()
    
    for epoch in range(3):
        
        
        print epoch, time.time() - start_time
        start_time = time.time()
        
        # main loop for optimization
        for i, a in enumerate(artists_in_set):
            #remove an artist
            s_p_a[a] = 0
            g_a[:, i] = 0
            a_rel[i, :] = 0
            a_rel[:, i] = 0

            change = 0
            local_goodness = - float('inf')
            local_c = 0.
            local_d = 0.
            local_s = 0.
            # add artist from other set one at a time
            for b in artists_not_in_set:

                # add from other set
                s_p_a[b] = songs_per_artist[b]
                ai[i] = artist_index[b]
                g_a[:, i] = genre_artist_master[:, artist_index[b]]
                a_rel[i, :] = artist_rel_master[artist_index[b], ai]
                a_rel[:, i] = artist_rel_master[ai, artist_index[b]]

                Cee = calc_C(N, g_a)
                Dee = calc_D(s_p_a, N)
                Sss = calc_S(a_rel, s_p_a)

                G = Cee * Dee * Sss
                if G > local_goodness:
                    local_goodness = G
                    change = b
                    local_c = Cee
                    local_d = Dee
                    local_s = Sss

                # delete other element
                s_p_a[b] = 0
                g_a[:, i] = 0
                a_rel[i, :] = 0
                a_rel[:, i] = 0

    #         print goodness
            if local_goodness > goodness:
                goodness = local_goodness
                C = local_c
                D = local_d
                S = local_s
                ai[i] = artist_index[change]
                s_p_a[change] = songs_per_artist[change]
                g_a[:, i] = genre_artist_master[:, artist_index[change]]
                a_rel[i, :] = artist_rel_master[artist_index[change], ai]
                a_rel[:, i] = artist_rel_master[ai, artist_index[change]]
                artists_not_in_set.append(a)
                artists_not_in_set.remove(change)
            else:
                ai[i] = artist_index[a]
                s_p_a[a] = songs_per_artist[a]
                g_a[:, i] = genre_artist_master[:, artist_index[a]]
                a_rel[i, :] = artist_rel_master[artist_index[a], ai]
                a_rel[:, i] = artist_rel_master[ai, artist_index[a]]



            g_list_local.append(goodness)
            c_list_local.append(C)
            d_list_local.append(D)
            s_list_local.append(S)


        g_list.append(np.mean(np.array(g_list_local), axis=0))
#         print epoch, g_list_local[-1]

    return g_list_local, c_list_local, d_list_local, s_list_local

def save_best(name, table):
    with open('saved_data/%s.pickle'%name, 'wb') as writer:
        pickle.dump(table, writer, protocol=pickle.HIGHEST_PROTOCOL)


# main run

#to be saved
overall_best_values = []
for i in range(int(sys.argv[3])):
    print i
    list_of_best_values = []
    for n in range(50, len(artist_all), int(sys.argv[4])):
        print "N = ", n
        a_rel, g_a, s_p_a, artists_in_set, artists_not_in_set, ai = main_setup(n)
        v, c, d, s = main_run(n)
        list_of_best_values.append([n, v[-1], c[-1], d[-1], s[-1]])
#         print v[-1], c[-1], d[-1], s[-1]
    overall_best_values.append(list_of_best_values)

save_best("overall_GCSD_%s_%s_%s"%(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])), overall_best_values)


# call arguments -- size, train/test/valid, number of iterations, jump size