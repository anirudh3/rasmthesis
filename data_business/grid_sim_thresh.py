# Grid Search for Similarity Distance Threshold

## Import
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import collections, numpy
from scipy import spatial
import numpy as np
import librosa
import math
import re
import os, sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE





def get_bog_model(data, num_clusters):
    # all features in all songs in all artists
    model = KMeans(num_clusters)
    model.fit(data)
    return model


# load artists Full Set
def load_spot_rel_artists(name):
    with open(name, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def get_artists():

    artist_rel = load_spot_rel_artists("/Users/anirudhmani/Prog/thesis_related/data_business/saved_data/artist_rel_small_training.pickle")

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


    return artist_rel, artist_fs, artist_index


# Input is num_clips * feature_dimension
# Output is 1 * num_clusters
def get_histogram_song(song_features, model, num_clusters):
    clusters = model.predict(song_features)
    Y = numpy.bincount(clusters, minlength=num_clusters)
    return Y

# Input is num_songs * num_clusters
# Output is 1 * num_clusters
def get_artist_feature(all_songs_y):
    artist_feature = numpy.sum(all_songs_y, axis=0)
    return artist_feature/numpy.linalg.norm(artist_feature)

# Output is num_artists * num_clusters
def construct_artist_similarity_matrix(model, num_clusters, num_artists, artist_index, fs, tracks_sm):


    artist_similarity_matrix = numpy.zeros((num_artists, num_clusters))

    # for artist_index, artist in enumerate(data):
    for artist in (artist_index):
        a_i = artist_index[artist]
        artist_feature = []
        # for song in artist:

        artist_fs = fs.loc[fs['feature'].isin(tracks_sm.loc[tracks_sm['artist.12'] == artist]['Unnamed: 0'])]
        artist_fs = artist_fs.values[:, 1:]

        for song in artist_fs:

            song = song.reshape(1, len(song))

#             song_feature = []
#             # for clip in song:
#             for clip in range(10):
#                 # is of 1 X F dimensions
#                 # features = get_feature(clip)
#                 features = numpy.random.rand(1,13)
#                 song_feature.append(features)

#             song_features = numpy.vstack(song_feature)

            artist_feature.append(get_histogram_song(song, model, num_clusters))

        if artist_feature:
            artist_feature = numpy.vstack(artist_feature)
            artist_similarity_matrix[a_i, :] = get_artist_feature(artist_feature)

    return artist_similarity_matrix

def get_similarity(i, j, artist_similarity_matrix):
#     v = 1.0 - spatial.distance.cosine(artist_similarity_matrix[i, :], artist_similarity_matrix[j, :])
#     v = spatial.distance.cosine(artist_similarity_matrix[i, :], artist_similarity_matrix[j, :]) # simple cosine distance
    v = spatial.distance.euclidean(artist_similarity_matrix[i, :], artist_similarity_matrix[j, :])
#     v = np.linalg.norm(artist_similarity_matrix[i, :] - artist_similarity_matrix[j, :])
    return v


def scale(mat):
    new = np.zeros(mat.shape)
    for i in range(len(mat)):
        m = np.amax(mat[i, :])
        if m>0:
            new[i, :] = 1 - mat[i, :]/m
    return new




def get_eval_metrics(model_out, ground):

    # Measure Performance
    ntp = numpy.sum((model_out == 1) & (ground == 1))
    nfn = numpy.sum((model_out == 0) & (ground == 1))
    ntn = numpy.sum((model_out == 0) & (ground == 0))
    nfp = numpy.sum((model_out == 1) & (ground == 0))

    tpr = ntp*1.0/(ntp+nfn)
    tnr = ntn*1.0/(ntn+nfp)

    acc = (ntp+ntn)*1.0/(ntp+nfn+ntn+nfp)

    # float(tp)/tpg
    return ntp, nfn, tpr, tnr, acc

def save_things(name, table):
    with open('/Users/anirudhmani/Prog/thesis_related/data_business/saved_data/%s.pickle'%name, 'wb') as writer:
        pickle.dump(table, writer, protocol=pickle.HIGHEST_PROTOCOL)


def main_run():

    # Load metadata and features.
    tracks = pd.read_csv('/Users/anirudhmani/Prog/thesis_related/data_business/fma_metadata/tracks.csv')
    r_tracks = pd.read_csv('/Users/anirudhmani/Prog/thesis_related/data_business/fma_metadata/raw_tracks.csv')
    genres = pd.read_csv('/Users/anirudhmani/Prog/thesis_related/data_business/fma_metadata/genres.csv')
    features = pd.read_csv('/Users/anirudhmani/Prog/thesis_related/data_business/fma_metadata/features.csv')
    echonest = pd.read_csv('/Users/anirudhmani/Prog/thesis_related/data_business/fma_metadata/echonest.csv')
    r_artists = pd.read_csv('/Users/anirudhmani/Prog/thesis_related/data_business/fma_metadata/raw_artists.csv')

    # overall subset
    tracks_sm = tracks.loc[(tracks['set.1'] == 'small') & (tracks['set'] == 'training')]

    # Get Features
    f1 = features['feature']
    # f2 = features[['spectral_centroid','spectral_centroid.1', 'spectral_centroid.2', 'spectral_centroid.3', 'spectral_centroid.4', 'spectral_centroid.5', 'spectral_centroid.6']]
    # f3 = features.iloc[:, 512:]
    # f3 = features.iloc[:, 393:400]
    # f2 = features.iloc[:, 253:393] # mfcc
    f2 = features.iloc[:, 400:470] # spectral bunch
    fs = pd.concat([f1, f2], axis=1)
    # fs = pd.concat([fs, f3], axis=1)

    fs = fs.loc[fs['feature'].isin(tracks_sm['Unnamed: 0'])]
    overall_fs = fs.values[:, 1:]


    model = get_bog_model(overall_fs, 60)

    artist_rel, artist_fs, artist_index = get_artists()

    artist_clusters_matrix = construct_artist_similarity_matrix(model, 60, len(artist_index), artist_index, fs, tracks_sm)

    artist_sim_matrix = np.zeros((len(artist_index), len(artist_index)))

    for a_i_1 in artist_index:
        for a_i_2 in artist_index:
    #         if a_i_1 == a_i_2:
    #             continue
    #         print a_i_1, a_i_2
    #         print artist_index[a_i_1], artist_index[a_i_2]
            sim = get_similarity(artist_index[a_i_1], artist_index[a_i_2], artist_clusters_matrix)
            artist_sim_matrix[artist_index[a_i_1], artist_index[a_i_2]] = sim
            artist_sim_matrix[artist_index[a_i_2], artist_index[a_i_1]] = sim

    # scaling and getting actual similarity rather than distance
    artist_sim_matrix = scale(artist_sim_matrix)

    # get the ground truth
    ground = np.load('/Users/anirudhmani/Prog/thesis_related/data_business/saved_data/ground.npy')

    # Prepping ground
    ground = ground > 0.5
    for i in range(ground.shape[0]):
        ground[i, i] = 1.0


    results = []

    # for thresh in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    for thresh in np.arange(0.0025, 0.2, 0.0025):

        model_out = np.zeros(artist_sim_matrix.shape)

        for i in range(len(artist_sim_matrix)):
            # model_out[i, :] = artist_sim_matrix[i, :] > sim_means[i]
            model_out[i, :] = artist_sim_matrix[i, :] > thresh

            #     model_out[i, :] = artist_sim_matrix[i, :] > 0.5

        ntp, nfn, tpr, tnr, acc = get_eval_metrics(model_out, ground)
        results.append([ntp, nfn, tpr, tnr, acc])

    save_things('results_eval', results)

main_run()
