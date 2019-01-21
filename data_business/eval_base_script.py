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


# overall subset
tracks_sm = tracks.loc[(tracks['set.1'] == 'small') & (tracks['set'] == 'training')]

# subset based on artists in set
# load
def load_spot_rel_artists(name):
    with open(name, 'rb') as handle:
        thing = pickle.load(handle)
    return thing
artist_rel = load_spot_rel_artists("saved_data/artist_rel_small_training.pickle")


# ipd.display(features.columns[253:400])


len(artist_rel)

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


artist_tracks = tracks_sm.loc[(tracks_sm['artist.12'] == artist_all.keys()[4])]

artist_tracks['Unnamed: 0']

class FeatureData:

    'Music audio features for genre classification'
    hop_length = None

    dir_trainfolder = "/Users/anirudhmani/Prog/thesis_related/data/fma_small_consolidated"
    dir_devfolder = "/Users/anirudhmani/Prog/thesis_related/data/fma_small/001/"
    dir_testfolder = "/Users/anirudhmani/Prog/thesis_related/data/fma_small/002/"
#     dir_all_files = "/Users/anirudhmani/Prog/thesis_related/data"

    train_X_preprocessed_data = 'data_train_input.npy'
    train_Y_preprocessed_data = 'data_train_target.npy'
    dev_X_preprocessed_data = 'data_validation_input.npy'
    dev_Y_preprocessed_data = 'data_validation_target.npy'
    test_X_preprocessed_data = 'data_test_input.npy'
    test_Y_preprocessed_data = 'data_test_target.npy'

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []

    def load_preprocess_data(self):
        self.trainfiles_list = self.path_to_audiofiles(self.dir_trainfolder)
        self.devfiles_list = self.path_to_audiofiles(self.dir_devfolder)
        self.testfiles_list = self.path_to_audiofiles(self.dir_testfolder)

        all_files_list = []
        all_files_list.extend(self.trainfiles_list)
        all_files_list.extend(self.devfiles_list)
        all_files_list.extend(self.testfiles_list)

        #self.precompute_min_timeseries_len(all_files_list)
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set
        self.train_X = self.extract_audio_features(self.trainfiles_list)
        with open(self.train_X_preprocessed_data, 'wb') as f:
            np.save(f, self.train_X)
#         with open(self.train_Y_preprocessed_data, 'wb') as f:
#             self.train_Y = self.one_hot(self.train_Y)
#             np.save(f, self.train_Y)

        # Validation set
#         self.dev_X, self.dev_Y = self.extract_audio_features(self.devfiles_list)
#         with open(self.dev_X_preprocessed_data, 'wb') as f:
#             np.save(f, self.dev_X)
#         with open(self.dev_Y_preprocessed_data, 'wb') as f:
#             self.dev_Y = self.one_hot(self.dev_Y)
#             np.save(f, self.dev_Y)

        # Test set
#         self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
#         with open(self.test_X_preprocessed_data, 'wb') as f:
#             np.save(f, self.test_X)
#         with open(self.test_Y_preprocessed_data, 'wb') as f:
#             self.test_Y = self.one_hot(self.test_Y)
#             np.save(f, self.test_Y)

    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
#         self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.dev_X = np.load(self.dev_X_preprocessed_data)
#         self.dev_Y = np.load(self.dev_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
#         self.test_Y = np.load(self.test_Y_preprocessed_data)

    def precompute_min_timeseries_len(self, list_of_audiofiles):
        for file in list_of_audiofiles:
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_length))

    def extract_audio_features(self, list_of_audiofiles):
        #timeseries_length = min(self.timeseries_length_list)
        timeseries_length = 128 # was 128 previously
        t_length = 12
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 13), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):

            print file

            try:
                y, sr = librosa.load(file)



                mfcc = []
                spectral_center = []
                chroma = []
                spectral_contrast = []


                mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
#             spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
#             chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
#             spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)



                splits = re.split('[ .]', file)
#             genre = re.split('[ /]', splits[1])[3]
#             target.append(genre)



                data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
#             data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
#             data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
#             data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

#             print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

            except:
                print "error reading.....continuing"
                continue

        print("Extracted audio features .....")
        return data

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".mp3"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio

    def get_meaned(self, tracks_data, ax):
        return np.mean(tracks_data, axis=ax)

song_features = FeatureData()

song_features.load_preprocess_data()

tracks_data = song_features.train_X # num of tracks * Time Length * Num of Features
tracks_data = tracks_data.reshape(tracks_data.shape[0]*tracks_data.shape[1], tracks_data.shape[2])

def get_bog_model(data, num_clusters):
    # all features in all songs in all artists
    # X = get_data
#     A = 100
#     F = 13
#     all_features = numpy.random.rand(A, F)
    model = KMeans(num_clusters)
    model.fit(data)
    return model

model = get_bog_model(tracks_data, 100)

def save_spot_rel_artists(name, table):
    with open('saved_data/%s.pickle'%name, 'wb') as writer:
        pickle.dump(table, writer, protocol=pickle.HIGHEST_PROTOCOL)


save_spot_rel_artists('cluster_model', model)
