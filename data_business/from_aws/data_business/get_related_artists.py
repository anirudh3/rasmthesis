#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:52:26 2018

@author: anirudhmani
"""

import sys
import spotipy
import pprint
''' shows recommendations for the given artist
'''


from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(client_id='ba6790bdcd434f06b7b577e344c6e0ae', client_secret='145715e565ff48469c306484896a34f5')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None




def get_related(uri):
    
    related = sp.artist_related_artists(uri)
    ans = [] 
    
    for artist in related['artists']:
        ans.append(artist['name'])
#        print artist['name']
        
    
    return ans
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(('Usage: {0} artist name'.format(sys.argv[0])))
    else:
        name = ' '.join(sys.argv[1:])
        artist = get_artist(name)
#        print artist['id']
        if artist:
#            show_recommendations_for_artist(artist)
            related_artists = get_related(artist['uri'])
#            print related_artists
        else:
            print "Can't find that artist", name