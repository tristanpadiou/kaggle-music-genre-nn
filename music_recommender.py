import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

from sklearn.metrics.pairwise import cosine_similarity


def data_cleanup():
    music_df=pd.read_csv('music_genre.csv')
    music_df=music_df.dropna()
    music_df=music_df[music_df['tempo']!='?']
    music_df=music_df[music_df['artist_name']!='empty_field']
    music_df=music_df[music_df['duration_ms']>0]
    data=music_df.copy().drop(['artist_name','track_name','instance_id','obtained_date','popularity','duration_ms'],axis=1).reset_index(drop=True)
    songs=music_df.copy()[['instance_id','artist_name','track_name','music_genre']].reset_index(drop=True)

    # encoding data
    le=LabelEncoder()
    data['key']=le.fit_transform(data['key'])
    data['mode']=le.fit_transform(data['mode'])
    return data, songs



def search_artist(songs):
    while True:
        print('-----------------------------------------------------------------')
        artist=input('Which artist would you like to search a song for ?')
        print('-----------------------------------------------------------------')
        if artist in songs['artist_name'].to_list():
            print(f'You have selected {artist}:')
            # searching for artist
            selected_artist=songs.loc[songs['artist_name']==artist]
            #list songs
            print('-----------------------------------------------------------------')
            print('Here are their songs on file:')
            print(selected_artist['track_name'])
            print('-----------------------------------------------------------------')
            return selected_artist, artist

        else: 
            print(f'{artist} not in the database.')
            print('please try something else.')

def song_to_search(songs,data):
    selected_artist,artist=search_artist(songs)
    # selecting song and getting its unique id
    song_name=input('Which song would you like to search ?')
    while True:
        if song_name.isdigit():
            id=selected_artist.loc[int(song_name)]['instance_id']
            
            song_type=songs.loc[int(song_name)]['music_genre']
            #isolating song_type
            filtered_songs=songs.loc[songs['music_genre']==song_type].reset_index(drop=True)
            # getting the song's index number
            song_idx=filtered_songs.loc[filtered_songs['instance_id']==id].index
            song_idx=song_idx[0]
            new_song_name=selected_artist.loc[int(song_name)]['track_name']
            print('-----------------------------------------------------------------')
            print(f'selected song: {artist} : {new_song_name} : {song_type}')
            print('-----------------------------------------------------------------')
            filtered_data=data.loc[data['music_genre']==song_type].reset_index(drop=True)
            return song_idx, filtered_songs, filtered_data
        else:
            id=selected_artist.loc[selected_artist['track_name']==song_name]['instance_id'].reset_index(drop=True)
            id=id[0]
            # getting song_type
            song_type=songs.loc[songs['instance_id']==id]['music_genre'].reset_index(drop=True)
            song_type=song_type[0]
            #isolating song_type
            filtered_songs=songs.loc[songs['music_genre']==song_type].reset_index(drop=True)
            # getting the song's index number
            song_idx=filtered_songs.loc[filtered_songs['instance_id']==id].index
            song_idx=song_idx[0]
            print('-----------------------------------------------------------------')
            print(f'selected song: {artist} : {song_name} : {song_type}')
            print('-----------------------------------------------------------------')
            filtered_data=data.loc[data['music_genre']==song_type].reset_index(drop=True)
            return song_idx, filtered_songs, filtered_data

def search_engine(songs,data):
    song_idx, filtered_songs, filtered_data=song_to_search(songs,data)
    filtered_data=filtered_data.drop('music_genre',axis=1)
    scaler=StandardScaler()
    data_scaled=scaler.fit_transform(filtered_data)     

    #compute similarities between songs
    cosine_sim =cosine_similarity(data_scaled,data_scaled)

    #converting matrix to dataframe
    cosine_sim=pd.DataFrame(cosine_sim)

    # iterating through the similarity scores to find closest to the song selected using index
    rec=cosine_sim[song_idx]
    score=rec.sort_values(ascending=False)
    similar=filtered_songs.iloc[score.index]
    best_10=similar[1:10].reset_index(drop=True)
    return best_10


data,songs=data_cleanup()

while True:
    results=search_engine(songs,data)
    print (results[['artist_name','track_name','music_genre']])
    do_over=input('Would you like to try another song ? [y]es or [n]o').lower()
    match do_over:
        case "y":
            print("What other song would you like to look for?")
            
        case "n":
            print("Have a good one.")
            break
            
        case _:
            print("Please try again.")