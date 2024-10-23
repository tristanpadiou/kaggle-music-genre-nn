# Music Recommender

Kaggle dataset link: 

<https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre/data>

File: music_genre.csv

Feature list:

- instance_id	
- artist_name	
- track_name	
- popularity	
- acousticness	
- danceability	
- duration_ms	
- energy	
- instrumentalness	
- liveness	
- loudness	
- speechiness	
- tempo	
- valence
- music_genre

## This project has 2 parts:

1) [The Music Genre Prediction Neural Network Model](#the-music-genre-prediction-neural-network-model)
2) [The Music Recommender](#the-music-recommender)

## The Music Genre Prediction Neural Network Model:

file:
```
data.ipynb
```

### Introduction:

The primary goal of this project was to experiment on a music dataset to create a neural network 
that would predict the music genre based on the song's attributes which looks something like this:

![Screenshot 2024-10-22 141440](https://github.com/user-attachments/assets/eada12c2-de9d-4e58-890d-7b7c4c2a1d87)

### Steps for nn:

- Data Cleaning:

The dataset needed quite a bit of cleaning as some nulls were present and some strings were among the floats and int turning some features into objects. But once all of that was cleaned up, I could take a good look at the data.

- Preprocessing:
  
Then I encoded the necessary features, created my X and y var, split the data into training and testing sets, and scaled X_train and X_test. 
```
y=data_encoded['music_genre']
y=pd.get_dummies(y).astype(int)
y=y.reset_index(drop=True)
y

X=data_encoded.drop(['music_genre','artist_name'],axis=1).reset_index(drop=True)
```
- Creating and tuning tensorflow.keras model:
  
I then initiated keras tuner for hyperparameters with a sequential keras model. Since the y variable is multiclass, the loss metric would be 'categorical_crossentropy' and the activation of the output: 'softmax', which would return probabilities of how sure the model was of certain classes.


~~~
def create_model(hp):
      nn_model = tf.keras.models.Sequential()
  
      # Allow kerastuner to decide which activation function to use in hidden layers
      activation = hp.Choice('activation',['relu','tanh'])
  
      # Allow kerastuner to decide number of neurons in first layer
      nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
          min_value=20,
          max_value=60,
          step=5), activation=activation, input_dim=len(X.columns)))
  
      # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
      for i in range(hp.Int('num_layers', 1, 5)):
          nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
              min_value=20,
              max_value=60,
              step=5),
              activation=activation))
  
      nn_model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
  
      # Compile the model
      nn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
  
      return nn_model
~~~
The model was then created as per the recommendation and had an accuracy of 0.96 at 200 epochs.

## The Music Recommender:

Files:
```
music_recommendation_data.ipynb

music_recommender.py

run_music_recommender.bat
```
### Introduction:

After looking at the data for the previous part, I decided to create a content based filtering music recommender that would find songs similar to a chosen song based on its attributes:

- popularity	
- acousticness	
- danceability	
- duration_ms	
- energy	
- instrumentalness	
- liveness	
- loudness	
- speechiness	
- tempo	
- valence
- music_genre

Ultimately, I elected to remove popularity from the features because I wanted to find less popular songs, making the recommendation more interesting.
### Steps:

- Data Cleaning & Preprocessing:

Same process as before, removing nulls and strings from columns that should be d_type float or int. Encoding where I needed to and scaling the result.

The main difference is I kept a list of the instance id, artist names, and track names in an object called songs to then use later to have the names of the recommended songs.

```
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
```

- Building the model:

After thoroughly looking online for ways to build this recommender, I ended up using cosine_similarity to create a similarity matrix of the data against itself. However, due to the sheer length of the data about 38000 rows, the matrix was 38000 by 38000, eating up all the ram. However, the recommendations were quite good for the amount of data available on the songs. Another problem is that the matrix is way too big to save and upload ~ 30gb.

- Optimizing the search:

After testing different methods I found the solution. Creating a function that would filter the dataset with the music_genre of the chosen song before creating the matrix. This resulted in a much smaller ram usage impact but also improved the quality of the recommendations.

```
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
```
- Building The Recommender:

The code above is part of the .py file where I moved all the functions I created to create a form of interactive application that works through a terminal.
I added some user inputs as well as some loops and created a .bat file to runs the .py file more conveniently (without having to look for the directory).

- Try it!!

To try it simply run the Run_music_recommender.bat file.







  


