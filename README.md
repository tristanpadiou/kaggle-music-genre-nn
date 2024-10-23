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
- 
[The Music Genre Prediction Neural Network Model](#the-music-genre-prediction-neural-network-model)

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


