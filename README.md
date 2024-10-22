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


## The Recommender

### Introduction

The primary goal of this project was to experiment on a music dataset to create a neural network 
that would predict the music genre based on the song's attributes which looks something like this:

![Screenshot 2024-10-22 141440](https://github.com/user-attachments/assets/eada12c2-de9d-4e58-890d-7b7c4c2a1d87)

The dataset needed quite a bit of cleaning as some nulls were present and some strings were among the floats and int

