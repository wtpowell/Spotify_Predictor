# Spotify_Predictor

The purpose of this project was to incorporate Machine Learning (ML) into a Spotify dataset that would predict a desired song that best matches a user input. 
We began by identifying CSV's from kaggle.com that provided various song metrics over multiple decades.
From there we cleaned the data through Pandas to create a combined CSV from 1960s-2010s. 
Once we had the data compiled into one CSV we began identifying which song metrics would resonate most with users. We landed on Energy, Temp and Danceability as desired metrics users would most likely associate song selection with. 
In order to provide this user experience we created a dynamic range slider using HTML, CSS, and JavaScript to display the user's selected value for danceability, energy, and tempo. 
As we are building a model for recommendation systems, we have adopted one of the models which uses the user’s history already in the system as this dataset has some good amount of user data to base our decisions on. 
One of them is clustering algorithms and we thought to use K-means clustering algorithm for our model. We use elbow method to estimate the k-value. 
As a result the user will be provided with a song that best matches their desired energy, danceability and tempo via our ML Spotify Predictor.
