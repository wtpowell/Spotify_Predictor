#Visualization Libraries
import matplotlib.pyplot as plt

#Data Handling, Manipulation Libraries
import pandas as pd
import numpy as np
from pandas.io.sql import execute

import sklearn.metrics  as metrics

from sklearn.cluster import KMeans

from tqdm import tqdm

from sqlalchemy import create_engine
rds_connection_string = "postgres:group6123@spotify-predictor.cpuqub4subu1.us-east-1.rds.amazonaws.com/spotify_cleaned_data"
engine = create_engine(f'postgresql://{rds_connection_string}')
print (engine.table_names())
df = pd.read_sql_query('''select * from "Spotify_Predictor_Data"''', con=engine)
print (df)


#df = pd.read_csv("fixed_cleaned_spotify_data.csv")

df.describe()

df.info()


#Remove duplicate records with same uri
print('unique records for uri',len(df['uri'].unique()))
print('Records before dropping duplicate records',len(df))
df.drop_duplicates(['uri'], keep='last',inplace=True)
df.reset_index(drop=True, inplace=True)
print('After data cleaning',len(df))

df=df.reindex(columns=['danceability','energy','tempo','artist','track','uri'])
df
df['danceability'] = pd.to_numeric(df['danceability'])
df['energy'] = pd.to_numeric(df['energy'])
df['tempo'] = pd.to_numeric(df['tempo'])

#Normalize function for columns
def normalize_column(col):
    max_data = df[col].max()
    min_data = df[col].min()
    df[col] = (df[col] - min_data)/(max_data - min_data)
    
#Normalizing all the numerical values so that minimum value is 0 and maximum value is 1
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df.select_dtypes(include=num_types)
        
for col in num.columns:
    normalize_column(col)
print(num)

cost =[]
for i in range(1, 10):
    KM = KMeans(n_clusters = i, max_iter = 500)
    KM.fit(num)
    #Calculates squared error for k-means
    cost.append(KM.inertia_)     

# plot the cost against K values
# plt.plot(range(1, 10), cost, color ='g', linewidth ='3')
# plt.xlabel("Value of K")
# plt.ylabel("Sqaured Error (Cost)")
# plt.show() 

#Predictions using K-means
kmeans = KMeans(n_clusters = 5)
predict = kmeans.fit_predict(num)
df['predict'] = predict
normalize_column('predict')
print(df)


class Song_Recommender():
    def __init__(self, data):
        self.data_ = data
    
#function that returns recommendations
    def get_recommendations(self,danceability, energy, tempo, n_top):
        distances = []
        new_df = {}
        new_df =[danceability,energy,tempo]
        #grabbing all the data except the data entered 
        remaining_data = self.data_[(self.data_.danceability != danceability) & (self.data_.energy != energy )& (self.data_.tempo != tempo)]
        #looping through all the records in the remaining data
        for i in tqdm(remaining_data.values):
            #making the distance 0 befoere calculating it for every other record 
            dist = 0
            #looping through all the columns in a record
            for col in np.arange(len(remaining_data.columns)-1):
                if not col in [3,4,5]:
                    #The distance measure we used was manhattan 
                    dist = dist + np.absolute(float(new_df[col]) - float(i[col]))
            distances.append(dist)
        remaining_data['distance'] = distances
        
        #sorting our data to be ascending by 'distance' feature
        remaining_data = remaining_data.sort_values('distance')
        columns = ['uri']
        return remaining_data[columns][:n_top]

#Instanstiate the Recommender Class
recommender = Song_Recommender(df)
results = recommender.get_recommendations(0.4,0.4,0.8, 1)

#string spliting

stringresults = str(results)
spotifysetup = stringresults.split(':')
verygoodvariable = "https://open.spotify.com/embed/{}/{}".format(spotifysetup[1],spotifysetup[2])
print (verygoodvariable)


#output = execute('/start')

