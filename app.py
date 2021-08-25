#Visualization Libraries
import matplotlib.pyplot as plt
#import seaborn as sns

#Data Handling, Manipulation Libraries
import pandas as pd
import numpy as np

import sklearn.metrics  as metrics
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

from tqdm import tqdm
from scipy.spatial.distance import cdist

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/my-link/')
def my_link():
  print ('I got clicked!')

  return 'Click.'

# # merging available five csv files
# df = pd.concat(
#     map(pd.read_csv, ['dataset-of-00s.csv', 'dataset-of-10s.csv','dataset-of-60s.csv','dataset-of-70s.csv','dataset-of-80s.csv','dataset-of-90s.csv']), ignore_index=True)
# print(df)

df = pd.read_csv("fixed_cleaned_spotify_data.csv")
df.head()
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


def normalize_column(col):
# 
#    col - column in the dataframe which needs to be normalized
#
    max_d = df[col].max()
    min_d = df[col].min()
    df[col] = (df[col] - min_d)/(max_d - min_d)
    
#Normalize allnumerical columns so that min value is 0 and max value is 1
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df.select_dtypes(include=num_types)
        
for col in num.columns:
    normalize_column(col)
print(num)

cost =[]
for i in range(1, 10):
    KM = KMeans(n_clusters = i, max_iter = 500)
    KM.fit(num)
      
    # calculates squared error
    # for the clustered points
    cost.append(KM.inertia_)     
  
# plot the cost against K values
plt.plot(range(1, 10), cost, color ='g', linewidth ='3')
plt.xlabel("Value of K")
plt.ylabel("Sqaured Error (Cost)")
plt.show() 




km = KMeans(n_clusters=5)
pred = km.fit_predict(num)
df['pred'] = pred
normalize_column('pred')
print(df)
# spotify_data['song_id']= [i for i,uri in enumerate(spotify_data['uri'])] 
# print(spotify_data)
# #Extracting Independent and dependent Variable  
# y= spotify_data.iloc[:,[6]].values  
# x= spotify_data.iloc[:,[3,4,5]].values  
  
# # # # Splitting the dataset into training and test set.  
# from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=42)  
# # # #feature Scaling  
# from sklearn.preprocessing import StandardScaler    
# st_x= StandardScaler()    
# X_train= st_x.fit_transform(X_train)    
# X_test= st_x.transform(X_test)   




class Song_Recommender():
#     """
#     Neighbourhood Based Collborative Filterng REcoomendation System using similarity Metrics
#     Manhattan Distance is calculated for all songs and Recommend Songs that are similar to it based on any given metrics
#     """
    def __init__(self, data):
        self.data_ = data
    
    #function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations(self, danceability, energy, tempo, n_top):
        distances = []
        new_df = {}
        #choosing the given song_name and dropping it from the data
        new_df =[danceability,energy,tempo]
        rem_data = self.data_[(self.data_.danceability != danceability) & (self.data_.energy != energy )& (self.data_.tempo != tempo)]
        for r_song in tqdm(rem_data.values):
            dist = 0
            for col in np.arange(len(rem_data.columns)-1):
                if not col in [3,4,5]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(new_df[col]) - float(r_song[col]))
            distances.append(dist)
        rem_data['distance'] = distances
        
        #sorting our data to be ascending by 'distance' feature
        rem_data = rem_data.sort_values('distance')
        columns = ['artist', 'track','uri','danceability','energy','tempo']
        return rem_data[columns][:n_top]




        #Instanstiate the Recommender Class
recommender = Song_Recommender(df)
print(recommender.get_recommendations(0.4,0.4,0.8, 5))

if __name__ == '__main__':
  app.run(debug=True)