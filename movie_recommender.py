import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer #to count and transform vector 
from sklearn.metrics.pairwise import cosine_similarity  #to find the similarities of vector 
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return movie_df[movie_df.index == index]["title"].values[0]

def get_index_from_title(title):
	return movie_df[movie_df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
movie_df = pd.read_csv('movie_dataset.csv')
#print (movie_df.head())

##Step 2: Select Features
#print(movie_df.columns)
features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
	movie_df[feature] = movie_df[feature].fillna('')

##Step 3: Create a column in DF which combines all selected features
def combined_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
			print ("Error", row)

#how to apply a function to a single or selected column 
movie_df['combined_features'] = movie_df.apply(combined_features, axis=1) #axis as 1 will pass all row individual
#print(movie_df['combined_features'].head())

##Step 4: Create count matrix from this new combined column
vectorizer = CountVectorizer()

count_matrix = vectorizer.fit_transform(movie_df['combined_features'])

##Step 5: Compute the Cosine Similarity based on the count_matrix
similarity = cosine_similarity(count_matrix)

movie_user_likes = "Space Dogs"

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movie = list(enumerate (similarity[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_movie = sorted(similar_movie, key=lambda x:x[1], reverse=True)

## Step 8: Print titles of first 50 movies
i=0
#print("Top 50 similar movies to "+movie_user_likes+" are:\n")
for movie in sorted_movie:
	#print(get_title_from_index(movie[0]))
	i+=1
	if (i>50):
		break
#find the similarity based on average 
#print(movie_df["vote_average"].isnull())

average_sorted_movie = sorted(similar_movie, key=lambda x:movie_df["vote_average"][x[0]], reverse=True)
#print(average_sorted_movie)


i=0
#print("Top 10 average voted movies to "+movie_user_likes+" are:\n")
for movie in average_sorted_movie:
	print(get_title_from_index(movie[0]))
	i+=1
	if (i>10):
		break