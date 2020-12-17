# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:52:38 2020

@author: Shen Fan
The last part of my code is commented out, because it'll cost more than 40mins
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('ratings.csv', usecols=['userId','movieId','rating'])
dataset = pd.read_csv('movies.csv')
data = data.merge(dataset, on='movieId')
#number of users
num_users = data['userId'].nunique()
#number of movies
num_movies = data['movieId'].nunique()
#preprocess movie ids, the original movie id numbers are too large
movie_ids = data['movieId'].unique().tolist()
newmovie = {x: i for i, x in enumerate(movie_ids)}
back2movie = {i: x for i, x in enumerate(movie_ids)}
data["movie"] = data["movieId"].map(newmovie)

#matrix of all the users' rating
ratings = np.zeros((num_users, num_movies))
for row in data.itertuples(index = False):
    ratings[row.userId - 1, row.movie] = row.rating
    
#training and test
X_train, X_test = train_test_split(data, test_size = 0.1)

train_matrix = np.zeros((num_users, num_movies))
for row in X_train.itertuples():
    train_matrix[row.userId - 1, row.movie - 1] = row.rating

test_matrix = np.zeros((num_users, num_movies))
for row in X_test.itertuples():
    test_matrix[row.userId - 1, row.movie - 1] = row.rating

#matrix factorization using gradient descent
def performance(train, test, iterations, latent, learning_rate, lmbda):
    #multiply 3 for better result, I also used latent features for better results
    U = 3 * np.random.rand(num_users, latent)
    V = 3 * np.random.rand(num_movies, latent)
    train_record = []
    train_error = []
    test_error = []
    users, movies = train.nonzero()
    for step in range(iterations):
        for i, j in zip(users, movies):
            error = train[i, j] - np.dot(U[i,:],V[j,:].T)
            U[i,:] += learning_rate * (error * V[j,:] - lmbda * U[i,:])
            V[j,:] += learning_rate * (error * U[i,:] - lmbda * V[j,:])
        train_rmse = rmse(train, predict(U, V))
        test_rmse = rmse(test, predict(U,V))
        train_error.append(train_rmse)
        test_error.append(test_rmse)
        #observed index for training set
        index = np.nonzero(train)
        UV = predict(U, V)
        e = (train[index] - UV[index]) ** 2
        e = e.sum() / 2
        U_norm = np.linalg.norm(U, 'fro')
        V_norm = np.linalg.norm(V, 'fro')
        F = e + lmbda/2 * (U_norm ** 2 + V_norm ** 2)
        train_record.append(F)
    return U, V, train_record, train_error, test_error  
        
def predict(U, V):
    return np.dot(U, V.T)

def rmse(original, prediction):
    #only calculate observed values 
    index = np.nonzero(original)
    return sqrt(mean_squared_error(original[index], prediction[index]))


import time
start_time = time.time()

U, V, train_record, train_error, test_error = performance(train_matrix, test_matrix, 40, 2, 0.05, 1)
plt.plot(train_record)
plt.xlabel('iterations')
plt.ylabel('F')
plt.show()

UV = predict(U, V)
test_rmse = rmse(test_matrix, UV)
print(f"RMSE for the choice \u03BB = 1: {test_rmse}.\n")

plt.plot(train_error, marker='v', label='Training Data')
plt.plot(test_error, marker='^', label='Testing Data')
plt.xlabel('iterations')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()

#select a user to see results 
user_id = data.userId.sample(1).iloc[0]
movies_watched_by_user = data[data.userId == user_id]
movies_not_watched = dataset[
    ~dataset["movieId"].isin(movies_watched_by_user.movieId.values)
]["movieId"]
movies_not_watched = list(
    set(movies_not_watched).intersection(set(newmovie.keys()))
)
movies_not_watched = [[newmovie.get(x)] for x in movies_not_watched]
user_movie_array = np.hstack(
    ([[user_id]] * len(movies_not_watched), movies_not_watched)
)
unwatched_list = user_movie_array[:,1].tolist()
#rating prediction
new_ratings = UV
user_ratings = new_ratings[user_id - 1, unwatched_list]
top_indices = user_ratings.argsort()[::-1][0:10]
recommendation_ids = [
    back2movie.get(movies_not_watched[x][0]) for x in top_indices
]

print("Movies with high ratings from user:")
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_rows = dataset[dataset["movieId"].isin(top_movies_user)]
for row in movie_rows.itertuples():
    print(row.title, ":", row.genres)
    
print("\n")
print("Top 10 movie recommendations:")
recommended_movies = dataset[dataset["movieId"].isin(recommendation_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)


# #performance evalutaion with different lambda
# lmbda_list = [10 ** -6, 10 ** -3, 0.1, 0.5, 2, 5, 10, 20, 50, 100, 500, 1000]
# rmse_record = []
# for l in lmbda_list:
#     #I decreased learning rate for larger lambda value
#     U, V, train_record, train_error, test_error  = performance(train_matrix, test_matrix, 150, 2, 0.001, l)
#     l_rmse = rmse(test_matrix, predict(U, V))
#     rmse_record.append(l_rmse)
    
# plt.plot(lmbda_list, rmse_record)
# plt.xlabel('\u03BB')
# plt.ylabel('RMSE')
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

