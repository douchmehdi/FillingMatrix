
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:43:52 2017

@author: mehdi
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# On se place dans le bon repertoire
import os
os.chdir("/home/mehdi/Documents/Convex optimization/Projet")


# Load rating file
data = pd.read_csv('rating.csv', sep='\t', names=['user', 'item', 'rating'])


# Ratings as a matrix users x movies
rating = pd.pivot_table(data, index='user', columns='item', values='rating')
n_user, n_item = rating.shape

# Display the matrix of ratings
plt.figure(figsize=(14, 6))
plt.imshow(rating)
plt.axis('off')
plt.colorbar()
plt.title('Rating');


# Mask of observed ratings
mask = np.asarray(rating > 0, dtype=int)  # Numpy array
print('Ratio of observed ratings: {0:0.1f}%'.format(100*mask.mean()))

# Minimisation -> alternate least-square
Q = rating.values
W = mask


lambda_ = 0.1
n_factors = 70
m, n = Q.shape
n_iterations = 20

X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)

def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)
    
def get_error_hat(Q, Q_hat, W):
    return np.sum((W * (Q - Q_hat))**2)

#errors = []
#for ii in range(n_iterations):
#    X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), 
#                        np.dot(Y, Q.T)).T
#    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
#                        np.dot(X.T, Q))
#    if ii % 100 == 0:
#        print('{}th iteration is completed'.format(ii))
#    errors.append(get_error(Q, X, Y, W))
#Q_hat = np.dot(X, Y)
#print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))
#
#plt.plot(errors);


weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)
#print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))

plt.plot(weighted_errors);
plt.xlabel('Iteration Number');
plt.ylabel('Mean Squared Error');


plt.figure(figsize=(14, 6))
plt.imshow(weighted_Q_hat)
plt.axis('off')
plt.colorbar()
plt.title('Rating');


# We prepare our completed matrix to be stored
frame_res = rating.copy()  
frame_res.loc[:, :] = weighted_Q_hat

rating_res_full_to_csv = frame_res.copy()
rating_res_full_to_csv['user'] = rating_res_full_to_csv.index
rating_res_full_to_csv = pd.melt(rating_res_full_to_csv, id_vars='user',
                                 value_vars=list(np.arange(n_item)+1),
                                 var_name='item', value_name='rating')

# Save ALL predictions 
rating_res_full_to_csv[['user', 'item', 'rating']].to_csv(
                                                        'MD_GMB_rating_full.csv',
                                                         sep='\t', header=False,
                                                         index=False
                                                         )
# Initializing with data (user/item)
rating_res_for_test_to_csv = pd.read_csv('rating_public_test.csv', sep='\t',
                                         names=['user', 'item'])
rating_res_for_test_to_csv['rating'] = np.zeros(rating_res_for_test_to_csv.shape[0]) 

# Fill with reconstructed ratings 
for index, (user, item, r) in rating_res_for_test_to_csv.iterrows():
    rating_res_for_test_to_csv.loc[index, 'rating'] = frame_res.loc[int(user), int(item)]
rating_res_for_test_to_csv[:10]

# Save predictions => THIS FILE SHOULD BE SENT TO LEADERBOARD RANKING
rating_res_for_test_to_csv[['user', 'item', 'rating']].to_csv(
                                                             'MD_GMB_rating_public_pred.csv',
                                                              sep='\t',
                                                              header=False,
                                                              index=False
                                                              )
