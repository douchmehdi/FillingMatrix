#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:42:42 2017

@author: mehdi
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute


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


# Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=3).complete(rating)

plt.figure(figsize=(14, 6))
plt.imshow(X_filled_knn)
plt.axis('off')
plt.colorbar()
plt.title('Rating');
# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
X_filled_nnm = NuclearNormMinimization().complete(rating)

plt.figure(figsize=(14, 6))
plt.imshow(X_filled_nnm)
plt.axis('off')
plt.colorbar()
plt.title('Rating');

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
X_filled_softimpute = SoftImpute().complete(rating)

plt.figure(figsize=(14, 6))
plt.imshow(X_filled_softimpute)
plt.axis('off')
plt.colorbar()
plt.title('Rating');


# We prepare our completed matrix to be stored
frame_res = rating.copy()  
frame_res.loc[:, :] = X_filled_nnm # name of the matrix you wish to store

rating_res_full_to_csv = frame_res.copy()
rating_res_full_to_csv['user'] = rating_res_full_to_csv.index
rating_res_full_to_csv = pd.melt(rating_res_full_to_csv, id_vars='user',
                                 value_vars=list(np.arange(n_item)+1),
                                 var_name='item', value_name='rating')

# Save ALL predictions 
rating_res_full_to_csv[['user', 'item', 'rating']].to_csv(
                                                        'MD_GMB_NNM_rating_full.csv',
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
                                                             'MD_GMB_NNM_rating_public_pred.csv',
                                                              sep='\t',
                                                              header=False,
                                                              index=False
                                                              )