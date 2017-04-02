#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:42:42 2017

@author: mehdi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
import time

from fancyimpute import NuclearNormMinimization


# On se place dans le bon repertoire
import os
os.chdir("/home/mehdi/Documents/Convex optimization/Projet")


# Load rating file
data = pd.read_csv('rating.csv', sep='\t', names=['user', 'item', 'rating'])

# Ratings as a matrix users x movies
rating = pd.pivot_table(data, index='user', columns='item', values='rating')

# Display the matrix of ratings
plt.figure(figsize=(14, 6))
plt.imshow(rating)
plt.axis('off')
plt.colorbar()
plt.title('Rating');

# Mask of observed ratings
mask = np.asarray(rating > 0, dtype=int)  # Numpy array
print('Ratio of observed ratings: {0:0.1f}%'.format(100*mask.mean()))

## Fuctions that give subdivisions of the initial matrix
def sub_inter(m, n):
	""" Returns indices of the signal subdivision
	"""
	l_interval = round(2 * m/(n + 1)) - 1
	pas = round(m/(n + 1))
	inters = []
	for i in range(n):
		inters.append((i*pas,i*pas + l_interval - 1))
	return inters

 
n1, n2 = rating.shape
nb_inter = 5

R = np.zeros((n1,n2))
C = np.zeros((n1,n2))

for int_x,int_y in product(sub_inter(n1, nb_inter), sub_inter(n2, nb_inter)):
    start = time.time()
    X_ = NuclearNormMinimization().complete(rating.values[int_x[0]:int_x[1],int_y[0]:int_y[1]])
    end = time.time()
    time_elapsed = end - start
    R[int_x[0]:int_x[1],int_y[0]:int_y[1]] += X_
    C[int_x[0]:int_x[1],int_y[0]:int_y[1]] += np.ones((int_x[1]-int_x[0],
                                                       int_y[1]-int_y[0]))
    

# We divide the matrix of the sum of scores by the number of score
X = R/C

# We look at the result
plt.figure(figsize=(14, 6))
plt.imshow(R)
plt.axis('off')
plt.colorbar()
plt.title('Rating');




# We prepare our completed matrix to be stored
frame_res = rating.copy()  
frame_res.loc[:, :] = X

rating_res_full_to_csv = frame_res.copy()
rating_res_full_to_csv['user'] = rating_res_full_to_csv.index
rating_res_full_to_csv = pd.melt(rating_res_full_to_csv, id_vars='user',
                                 value_vars=list(np.arange(n2)+1),
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