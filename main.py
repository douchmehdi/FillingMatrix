#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:01:21 2017

@author: mehdi
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cvxpy


# On se place dans le bon repertoire
import os
os.chdir("/home/mehdi/Documents/Convex optimization/Projet")


# Load rating file
data = pd.read_csv('rating.csv', sep='\t', names=['user', 'item', 'rating'])
data[:10]

# Ratings as a matrix users x movies
rating = pd.pivot_table(data, index='user', columns='item', values='rating')
rating = rating.fillna(value=0)
n_user, n_item = rating.shape
rating[:10]

# Display the matrix of ratings
plt.figure(figsize=(14, 6))
plt.imshow(rating)
plt.axis('off')
plt.colorbar()
plt.title('Rating');


# Mask of observed ratings
mask = np.asarray(rating > 0, dtype=int)  # Numpy array
print('Ratio of observed ratings: {0:0.1f}%'.format(100*mask.mean()))


# Fonctions dont on a besoin pour le benchmark
def proxS1(x, gamma):
    U, s, V = np.linalg.svd(x)
    s_gamma = np.fmax(0, s - gamma)
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s_gamma), :len(s_gamma)] = np.diag(s_gamma)
    return np.dot(U, np.dot(S, V))
    
def DRmethod(im_masked, mat_mask, n_it=100, version=1):
    proxA = lambda x: (1-mat_mask)*x + im_masked
    
    if version == 1:
        proxf = lambda x, s: proxS1(x, s)
        proxg = lambda x, s: proxA(x)
    else:
        proxf = lambda x, s: proxA(x)
        proxg = lambda x, s: proxS1(x, s)
    
    z = np.zeros(im_masked.shape)  # Initial point
    mu, s = 1, 1  # Relaxation and step size
    
    obj = []
    for it in range(n_it):
        x = proxf(z, s)
        z = z + mu*(proxg(2*x-z, s) - x)
        obj.append( np.linalg.norm(x, ord='nuc') )
    return x, obj
    
def fista(im_masked, mat_mask, mu=1., n_it=100):
    gamma = 1 / (mu * np.abs(mat_mask).max())
    x = np.zeros(im_masked.shape)
    z = np.zeros(im_masked.shape)
    m = 0
    
    obj = []
    for it in range(n_it):
        x_new = proxS1(z - gamma*mu*(mat_mask*z - im_masked), gamma)
        m_new = (1 + np.sqrt(1 + 4*m**2))/2
        z = x_new + (m-1)/m_new * (x_new - x)
        x = x_new
        m = m_new
        obj.append( np.linalg.norm(x, ord='nuc') + mu/2*np.linalg.norm(mat_mask*x - im_masked)**2 )

    return x, obj
    
# Douglas Rachford
n_it = 1000
imdr1, objdr1 = DRmethod(rating, mask, n_it=n_it)
imdr2, objdr2 = DRmethod(rating, mask, version=2, n_it=n_it)

plt.figure(figsize=(14, 6))
plt.imshow(imdr1)
plt.axis('off')
plt.colorbar()
plt.title('Rating');

# Fista -> régularisation
mu = 1e1
imfista, objfista = fista(rating, mask, mu=mu, n_it=200)

plt.figure(figsize=(14, 6))
plt.imshow(imfista)
plt.axis('off')
plt.colorbar()
plt.title('Rating');


# We prepare our completed matrix to be stored

frame_res = rating.copy()  
frame_res.loc[:, :] = imdr2 # choisir la matrice à stocker

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
