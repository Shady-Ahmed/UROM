# -*- coding: utf-8 -*-
"""
Utilities file for reduced order modeling of 1D Burgers problem for the paper: 
 "A long short-term memory embedding for hybrid uplifted reduced order models",
 Physica D: Nonlinear Phenomena, 2020.
 authors: Shady E. Ahmed, Omer San, Adil Rasheed, Traian Iliescu

 Last update: 03_15_2020
 Contact: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# from numpy.random import seed
# seed(1)
# import tensorflow as tf
# tf.random.set_seed(2)

import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import joblib

#%% Define Functions

###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u


###############################################################################
#Interpolation Routines
###############################################################################  
# Grassmann Interpolation
def GrassInt(Phi,pref,p,pTest):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    
    nx,nr,nc = Phi.shape
    Phi0 = Phi[:,:,pref] 
    Phi0H = Phi0.T 

    
    Gamma = np.zeros((nx,nr,nc))
    for i in range(nc):
        templ = ( np.identity(nx) - Phi0.dot(Phi0H) )
        tempr = LA.inv( Phi0H.dot(Phi[:,:,i]) )
        temp = LA.multi_dot([templ,Phi[:,:,i],tempr])
               
        U, S, Vh = LA.svd(temp, full_matrices=False)
        S = np.diag(S)
        Gamma[:,:,i] = LA.multi_dot([U,np.arctan(S),Vh])
    
    alpha = np.ones(nc)
    GammaL = np.zeros((nx,nr))
    #% Lagrange Interpolation
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-p[j])/(p[i]-p[j])
    for i in range(nc-1):
        GammaL = GammaL + alpha[i] * Gamma[:,:,i]
            
    U, S, Vh = LA.svd(GammaL, full_matrices=False)
    PhiL = LA.multi_dot([ Phi0 , Vh.T ,np.diag(np.cos(S)) ]) + \
           LA.multi_dot([ U , np.diag(np.sin(S)) ])
    PhiL = PhiL.dot(Vh)
   
    return PhiL



###############################################################################
#LSTM Routines
############################################################################### 
def create_training_data_lstm1(input_set,output_set, m, nr, lookback):
    
    ytrain = [output_set[i,:] for i in range(lookback-1,m)]
    ytrain = np.array(ytrain)    
    xtrain = np.zeros((m-lookback+1,lookback,nr+1))
    for i in range(m-lookback+1):
        a = input_set[i,:nr+1]
        for j in range(1,lookback):
            a = np.vstack((a,input_set[i+j,:nr+1]))
        xtrain[i,:,:] = a
    return xtrain , ytrain


def create_training_data_lstm2(training_set, m, nr, lookback):
    ytrain = [training_set[i,nr+1:] for i in range(lookback-1,m)]
    ytrain = np.array(ytrain)    
    xtrain = np.zeros((m-lookback+1,lookback,nr+1))
    for i in range(m-lookback+1):
        a = training_set[i,:nr+1]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j,:nr+1]))
        xtrain[i,:,:] = a
    return xtrain , ytrain


def create_training_data_lstm3(training_set, m, nr, lookback):
    ytrain = [training_set[i,1:] for i in range(lookback,m)]
    ytrain = np.array(ytrain)    
    xtrain = np.zeros((m-lookback,lookback,nr+1))
    for i in range(m-lookback):
        a = training_set[i,:nr+1]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j,:nr+1]))
        xtrain[i,:,:] = a
    return xtrain , ytrain

###############################################################################
# Burgers Routines
###############################################################################
def uexact(x, t, nu):  #Exact Solution [Sirisup]
    t0 = np.exp(1.0/(8.0*nu))
    uexact = (x/(t+1.0))/(1.0+np.sqrt((t+1.0)/t0)*np.exp(x*x/(4.0*nu*(t+1.0))))
    return uexact

def rhs(nr, b_l, b_nl, a): # Right Handside of Galerkin Projection
    r2, r3, r = [np.zeros(nr) for _ in range(3)]
    
    for k in range(nr):
        r2[k] = 0.0
        for i in range(nr):
            r2[k] = r2[k] + b_l[i,k]*a[i]
    
    for k in range(nr):
        r3[k] = 0.0
        for j in range(nr):
            for i in range(nr):
                r3[k] = r3[k] + b_nl[i,j,k]*a[i]*a[j]
    
    r = r2 + r3    
    return r
###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]   
    up[e] = r[e]/b[e]   
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd


