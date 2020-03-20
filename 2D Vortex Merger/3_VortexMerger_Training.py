# -*- coding: utf-8 -*-
"""
LSTM training for reduced order modeling of 2D vortex merger problem for the paper: 
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
import pyfftw

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
#LSTM Routines
############################################################################### 
def create_training_data_lstm1(input_set,output_set, m, nr, lookback):
    
    for i in range(lookback-1,m):
        ytrain = rhs
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

#%% Galerkin Projection
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

#%% fast poisson solver using second-order central difference scheme
def fpsi(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f,0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
        
    return ut

#%%
def nonlinear_term(nx,ny,dx,dy,wf,sf):
    '''
    this function returns -(Jacobian)
    
    '''
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = wf
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    s = np.zeros((nx+3,ny+3))
    
    s[1:nx+1,1:ny+1] = sf
    
    # periodic
    s[:,ny+1] = s[:,1]
    s[nx+1,:] = s[1,:]
    s[nx+1,ny+1] = s[1,1]
    
    # ghost points
    s[:,0] = s[:,ny]
    s[:,ny+2] = s[:,2]
    s[0,:] = s[nx,:]
    s[nx+2,:] = s[2,:]
    
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+1,ny+1))
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    f = -(j1+j2+j3)*hh
                  
    return f[1:nx+1,1:ny+1]

def linear_term(nx,ny,dx,dy,re,f):
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = f
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    
    f = np.zeros((nx+1,ny+1))
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
    
    f = lap/re
            
    return f[1:nx+1,1:ny+1]

def pbc(w):
    f = np.zeros((nx+1,ny+1))
    f[:nx,:ny] = w
    f[:,ny] = f[:,0]
    f[nx,:] = f[0,:]
    
    return f

#%% Main program:
# Inputs
nx = 256   #spatial grid number
ny = 256
nc = 4      #number of control parameters (nu)
ns = 200    #number of snapshot per each Parameter 
R = 4       #number of modes 
Q = 4*R     #number of modes for 'super-resolution' [Hybridization]
Re_start = 200.0
Re_final = 800.0
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
dt = 1e-1
tm = 20.0

lookback = 3 #Number of lookbacks

#%%
print('Loading POD data')      
Phiw = np.load('./Basis/wBasis_Train.npy')
Phis = np.load('./Basis/sBasis_Train.npy')
aTrain = np.load('./Basis/Coeff_Train.npy')

        

#%% UROM Training

# create folder
if os.path.isdir("./LSTM Models"):
    print('LSTM models folder already exists')
else: 
    print('Creating LSTM models folder')
    os.makedirs("./LSTM Models")
    
    
# 1- LSTM for Corrrection (Closure) [f map in UROM paper]

# first R approxoimation of modal coefficients (\hat{a}) as input 
# R corrections (c) at same time as ouput

# Removing old models
model_name = 'LSTM Models/UROM_closure.h5'
if os.path.isfile(model_name):
   os.remove(model_name)


ahat = np.zeros((nc,ns+1,R+1))
c = np.zeros((nc,ns+1,R))

for p in range(nc):
    ahat[p,:,0] =  Re[p]

    #####################################
    # Galerkin projection RHS correction
    #####################################
    b_l = np.zeros((R,R,nc))
    b_nl = np.zeros((R,R,R,nc))
    linear_phi = np.zeros((nx*ny,R,nc))
    nonlinear_phi = np.zeros((nx*ny,R,nc))
    
    # linear term   
    for i in range(R):
        phi_w = np.reshape(Phiw[:,i,p],[nx,ny])
        
        lin_term = linear_term(nx,ny,dx,dy,Re[p],phi_w)
        linear_phi[:,i,p] = np.reshape(lin_term,nx*ny)

    for k in range(R):
        for i in range(R):
            b_l[i,k,p] = np.dot(linear_phi[:,i,p].T , Phiw[:,k,p]) 
                       
    # nonlinear term 
    for i in range(R):
        phi_w = np.reshape(Phiw[:,i,p],[nx,ny])
        for j in range(R):  
            phi_s = np.reshape(Phis[:,j,p],[nx,ny])
            nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
            jacobian_phi = np.reshape(nonlin_term,nx*ny)
            for k in range(R):    
                b_nl[i,j,k,p] = np.dot(jacobian_phi.T, Phiw[:,k,p]) 
                
    ahat[p,:3,1:] = aTrain[p,:3,:R]
    for k in range(3,ns+1): # 3 because we are using Adam-Bashforth   
        r1 = rhs(R, b_l[:,:,p], b_nl[:,:,:,p], aTrain[p,k-1,:R])
        r2 = rhs(R, b_l[:,:,p], b_nl[:,:,:,p], aTrain[p,k-2,:R])
        r3 = rhs(R, b_l[:,:,p], b_nl[:,:,:,p], aTrain[p,k-3,:R])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        ahat[p,k,1:] = aTrain[p,k-1,:R] + dt*temp 
        
        c[p,k,:] = aTrain[p,k,:R] - ahat[p,k,1:]
        
        
# Create training data for LSTM: xtrain is input, ytrain is output
# Stacking data
for p in range(nc):
    xt, yt = create_training_data_lstm1(ahat[p,:,:], c[p,:,:], ns+1, R, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

#% Scaling data
m,n = ytrain.shape #m is number of training samples, n is number of output features [i.e., n=R]
scalerOut1 = MinMaxScaler(feature_range=(-1,1))
scalerOut1 = scalerOut1.fit(ytrain)
ytrain = scalerOut1.transform(ytrain)


temp = ahat.reshape(-1,R+1)
scalerIn1 = MinMaxScaler(feature_range=(-1,1))
scalerIn1 = scalerIn1.fit(temp)
for i in range(m):
    xtrain[i,:,:] = scalerIn1.transform(xtrain[i,:,:])


# Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

# create the LSTM architecture
model1 = Sequential()
model1.add(LSTM(80, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model1.add(LSTM(80, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model1.add(LSTM(80, input_shape=(lookback, R+1), activation='tanh'))
model1.add(Dense(R))

# compile model
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# run the model
history = model1.fit(xtrain, ytrain, epochs=200, batch_size=64, validation_split=0.2)

# evaluate the model
scores = model1.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'LSTM Models/loss1.png'
plt.savefig(filename, dpi = 200)
plt.show()

# Save the model
model1.save(model_name)

# Save the scales
filename = 'LSTM Models/UROM_input_scaler1.save'
joblib.dump(scalerIn1,filename) 
filename = 'LSTM Models/UROM_output_scaler1.save'
joblib.dump(scalerOut1,filename) 
     

#%% Second LSTM [Superresolution] 

# 2- LSTM for super-resolution [g map in UROM paper]

# first R (true/corrected) modal coefficients (a) as input 
# next (Q-R) modal coefficients (a) at same time as ouput

# Removing old models
model_name = 'LSTM Models/UROM_superresolution.h5'
if os.path.isfile(model_name):
   os.remove(model_name)
    
# Stacking data
a = np.zeros((nc,ns+1,Q+1))
for p in range(nc):
    a[p,:,0] = Re[p]
    a[p,:,1:] = aTrain[p,:,:]
   

# Create training data for LSTM
for p in range(nc):
    xt, yt = create_training_data_lstm2(a[p,:,:], ns+1, R, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

# Scaling data
m,n = ytrain.shape # m is number of training samples, n is number of output features [i.e., n=Q-R]
scalerOut2 = MinMaxScaler(feature_range=(-1,1))
scalerOut2 = scalerOut2.fit(ytrain)
ytrain = scalerOut2.transform(ytrain)

temp = a.reshape(-1,Q+1)
scalerIn2 = MinMaxScaler(feature_range=(-1,1))
scalerIn2 = scalerIn2.fit(temp[:-1,:R+1])
for i in range(m):
    xtrain[i,:,:] = scalerIn2.transform(xtrain[i,:,:])


# Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

# create the LSTM architecture
model2 = Sequential()
model2.add(LSTM(80, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model2.add(LSTM(80, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model2.add(LSTM(80, input_shape=(lookback, R+1), activation='tanh'))
model2.add(Dense(Q-R))

# compile model
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# run the model
history = model2.fit(xtrain, ytrain, epochs=200, batch_size=64, validation_split=0.3)

# evaluate the model
scores = model2.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'LSTM Models/loss2.png'
plt.savefig(filename, dpi = 200)
plt.show()


# Save the model
model2.save(model_name)

# Save the scales
filename = 'LSTM Models/UROM_input_scaler2.save'
joblib.dump(scalerIn2,filename) 
filename = 'LSTM Models/UROM_output_scaler2.save'
joblib.dump(scalerOut2,filename) 


#%% NIROM Training

# first Q modal coefficients at time n as input 
# first Q modal coefficients at time n+1 as ouput

# Removing old models
model_name = 'LSTM Models/NIROM.h5'
if os.path.isfile(model_name):
   os.remove(model_name)
   
# Stacking data
a = np.zeros((nc,ns+1,Q+1))

for p in range(nc):
    a[p,:,0] =  Re[p]
    a[p,:,1:] = aTrain[p,:,:Q]
    
# Create training data for LSTM
# Stacking data
for p in range(nc):
    xt, yt = create_training_data_lstm3(a[p,:,:], ns+1, Q, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

# Scaling data
temp = a.reshape(-1,Q+1)
scalerIn3 = MinMaxScaler(feature_range=(-1,1))
scalerIn3 = scalerIn3.fit(temp[:-1,:])
m,n = ytrain.shape # m is number of training samples, n is number of output features [i.e., n=Q]
for i in range(m):
    xtrain[i,:,:] = scalerIn3.transform(xtrain[i,:,:])

scalerOut3 = MinMaxScaler(feature_range=(-1,1))
scalerOut3 = scalerOut3.fit(ytrain)
ytrain = scalerOut3.transform(ytrain)

# Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]
    
# create the LSTM architecture
model3 = Sequential()
model3.add(LSTM(80, input_shape=(lookback, Q+1), return_sequences=True, activation='tanh'))
model3.add(LSTM(80, input_shape=(lookback, Q+1), return_sequences=True, activation='tanh'))
model3.add(LSTM(80, input_shape=(lookback, Q+1), activation='tanh'))
model3.add(Dense(Q))

# compile model
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# run the model
history = model3.fit(xtrain, ytrain, epochs=200, batch_size=64, validation_split=0.2)

# evaluate the model
scores = model3.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model3.metrics_names[1], scores[1]*100))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'LSTM Models/loss3.png'
plt.savefig(filename, dpi = 200)
plt.show()

# Save the model
model3.save(model_name)

# Save the scales
filename = 'LSTM Models/NIROM_input_scaler.save'
joblib.dump(scalerIn3,filename) 
filename = 'LSTM Models/NIROM_output_scaler.save'
joblib.dump(scalerOut3,filename) 


