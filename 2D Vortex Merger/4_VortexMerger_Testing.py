# -*- coding: utf-8 -*-
"""
Testing UROM, GROM, NIROM on the 2D vortex merger problem for the paper: 
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
       
        temp = LA.multi_dot( [(Phi[:,:,i] - LA.multi_dot([Phi0, Phi0H, Phi[:,:,i] ]) ) \
                                 , LA.inv( Phi0H.dot(Phi[:,:,i]))] )       
        U, S, Vh = LA.svd(temp, full_matrices=False) # reduced SVD
        S = np.diag(S)
        Gamma[:,:,i] = LA.multi_dot([U,np.arctan(S),Vh])

    alpha = np.ones(nc)
    GammaL = np.zeros((nx,nr))
    #% Lagrange Interpolation
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-p[j])/(p[i]-p[j])
    for i in range(nc):
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
R = 4      #number of modes 
Q = 4*R #number of modes for 'super-resolution' [Hybridization]
Re_start = 200.0
Re_final = 800.0
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
dt = 1e-1 #timestep for ROM
tm = 20.0

lookback = 3 #Number of lookbacks

ReTest = np.array([500,1000]) #testing Reynolds number
nuTest = 1/ReTest
ncTest = len(ReTest)

#%%
print('Loading data')    
  
Phiw = np.load('./Basis/wBasis_Train.npy')
Phis = np.load('./Basis/sBasis_Train.npy')
aTrain = np.load('./Basis/Coeff_Train.npy')


#%% Testing

pref = np.array([2,3]) # #Reference case: 2 for ReTest=500, 3 for ReTest=1000

for p in range(ncTest):
    
    print('Testing for Re= '+str(ReTest[p]) )
    
    uTest = np.zeros((nx*ny, ns+1))
    for n in range(ns+1):
       file_input = "./snapshots/Re_"+str(int(ReTest[p]))+"/w_"+str(int(n))+ ".csv"
       w = np.genfromtxt(file_input, delimiter=',')
       w1 = w[1:nx+1,1:ny+1]
       uTest[:,n] = np.reshape(w1,(nx)*(ny)) 
        
    # Basis Interpolation
    print('Performing Grassmann Interpolation...')
    PhiwTest = GrassInt(Phiw,pref[p],nu,nuTest[p])
    
    aTest = PODproj(uTest,PhiwTest)
    # sign correction
    PhiwTest = PhiwTest/np.sign(aTest[0,:])
    aTest = aTest/np.sign(aTest[0,:])
    
    
    PhisTest = np.zeros((nx*ny,Q))
    for i in range(Q):
        phi_w = np.reshape(PhiwTest[:,i],[nx,ny])
        phi_s = fpsi(nx, ny, dx, dy, -phi_w)    
        PhisTest[:,i] = np.reshape(phi_s,(nx)*(ny))
    
    
    # Galerkin ROM (GROM) [Fully Intrusive]
    # GROM(4)
    print('Solving GROM(4)...')
    ###############################
    # Galerkin projection with R 
    ###############################
    b_l4 = np.zeros((R,R))
    b_nl4 = np.zeros((R,R,R))
    linear_phi = np.zeros((nx*ny,R))
    nonlinear_phi = np.zeros((nx*ny,R))
 
    for k in range(R):
        phi_w = np.reshape(PhiwTest[:,k],[nx,ny])
        lin_term = linear_term(nx,ny,dx,dy,ReTest[p],phi_w)
        linear_phi[:,k] = np.reshape(lin_term,nx*ny)
    
    for k in range(R):
        for i in range(R):
            b_l4[i,k] = np.dot(linear_phi[:,i].T , PhiwTest[:,k]) 
                       
    for i in range(R):
        phi_w = np.reshape(PhiwTest[:,i],[nx,ny])
        for j in range(R):  
            phi_s = np.reshape(PhisTest[:,j],[nx,ny])
            nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
            jacobian_phi = np.reshape(nonlin_term,nx*ny)
            for k in range(R):    
                b_nl4[i,j,k] = np.dot(jacobian_phi.T, PhiwTest[:,k]) 
           
    # solving ROM             
    aGP4 = np.zeros((ns+1,R))
    aGP4[0,:] = aTest[0,:R]
    aGP4[1,:] = aTest[1,:R]
    aGP4[2,:] = aTest[2,:R]
    
    for k in range(3,ns+1):
        r1 = rhs(R, b_l4[:,:], b_nl4[:,:,:], aGP4[k-1,:])
        r2 = rhs(R, b_l4[:,:], b_nl4[:,:,:], aGP4[k-2,:])
        r3 = rhs(R, b_l4[:,:], b_nl4[:,:,:], aGP4[k-3,:])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aGP4[k,:] = aGP4[k-1,:] + dt*temp 
            
    
    # GROM(16)    
    print('Solving GROM(16)...')
    #############################
    # Galerkin projection with Q
    #############################

    b_l16 = np.zeros((Q,Q))
    b_nl16 = np.zeros((Q,Q,Q))
    linear_phi = np.zeros((nx*ny,Q))
    nonlinear_phi = np.zeros((nx*ny,Q))
 
    for k in range(Q):
        phi_w = np.reshape(PhiwTest[:,k],[nx,ny])
        lin_term = linear_term(nx,ny,dx,dy,ReTest[p],phi_w)
        linear_phi[:,k] = np.reshape(lin_term,nx*ny)
    
    for k in range(Q):
        for i in range(Q):
            b_l16[i,k] = np.dot(linear_phi[:,i].T , PhiwTest[:,k]) 
                       
    for i in range(Q):
        phi_w = np.reshape(PhiwTest[:,i],[nx,ny])
        for j in range(Q):  
            phi_s = np.reshape(PhisTest[:,j],[nx,ny])
            nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
            jacobian_phi = np.reshape(nonlin_term,nx*ny)
            for k in range(Q):    
                b_nl16[i,j,k] = np.dot(jacobian_phi.T, PhiwTest[:,k]) 
           
    # solving ROM             
    aGP16 = np.zeros((ns+1,Q))
    aGP16[0,:] = aTest[0,:Q]
    aGP16[1,:] = aTest[1,:Q]
    aGP16[2,:] = aTest[2,:Q]
    
    for k in range(3,ns+1):
        r1 = rhs(Q, b_l16[:,:], b_nl16[:,:,:], aGP16[k-1,:])
        r2 = rhs(Q, b_l16[:,:], b_nl16[:,:,:], aGP16[k-2,:])
        r3 = rhs(Q, b_l16[:,:], b_nl16[:,:,:], aGP16[k-3,:])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aGP16[k,:] = aGP16[k-1,:] + dt*temp 
        


    # UROM 
    print('Solving UROM...')
    ################
    # Uplifted ROM 
    ################
    # For the first R modes
    # Load model
    model_name = 'LSTM Models/UROM_closure.h5'
    model1 = load_model(model_name)  
    
    # load scales
    filename = 'LSTM Models/UROM_input_scaler1.save'
    scalerIn1 = joblib.load(filename)  
    filename = 'LSTM Models/UROM_output_scaler1.save'
    scalerOut1 = joblib.load(filename)  
           
    # testing
    testing_set = np.hstack((ReTest[p]*np.ones((ns+1,1)), aTest[:,:R]))
    m,n = testing_set.shape
    xtest = np.zeros((1,lookback,R+1))
    aUROM = np.zeros((ns+1,Q))
    
    # Initializing
    for i in range(lookback):
        temp = testing_set[i,:R+1]
        temp = temp.reshape(1,-1)
        xtest[0,i,:]  = scalerIn1.transform(temp) 
        aUROM[i, :] = aTest[i,:]
        
    # Prediction    
    for i in range(lookback,ns+1):
        ytest = model1.predict(xtest)
        ytest = scalerOut1.inverse_transform(ytest) # rescale  
        
        # solving ROM             
        r1 = rhs(R, b_l4, b_nl4, aUROM[i-1,:]) 
        r2 = rhs(R, b_l4, b_nl4, aUROM[i-2,:])
        r3 = rhs(R, b_l4, b_nl4, aUROM[i-3,:])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aUROM[i,:R] = aUROM[i-1,:R] + dt*temp + ytest
          
        # Update xtest
        for k in range(lookback-1):
            xtest[0,k,:] = xtest[0,k+1,:]
        temp = np.hstack(( ReTest[p]*np.ones((1,1)) ,aUROM[i,:R].reshape((1,-1)) ))
        xtest[0,lookback-1,:] = scalerIn1.transform(temp) 
        
    
    # For the next Q-R modes   
    testing_set = np.hstack((ReTest[p]*np.ones((ns+1,1)), aTest))
    m,n = testing_set.shape
    xtest = np.zeros((1,lookback,R+1))
    
    # Load model
    model_name = 'LSTM Models/UROM_superresolution.h5'
    model2 = load_model(model_name)  
    
    # load scales
    filename = 'LSTM Models/UROM_input_scaler2.save'
    scalerIn2 = joblib.load(filename)  
    filename = 'LSTM Models/UROM_output_scaler2.save'
    scalerOut2 = joblib.load(filename)  
    
    # Initializing
    for i in range(lookback):
        temp = testing_set[i,:R+1]
        temp = temp.reshape(1,-1)
        xtest[0,i,:]  = scalerIn2.transform(temp) 
        aUROM[i, R:] = testing_set[i,R+1:] 
    
    # Prediction   
    for i in range(lookback,ns+1):
        
        # update xtest
        for k in range(lookback-1):
            xtest[0,k,:] = xtest[0,k+1,:]
        temp =  np.hstack((ReTest[p]*np.ones((1,1)) , aUROM[i,:R].reshape((1,-1)) ))  
        xtest[0,lookback-1,:] = scalerIn2.transform(temp) 
        
        ytest = model2.predict(xtest)
        ytest = scalerOut2.inverse_transform(ytest) # rescale
        aUROM[i, R:] = ytest  
    

    # NIROM
    print('Solving NIROM...')
    ####################
    # Nonintrusive ROM
    ####################
    # Load model
    model_name = 'LSTM Models/NIROM.h5'
    model3 = load_model(model_name)  
    
    # load scales
    filename = 'LSTM Models/NIROM_input_scaler.save'
    scalerIn3 = joblib.load(filename)  
    filename = 'LSTM Models/NIROM_output_scaler.save'
    scalerOut3 = joblib.load(filename)  
    
    # testing
    testing_set = np.hstack((ReTest[p]*np.ones((ns+1,1)), aTest[:,:Q]))
    m,n = testing_set.shape
    xtest = np.zeros((1,lookback,Q+1))
    aNIROM = np.zeros((ns+1,Q))
    
    # Initializing
    for i in range(lookback):
        temp = testing_set[i,:Q+1]
        temp = temp.reshape(1,-1)
        xtest[0,i,:]  = scalerIn3.transform(temp) 
        aNIROM[i,:] = aTest[i,:]
        
    # Prediction
    for i in range(lookback,ns+1):
        ytest = model3.predict(xtest)
        
        ytest = scalerOut3.inverse_transform(ytest) # rescale  
        aNIROM[i,:] =  ytest
          
        # Update xtest
        for k in range(lookback-1):
            xtest[0,k,:] = xtest[0,k+1,:]
        temp = np.hstack(( ReTest[p]*np.ones((1,1)) ,aNIROM[i,:].reshape((1,-1)) ))
        xtest[0,lookback-1,:] = scalerIn3.transform(temp) 
        


    wFOM = uTest[:,-1] # save last time step

    # Reconstruction
    wPOD = PODrec(aTest[-1,:],PhiwTest[:,:Q])
    wGP4 = PODrec(aGP4[-1,:],PhiwTest[:,:R])
    wGP16 = PODrec(aGP16[-1,:],PhiwTest[:,:Q])
    wUROM = PODrec(aUROM[-1,:],PhiwTest[:,:Q])
    wNIROM = PODrec(aNIROM[-1,:],PhiwTest[:,:Q])
    
    # apply boundary conditions    
    wFOM = np.reshape(wFOM,[nx,ny])
    wFOM = pbc(wFOM)
    
    wPOD = np.reshape(wPOD,[nx,ny])
    wPOD = pbc(wPOD)
    
    wGP4 = np.reshape(wGP4,[nx,ny])
    wGP4 = pbc(wGP4)
    
    wGP16 = np.reshape(wGP16,[nx,ny])
    wGP16 = pbc(wGP16)

    wUROM = np.reshape(wUROM,[nx,ny])
    wUROM = pbc(wUROM)
    
    wNIROM = np.reshape(wNIROM,[nx,ny])
    wNIROM = pbc(wNIROM)

    # save results
    # create folder
    if os.path.isdir("./Results"):
        print('Results folder already exists')
    else: 
        print('Creating results folder')
        os.makedirs("./Results")
     
    print('Saving data')      
    np.save('./Results/aTest_Re'+str(ReTest[p])+'.npy',aTest)
    np.save('./Results/aGP4_Re'+str(ReTest[p])+'.npy',aGP4)
    np.save('./Results/aGP16_Re'+str(ReTest[p])+'.npy',aGP16)
    np.save('./Results/aUROM_Re'+str(ReTest[p])+'.npy',aUROM)
    np.save('./Results/aNIROM_Re'+str(ReTest[p])+'.npy',aNIROM)
            
    np.save('./Results/wFOM_Re'+str(ReTest[p])+'.npy',wFOM)
    np.save('./Results/wPOD_Re'+str(ReTest[p])+'.npy',wPOD)
    np.save('./Results/wGP4_Re'+str(ReTest[p])+'.npy',wGP4)
    np.save('./Results/wGP16_Re'+str(ReTest[p])+'.npy',wGP16)
    np.save('./Results/wUROM_Re'+str(ReTest[p])+'.npy',wUROM)
    np.save('./Results/wNIROM_Re'+str(ReTest[p])+'.npy',wNIROM)



