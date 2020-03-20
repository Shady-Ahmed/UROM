# -*- coding: utf-8 -*-
"""
POD basis generation for 2D vortex merger problem for the paper: 
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

from numpy.random import seed
seed(1)
import os

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


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



#%% Main program:
# Inputs
nx = 256    #spatial grid number
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
dt = 1e-1 #timestep between snapshots
tm = 20.0

#%% Data generation for training
x = np.linspace(0, lx, nx+1)
y = np.linspace(0, ly, ny+1)
t = np.linspace(0, tm, ns+1)

um = np.zeros(((nx)*(ny), ns+1, nc))
up = np.zeros(((nx)*(ny), ns+1, nc))
u = np.zeros(((nx)*(ny), ns+1, nc))

for p in range(0,nc):
    for n in range(0,ns+1):
        file_input = "./snapshots/Re_"+str(int(Re[p]))+"/w_"+str(int(n))+ ".csv"
        w = np.genfromtxt(file_input, delimiter=',')
        w1 = w[1:nx+1,1:ny+1]
        u[:,n,p] = np.reshape(w1,(nx)*(ny)) 

#%% POD basis computation
Phiw = np.zeros((nx*ny,Q,nc))
Phis = np.zeros((nx*ny,Q,nc))             
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

print('Computing POD basis for vorticity ...')
for p in range(nc):
    Phiw[:,:,p], L[:,p], RIC[p]  = POD(u[:,:,p], Q) 

#%% Calculating true POD coefficients 
aTrain = np.zeros((nc,ns+1,Q))
print('Computing true POD coefficients...')
for p in range(nc):
    aTrain[p,:,:] = PODproj(u[:,:,p],Phiw[:,:,p])
    #Unifying signs for proper training and interpolation
    Phiw[:,:,p] = Phiw[:,:,p]/np.sign(aTrain[p,0,:])
    aTrain[p,:,:] = aTrain[p,:,:]/np.sign(aTrain[p,0,:])

#%%    
print('Computing POD basis for streamfunction ...')
for p in range(nc):
    for i in range(Q):
        phi_w = np.reshape(Phiw[:,i,p],[nx,ny])
        phi_s = fpsi(nx, ny, dx, dy, -phi_w)
        Phis[:,i,p] = np.reshape(phi_s,(nx)*(ny))
        

#%% Saving data

# create folder
if os.path.isdir("./Basis"):
    print('POD bsis folder already exists')
else: 
    print('Creating POD basis folder')
    os.makedirs("./Basis")
 
print('Saving POD data')      
np.save('./Basis/wBasis_Train.npy',Phiw)
np.save('./Basis/sBasis_Train.npy',Phis)
np.save('./Basis/Coeff_Train.npy',aTrain)


