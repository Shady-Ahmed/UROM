# -*- coding: utf-8 -*-
"""
Snapshot and POD basis generation for 1D Burgers problem for the paper: 
 "A long short-term memory embedding for hybrid uplifted reduced order models",
 Physica D: Nonlinear Phenomena, 2020.
 authors: Shady E. Ahmed, Omer San, Adil Rasheed, Traian Iliescu

 Last update: 03_15_2020
 Contact: shady.ahmed@okstate.edu
"""

#%% Import libraries and functions
from Burgers_utilities import *


#%% Main program:
    
# Inputs
nx =  1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)



nc = 4      #number of control parameters [nu] (for training)
Re_start = 200.0 #lower limit for training Reynolds number
Re_final = 800.0 #higher limit for training Reynolds number
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation

ReTest = np.array([500,1000]) #testing Reynolds number
nuTest = 1/ReTest
ncTest = len(ReTest)

tm = 1      #maximum time
ns = 1000   #number of snapshot per each parameter value 
dt = tm/ns
t = np.linspace(0, tm, ns+1)

R = 4      #lower number of modes
Q = 4*R    #higher number of modes

#%% FOM snapshot generation for training

print('Computing FOM snapshots...')
uTrain = np.zeros((nx+1, ns+1, nc))
for p in range(0,nc):
    for n in range(0,ns+1):
        uTrain[:,n,p]=uexact(x,t[n],nu[p]) #snapshots from exact solution

#%% FOM snapshot generation for testing
        
uTest = np.zeros((nx+1, ns+1, ncTest))
for p in range(ncTest):
    for n in range(ns+1):
        uTest[:,n,p]=uexact(x,t[n],nuTest[p]) #snapshots from exact solution


#%% POD basis computation for training data

print('Computing POD basis...')
Phi = np.zeros((nx+1,Q,nc)) # POD modes     
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

for p in range(nc):
    Phi[:,:,p], L[:,p], RIC[p]  = POD(uTrain[:,:,p], Q) 
        
#%% Calculating true POD modal coefficients
aTrain = np.zeros((nc,ns+1,Q))
print('Computing true POD coefficients...')
for p in range(nc):
    aTrain[p,:,:] = PODproj(uTrain[:,:,p],Phi[:,:,p])
    #Unifying signs for proper training and interpolation
    Phi[:,:,p] = Phi[:,:,p]/np.sign(aTrain[p,0,:])
    aTrain[p,:,:] = aTrain[p,:,:]/np.sign(aTrain[p,0,:])


#%% Saving data

# create folder
if os.path.isdir("./Data"):
    print('Data folder already exists')
else: 
    print('Creating data folder')
    os.makedirs("./Data")
 
print('Saving data')      
np.save('./Data/FOM_Train.npy',uTrain)
np.save('./Data/FOM_Test.npy',uTest)
np.save('./Data/Basis_Train.npy',Phi)
np.save('./Data/Coeff_Train.npy',aTrain)

