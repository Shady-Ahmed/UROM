# -*- coding: utf-8 -*-
"""
Testing UROM, GROM, NIROM on the 1D Burgers problem for the paper: 
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

R = 4       #number of modes
Q = 4*R    #number of modes for 'super-resolution' [Hybridization]

lookback = 3 #number of lookbacks


#%% load POD modes and coefficients
aTrain = np.load('./Data/Coeff_Train.npy')
Phi = np.load('./Data/Basis_Train.npy')
uTest = np.load('./Data/FOM_Test.npy')

   
#%% Testing

pref = np.array([2,3]) # #Reference case: 2 for ReTest=500, 3 for ReTest=1000

for p in range(ncTest):
    
    print('Testing for Re= '+str(ReTest[p]) )
    
    # Basis Interpolation
    print('Performing Grassmann Interpolation...')
    PhiTest = GrassInt(Phi,pref[p],nu,nuTest[p])
    
    aTest = PODproj(uTest[:,:,p],PhiTest)
    PhiTest = PhiTest/np.sign(aTest[0,:])
    aTest = aTest/np.sign(aTest[0,:])
    
    # Galerkin ROM (GROM) [Fully Intrusive]
    # GROM(4)
    print('Solving GROM(4)...')
    ###############################
    # Galerkin projection with R 
    ###############################
    b_l4 = np.zeros((R,R))
    b_nl4 = np.zeros((R,R,R))
    Phid = np.zeros((nx+1,R))
    Phidd = np.zeros((nx+1,R))
    
    for i in range(R):
        Phid[:,i] = pade4d(PhiTest[:,i],dx,nx)
        Phidd[:,i] = pade4dd(PhiTest[:,i],dx,nx)
    
    # linear term   
    for k in range(R):
        for i in range(R):
            b_l4[i,k] = nuTest[p]*np.dot(Phidd[:,i].T , PhiTest[:,k]) 
                       
    # nonlinear term 
    for k in range(R):
        for j in range(R):
            for i in range(R):
                temp = PhiTest[:,i]*Phid[:,j]
                b_nl4[i,j,k] = - np.dot( temp.T, PhiTest[:,k] ) 
    
    # solving ROM             
    aGP4 = np.zeros((ns+1,R))
    aGP4[0,:] = aTest[0,:R]
    aGP4[1,:] = aTest[1,:R]
    aGP4[2,:] = aTest[2,:R]
    
    for k in range(3,ns+1):
        r1 = rhs(R, b_l4, b_nl4, aGP4[k-1,:])
        r2 = rhs(R, b_l4, b_nl4, aGP4[k-2,:])
        r3 = rhs(R, b_l4, b_nl4, aGP4[k-3,:])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aGP4[k,:] = aGP4[k-1,:] + dt*temp 
    
    
    # GROM(16)    
    print('Solving GROM(16)...')
    #############################
    # Galerkin projection with Q
    #############################
    b_l16 = np.zeros((Q,Q))
    b_nl16 = np.zeros((Q,Q,Q))
    Phid = np.zeros((nx+1,Q))
    Phidd = np.zeros((nx+1,Q))
    
    for i in range(Q):
        Phid[:,i] = pade4d(PhiTest[:,i],dx,nx)
        Phidd[:,i] = pade4dd(PhiTest[:,i],dx,nx)
    
    # linear term   
    for k in range(Q):
        for i in range(Q):
            b_l16[i,k] = nuTest[p]*np.dot(Phidd[:,i].T , PhiTest[:,k]) 
                       
    # nonlinear term 
    for k in range(Q):
        for j in range(Q):
            for i in range(Q):
                temp = PhiTest[:,i]*Phid[:,j]
                b_nl16[i,j,k] = - np.dot( temp.T, PhiTest[:,k] ) 
    
    # solving ROM             
    aGP16 = np.zeros((ns+1,Q))
    aGP16[0,:] = aTest[0,:Q]
    aGP16[1,:] = aTest[1,:Q]
    aGP16[2,:] = aTest[2,:Q]
    
    for k in range(3,ns+1):
        r1 = rhs(Q, b_l16, b_nl16, aGP16[k-1,:])
        r2 = rhs(Q, b_l16, b_nl16, aGP16[k-2,:])
        r3 = rhs(Q, b_l16, b_nl16, aGP16[k-3,:])
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
        
    
    
    # Reconstruction
    uFOM = uTest[:,:,p]    
    uPOD = PODrec(aTest,PhiTest[:,:Q])
    uGP4 = PODrec(aGP4,PhiTest[:,:R])
    uGP16 = PODrec(aGP16,PhiTest[:,:Q])
    uUROM = PODrec(aUROM,PhiTest[:,:Q])
    uNIROM = PODrec(aNIROM,PhiTest[:,:Q])

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
            
    np.save('./Results/uFOM_Re'+str(ReTest[p])+'.npy',uFOM)
    np.save('./Results/uPOD_Re'+str(ReTest[p])+'.npy',uPOD)
    np.save('./Results/uGP4_Re'+str(ReTest[p])+'.npy',uGP4)
    np.save('./Results/uGP16_Re'+str(ReTest[p])+'.npy',uGP16)
    np.save('./Results/uUROM_Re'+str(ReTest[p])+'.npy',uUROM)
    np.save('./Results/uNIROM_Re'+str(ReTest[p])+'.npy',uNIROM)
