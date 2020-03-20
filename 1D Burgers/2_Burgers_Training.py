# -*- coding: utf-8 -*-
"""
LSTM training for reduced order modeling of 1D Burgers problem for the paper: 
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

#%% load training (True) POD modes and coefficients
aTrain = np.load('./Data/Coeff_Train.npy')
Phi = np.load('./Data/Basis_Train.npy')

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
    ahat[p,:,0] =  Re[p] #augmenting inputs with control parameter value

    #####################################
    # Galerkin projection RHS correction
    #####################################
    b_l = np.zeros((R,R))
    b_nl = np.zeros((R,R,R))
    Phid = np.zeros((nx+1,R))
    Phidd = np.zeros((nx+1,R))
    
    for i in range(R):
        Phid[:,i] = pade4d(Phi[:,i,p],dx,nx)   #dPhi/dx
        Phidd[:,i] = pade4dd(Phi[:,i,p],dx,nx) #d^2Phi/dx^2
    
    # linear term   
    for k in range(R):
        for i in range(R):
            b_l[i,k] = nu[p]*np.dot(Phidd[:,i].T , Phi[:,k,p]) 
                       
    # nonlinear term 
    for k in range(R):
        for j in range(R):
            for i in range(R):
                temp = Phi[:,i,p]*Phid[:,j]
                b_nl[i,j,k] = - np.dot( temp.T, Phi[:,k,p] ) 
    
    ahat[p,:3,1:] = aTrain[p,:3,:R]      
    for k in range(3,ns+1): #Because we are using Adam-Bashforth
        r1 = rhs(R, b_l, b_nl, aTrain[p,k-1,:R])  
        r2 = rhs(R, b_l, b_nl, aTrain[p,k-2,:R])  
        r3 = rhs(R, b_l, b_nl, aTrain[p,k-3,:R])  
        temp = (23/12) * r1 - (16/12) * r2 + (5/12) * r3
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
model1.add(LSTM(60, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model1.add(LSTM(60, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model1.add(LSTM(60, input_shape=(lookback, R+1), activation='tanh'))
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

temp = a.reshape(-1,Q+1)
scalerIn2 = MinMaxScaler(feature_range=(-1,1))
scalerIn2 = scalerIn2.fit(temp[:-1,:R+1])
for i in range(m):
    xtrain[i,:,:] = scalerIn2.transform(xtrain[i,:,:])
ytrain = scalerOut2.transform(ytrain)


# Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

# create the LSTM architecture
model2 = Sequential()
#model.add(Dropout(0.2))
model2.add(LSTM(60, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model2.add(LSTM(60, input_shape=(lookback, R+1), return_sequences=True, activation='tanh'))
model2.add(LSTM(60, input_shape=(lookback, R+1), activation='tanh'))
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
