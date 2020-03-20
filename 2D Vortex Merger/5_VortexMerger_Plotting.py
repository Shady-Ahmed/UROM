# -*- coding: utf-8 -*-
"""
Plotting results for reduced order modeling of 1D Burgers problem for the paper: 
 "A long short-term memory embedding for hybrid uplifted reduced order models",
 Physica D: Nonlinear Phenomena, 2020.
 authors: Shady E. Ahmed, Omer San, Adil Rasheed, Traian Iliescu

 Last update: 03_15_2020
 Contact: shady.ahmed@okstate.edu
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os


font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}	
#plt.rc('font', **font)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

rc('font',**font)
rc('text', usetex=True)

#%%
nx = 256   #spatial grid number
ny = 256
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
x = np.linspace(0, lx, nx+1)
y = np.linspace(0, ly, ny+1)

ns = 200
R = 4
Q = 4*R

ReTest = np.array([500,1000]) #testing Reynolds number
nuTest = 1/ReTest
ncTest = len(ReTest)

tm = 20.0
t = np.linspace(0, tm, ns+1)

#%% Reading results

aTest = np.zeros((ns+1,Q,ncTest))   
aGP4 = np.zeros((ns+1,R,ncTest))         
aGP16 = np.zeros((ns+1,Q,ncTest))         
aUROM = np.zeros((ns+1,Q,ncTest))         
aNIROM = np.zeros((ns+1,Q,ncTest))         
     
wFOM = np.zeros((nx+1,ny+1,ncTest))
wPOD = np.zeros((nx+1,ny+1,ncTest))
wGP4 = np.zeros((nx+1,ny+1,ncTest))
wGP16 = np.zeros((nx+1,ny+1,ncTest))
wUROM = np.zeros((nx+1,ny+1,ncTest))
wNIROM = np.zeros((nx+1,ny+1,ncTest))

for p in range(ncTest):
    aTest[:,:,p] = np.load('./Results/aTest_Re'+str(ReTest[p])+'.npy')
    aGP4[:,:,p] = np.load('./Results/aGP4_Re'+str(ReTest[p])+'.npy')
    aGP16[:,:,p] = np.load('./Results/aGP16_Re'+str(ReTest[p])+'.npy')
    aUROM[:,:,p] = np.load('./Results/aUROM_Re'+str(ReTest[p])+'.npy')
    aNIROM[:,:,p] = np.load('./Results/aNIROM_Re'+str(ReTest[p])+'.npy')
            
    wFOM[:,:,p] = np.load('./Results/wFOM_Re'+str(ReTest[p])+'.npy')
    wPOD[:,:,p] = np.load('./Results/wPOD_Re'+str(ReTest[p])+'.npy')
    wGP4[:,:,p] = np.load('./Results/wGP4_Re'+str(ReTest[p])+'.npy')
    wGP16[:,:,p] = np.load('./Results/wGP16_Re'+str(ReTest[p])+'.npy')
    wUROM[:,:,p] = np.load('./Results/wUROM_Re'+str(ReTest[p])+'.npy')
    wNIROM[:,:,p] = np.load('./Results/wNIROM_Re'+str(ReTest[p])+'.npy')


# create plots folder
if os.path.isdir("./Plots"):
    print('Plots folder already exists')
else: 
    print('Creating plots folder')
    os.makedirs("./Plots")

#%% Timeseries plots

for p in range(ncTest):
    
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9,6))
    ax = ax.flat
    for k in range(R):
        ax[k].plot(t,aTest[:,k,p], label=r'True', color = 'C0')
        ax[k].plot(t,aUROM[:,k,p], label=r'UROM',linestyle='-.', color = 'C1')
        ax[k].plot(t,aGP16[:,k,p], label=r'GROM('+str(Q)+')',linestyle='--', color = 'C2')
        ax[k].plot(t,aGP4[:,k,p], label=r'GROM('+str(R)+')',linestyle=':', color = 'C3')
        ax[k].plot(t,aNIROM[:,k,p], label=r'NIROM',linestyle=':', color = 'C4')
        ax[k].set_xlabel(r'$t$',fontsize=14)
        ax[k].set_ylabel(r'$a_{'+str(k+1) +'}(t)$',fontsize=14)
        
    ax[0].legend(loc="center", bbox_to_anchor=(1.16,1.4),ncol =5,fontsize=12)
    fig.subplots_adjust(bottom=0.15,hspace=0.7, wspace=0.35)
    plt.savefig('Plots/vortex_merger_a_Re' + str(ReTest[p]) + \
                '_R=' + str(R) + '_Q=' + str(Q) + '.png',\
                dpi = 500, bbox_inches = 'tight')
    fig.show()
 
#%% contour plots
    
nlvls = 30
ctick = np.linspace(0, 0.74, 11, endpoint=True)
nticks = [0,2,4,6]


for p in range(ncTest):

    fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(9,6))
    ax = ax.flat
    
    ax0 = ax[0].contour(x,y,wFOM[:,:,p].T, nlvls, cmap = 'jet', linewidths=0.6)
    ax0.set_clim([0, 0.74])
    ax[0].set_title(r'FOM')

    ax1 = ax[1].contour(x,y,wPOD[:,:,p].T, nlvls, cmap = 'jet', linewidths=0.6)
    ax1.set_clim([0, 0.74])
    ax[1].set_title(r'True')

    ax2 = ax[2].contour(x,y,wUROM[:,:,p].T, nlvls, cmap = 'jet', linewidths=0.6)
    ax2.set_clim([0, 0.74])
    ax[2].set_title(r'UROM')

    ax3 = ax[3].contour(x,y,wGP16[:,:,p].T, nlvls, cmap = 'jet', linewidths=0.6)
    ax3.set_clim([0, 0.74])
    ax[3].set_title(r'GROM('+str(Q)+')')

    ax4 = ax[4].contour(x,y,wGP4[:,:,p].T, nlvls, cmap = 'jet', linewidths=0.6)
    ax4.set_clim([0, 0.74])
    ax[4].set_title(r'GROM('+str(R)+')')
    
    ax5 = ax[5].contour(x,y,wNIROM[:,:,p].T, nlvls, cmap = 'jet', linewidths=0.6)
    ax5.set_clim([0, 0.74])
    ax[5].set_title(r'NIROM')
    
    for i in range(6):
        ax[i].set_xlabel(r'$x$',fontsize=14)
        ax[i].set_ylabel(r'$y$',fontsize=14)
        ax[i].set_xticks(nticks)
        ax[i].set_yticks(nticks)
        
    fig.subplots_adjust(right=0.9,hspace=0.7, wspace=0.4)
    
    cbar_ax = fig.add_axes([0.95, 0.15, 0.04, 0.7])
    fig.colorbar(ax0, cax = cbar_ax,  orientation='vertical')
    
    plt.savefig('plots/vortex_merger_final_Re' + str(ReTest[p]) + \
                '_R=' + str(R) + '_Q=' + str(Q) + '.png',\
                dpi = 500, bbox_inches = 'tight')
    plt.show()
