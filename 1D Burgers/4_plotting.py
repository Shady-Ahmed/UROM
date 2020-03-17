# -*- coding: utf-8 -*-
"""
Plotting results for reduced order modeling of 1D Burgers problem for the paper: 
 "A long short-term memory embedding for hybrid uplifted reduced order models",
 Physica D: Nonlinear Phenomena, 2020.
 authors: Shady E. Ahmed, Omer San, Adil Rasheed, Traian Iliescu

 Last update: 03_15_2020
 Contact: shady.ahmed@okstate.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}	
#plt.rc('font', **font)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

rc('font',**font)
rc('text', usetex=True)

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


#%% Reading results

aTest = np.zeros((ns+1,Q,ncTest))   
aGP4 = np.zeros((ns+1,R,ncTest))         
aGP16 = np.zeros((ns+1,Q,ncTest))         
aUROM = np.zeros((ns+1,Q,ncTest))         
aNIROM = np.zeros((ns+1,Q,ncTest))         
     
uFOM = np.zeros((nx+1,ns+1,ncTest))
uPOD = np.zeros((nx+1,ns+1,ncTest))
uGP4 = np.zeros((nx+1,ns+1,ncTest))
uGP16 = np.zeros((nx+1,ns+1,ncTest))
uUROM = np.zeros((nx+1,ns+1,ncTest))
uNIROM = np.zeros((nx+1,ns+1,ncTest))

for p in range(ncTest):
    aTest[:,:,p] = np.load('./Results/aTest_Re'+str(ReTest[p])+'.npy')
    aGP4[:,:,p] = np.load('./Results/aGP4_Re'+str(ReTest[p])+'.npy')
    aGP16[:,:,p] = np.load('./Results/aGP16_Re'+str(ReTest[p])+'.npy')
    aUROM[:,:,p] = np.load('./Results/aUROM_Re'+str(ReTest[p])+'.npy')
    aNIROM[:,:,p] = np.load('./Results/aNIROM_Re'+str(ReTest[p])+'.npy')
            
    uFOM[:,:,p] = np.load('./Results/uFOM_Re'+str(ReTest[p])+'.npy')
    uPOD[:,:,p] = np.load('./Results/uPOD_Re'+str(ReTest[p])+'.npy')
    uGP4[:,:,p] = np.load('./Results/uGP4_Re'+str(ReTest[p])+'.npy')
    uGP16[:,:,p] = np.load('./Results/uGP16_Re'+str(ReTest[p])+'.npy')
    uUROM[:,:,p] = np.load('./Results/uUROM_Re'+str(ReTest[p])+'.npy')
    uNIROM[:,:,p] = np.load('./Results/uNIROM_Re'+str(ReTest[p])+'.npy')


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
    plt.savefig('Plots/burgers_a_Re' + str(ReTest[p]) + \
                '_R=' + str(R) + '_Q=' + str(Q) + '.png',\
                dpi = 500, bbox_inches = 'tight')
    fig.show()
    
#%% Final velocity plots
for p in range(ncTest):
    
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(13,6))
    ax = ax.flat
    ax[0].plot(x,uFOM[:,-1,p],label=r'FOM', color = 'black')
    ax[0].plot(x,uPOD[:,-1,p],label=r'True', color = 'C0')
    ax[0].plot(x,uUROM[:,-1,p],label=r'UROM',linestyle='-.', color = 'C1')
    ax[0].plot(x,uGP16[:,-1,p],label=r'GROM('+str(Q)+')',linestyle='--', color = 'C2')
    ax[0].plot(x,uGP4[:,-1,p],label=r'GROM('+str(R)+')',linestyle=':', color = 'C3')
    ax[0].plot(x,uNIROM[:,-1,p],label=r'NIROM',linestyle=':', color = 'C4')
    
    ax[0].set_xlabel(r'$x$',fontsize=16)
    ax[0].set_ylabel(r'$u(x)$',fontsize=16)
    ax[0].set_ylim(top=0.5)
    
    if p == 0:
        x1, x2, y1, y2 = 0.59, 0.69, 0.285, 0.335 # specify the limits
    elif p == 1:
        x1, x2, y1, y2 = 0.60, 0.71, 0.3, 0.35 # specify the limits
    
   
    import matplotlib.patches as patches
    # Create a Rectangle patch
    rect = patches.Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1.5,edgecolor='0.7',facecolor='0.95')
    
    # Add the patch to the Axes
    ax[0].add_patch(rect)
    
    #axins = zoomed_inset_axes(ax[0], 2.5, loc=3 ,bbox_to_anchor=(0.735,0.73), bbox_transform=ax[0].transAxes ) # zoom-factor: 2.5, location: upper-left
    ax[1].plot(x,uFOM[:,-1,p],label=r'FOM', color = 'black')
    ax[1].plot(x,uPOD[:,-1,p],label=r'True', color = 'C0')
    ax[1].plot(x,uUROM[:,-1,p],label=r'UROM',linestyle='-.', color = 'C1')
    ax[1].plot(x,uGP16[:,-1,p],label=r'GROM('+str(Q)+')',linestyle='--', color = 'C2')
    ax[1].plot(x,uGP4[:,-1,p],label=r'GROM('+str(R)+')',linestyle=':', color = 'C3')
    ax[1].plot(x,uNIROM[:,-1,p],label=r'NIROM',linestyle=':', color = 'C4')
    
    ax[1].set_xlim(x1, x2) # apply the x-limits
    ax[1].set_ylim(y1, y2) # apply the y-limits
    ax[1].set_xlabel(r'$x$',fontsize=16)
    ax[1].set_ylabel(r'$u(x)$',fontsize=16)
    
    ax[0].legend(loc="center", bbox_to_anchor=(1.1,1.1),ncol =6,fontsize=12)
    fig.subplots_adjust(bottom=0.15,hspace=0.5, wspace=0.25)
    
    plt.savefig('Plots/burgers_ufinal_Re' + str(ReTest[p]) + \
                '_R=' + str(R) + '_Q=' + str(Q) + '.png',\
                dpi = 500, bbox_inches = 'tight')
    plt.show()


#%% surface plots of spatio-temporal evolution

from mpl_toolkits.mplot3d import Axes3D  

X, Y = np.meshgrid(t, x)
r = 100 #change to 1 for high resolution, but it will take some time
c = 100 #change to 1 for high resolution, but it will take some time

for p in range(ncTest):
    fig = plt.figure(figsize=(14,7))
    
    ######### FOM #########
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    surf = ax.plot_surface(Y, X, uFOM[:,:,p], cmap='coolwarm',
                               linewidth=1, shade=False,antialiased=False,rstride=r,
                                cstride=c,rasterized=True)
    
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
    
    #fig.colorbar(surf,shrink=0.6, aspect=10)
    ax.set_title(r'FOM')
    ax.set_xticks([0,0.5,1])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_yticks([0,0.5,1.0])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_zlim([0,0.4])
    ax.set_zticks([0,0.2,0.4])
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.tick_params(direction='out', pad=0)
    ax.set_xlabel('$x$', labelpad=-4)
    ax.set_ylabel('$t$', labelpad=-2)
    ax.set_zlabel('$u$', labelpad=-4)
    
    ######### True POD #########
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    surf = ax.plot_surface(Y, X, uPOD[:,:,p], cmap='coolwarm',
                               linewidth=1, shade=False, antialiased=False,rstride=r,
                                cstride=c,rasterized=True)
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
      
    #fig.colorbar(surf,shrink=0.6, aspect=10)
    ax.set_title(r'True')
    ax.set_xticks([0,0.5,1])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_yticks([0,0.5,1.0])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_zlim([0,0.4])
    ax.set_zticks([0,0.2,0.4])
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.tick_params(direction='out', pad=0)
    ax.set_xlabel('$x$', labelpad=-4)
    ax.set_ylabel('$t$', labelpad=-2)
    ax.set_zlabel('$u$', labelpad=-4)
    
    ######### UROM #########
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    surf = ax.plot_surface(Y, X, uUROM[:,:,p], cmap='coolwarm',
                               linewidth=1, shade=False, antialiased=False,rstride=r,
                                cstride=c,rasterized=True)
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
        
    #fig.colorbar(surf,shrink=0.6, aspect=10)
    ax.set_title(r'UROM')
    ax.set_xticks([0,0.5,1])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_yticks([0,0.5,1.0])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_zlim([0,0.4])
    ax.set_zticks([0,0.2,0.4])
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.tick_params(direction='out', pad=0)
    ax.set_xlabel('$x$', labelpad=-4)
    ax.set_ylabel('$t$', labelpad=-2)
    ax.set_zlabel('$u$', labelpad=-4)
    
    ######### GP16 #########
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    surf = ax.plot_surface(Y, X, uGP16[:,:,p], cmap='coolwarm',
                               linewidth=1, shade=False,antialiased=False,rstride=r,
                                cstride=c,rasterized=True)
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
    
    
    ax.set_title(r'GROM('+str(Q)+')')
    ax.set_xticks([0,0.5,1])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_yticks([0,0.5,1.0])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_zlim([0,0.4])
    ax.set_zticks([0,0.2,0.4])
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.tick_params(direction='out', pad=0)
    ax.set_xlabel('$x$', labelpad=-4)
    ax.set_ylabel('$t$', labelpad=-2)
    ax.set_zlabel('$u$', labelpad=-4)
    
    ######### GP4 #########
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    surf = ax.plot_surface(Y, X, uGP4[:,:,p], cmap='coolwarm',
                               linewidth=1, shade=False,antialiased=False,rstride=r,
                                cstride=c,rasterized=True)
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
    
    ax.set_title(r'GROM('+str(R)+')')
    ax.set_xticks([0,0.5,1])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_yticks([0,0.5,1.0])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_zlim(top = 0.4)
    ax.set_zticks([0,0.2,0.4])
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.tick_params(direction='out', pad=0)
    ax.set_xlabel('$x$', labelpad=-4)
    ax.set_ylabel('$t$', labelpad=-2)
    ax.set_zlabel('$u$', labelpad=-4)
    
    ######### NIROM #########
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    surf = ax.plot_surface(Y, X, uNIROM[:,:,p], cmap='coolwarm',
                               linewidth=1, shade=False,antialiased=False,rstride=r,
                                cstride=c,rasterized=True)
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
    
    ax.set_title(r'NIROM')
    ax.set_xticks([0,0.5,1])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_yticks([0,0.5,1.0])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.set_zlim(top = 0.4)
    ax.set_zticks([0,0.2,0.4])
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax.tick_params(direction='out', pad=0)
    ax.set_xlabel('$x$', labelpad=-4)
    ax.set_ylabel('$t$', labelpad=-2)
    ax.set_zlabel('$u$', labelpad=-4)
    
    fig.subplots_adjust(right=0.8,hspace=0.3, wspace=0.15)
    cbar_ax = fig.add_axes([0.83, 0.27, 0.02, 0.5])
    fig.colorbar(surf,cax=cbar_ax)
    
    fig.savefig('plots/burgers_u_Re' + str(ReTest[p]) + \
                '_R=' + str(R) + '_Q=' + str(Q) + '.png',\
                dpi = 500, bbox_inches = 'tight')

