#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:54:10 2023

@author: ckettmayer
"""


###  3D trajectory mean squared displacement analysis ###

import numpy as np
import matplotlib.pyplot as  plt
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText

# Load data
filename = 'outside_3.txt'
data = np.loadtxt(filename, delimiter=None, skiprows=1, usecols = (0,1,2)) 

# Parameters
pixel_size = 0.056       #(um)
point_time = 0.032768    #(s)

# Process coordinates
x = (data[:,0] - data[0,0])*pixel_size
y = (data[:,1] - data[0,1])*pixel_size
z = (data[:,2] - data[0,2])*pixel_size
t = np.arange(0, len(x)*point_time, point_time)   

N = len(x)
t = np.arange(0, N*point_time, point_time)  
lag = t


# Percentage of MSD points to be calculated and fitted (standard is 25% or less)
percentage = 10
Nplot = int(N*(percentage/100))   # Number of MSD points to be plotted and adjusted



# Calculate MSD
msd=np.zeros(N)
sigma=np.zeros(N)

for tau in range(Nplot):
    d=np.zeros(N-tau)
    for i in range(N-tau):
        d[i]=(x[i+tau]-x[i])**2 + (y[i+tau]-y[i])**2 + (z[i+tau]-z[i])**2  # Computes every squared displacement for a lag time tau
    msd[tau]=np.mean(d) # Mean squared displacement MSD(tau)
    sigma[tau]=np.std(d)/((N-tau)**(1/2)) # MSD error 




# Anomalous diffusion model fit
def anomalous_diff(lag,D_gen,alpha,s):
    return (6 * D_gen*lag**alpha+s)

p0_an=[0.1, 0.1, 0.1] # Initial parameters
popt_an, pcov_an = curve_fit(anomalous_diff, lag[1:Nplot], msd[1:Nplot], p0=p0_an, maxfev=2000)

alpha, alpha_err = popt_an[1], np.sqrt(pcov_an[1,1])




# Additional model fit

def brownian_diff(lag,D,s):
    return (6 * D * lag + s)

def confined_diff(lag,Dm,L,s):
    return ((L**2/3)*(1-np.exp(-(18*Dm*lag)/L**2))) 

def directed_diff(lag,D,V,s):
    return (6 * D * lag + (V * lag)**2 + s)



# Selects model according to alpha value

if 1 - alpha_err < alpha < 1 + alpha_err:
    model = 'Brownian diffusion'
elif alpha < 1:
    model = 'Confined diffusion'
else:
    model = 'Directed motion and diffusion'


if model=='Brownian diffusion':
    p0_br=[0.1, 0] # Initial parameters
    popt_br, pcov_br = curve_fit(brownian_diff, lag[1:Nplot], msd[1:Nplot], p0=p0_br, maxfev=2000)

    D, D_err = popt_br[0], np.sqrt(pcov_br[0,0])
    s, s_err = popt_br[1], np.sqrt(pcov_br[1,1])


if model=='Confined diffusion':
    p0_cf=[0.1, 0.1, 0] # Initial parameters
    popt_cf, pcov_cf = curve_fit(confined_diff, lag[1:Nplot], msd[1:Nplot], p0=p0_cf, maxfev=2000)

    Dm, Dm_err = popt_cf[0], np.sqrt(pcov_cf[0,0])
    L, L_err = popt_cf[1], np.sqrt(pcov_cf[1,1])
    s, s_err = popt_cf[2], np.sqrt(pcov_cf[2,2])


if model=='Directed motion and diffusion':
    p0_dm=[0.1, 0.1, 0] # Initial parameters
    popt_dm, pcov_dm = curve_fit(directed_diff, lag[1:Nplot], msd[1:Nplot], p0=p0_dm, maxfev=2000)

    D, D_err = popt_dm[0], np.sqrt(pcov_dm[0,0])
    V, V_err = popt_dm[1], np.sqrt(pcov_dm[1,1])
    s, s_err = popt_dm[2], np.sqrt(pcov_dm[2,2])





    
lag_model  = np.linspace(lag[1], lag[Nplot], 1000)   

fig = plt.figure(figsize = (12,4))

ax1  = fig.add_subplot(131)
plt.errorbar(lag[1:Nplot], msd[1:Nplot], yerr=sigma[1:Nplot],   
               marker='.', markersize=1, alpha=0.4, label='MSD', color='k',linestyle='')
         
plt.title('MSD')            
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'MSD ($\mu m^2$)')
plt.legend()


ax2  = fig.add_subplot(132)
plt.errorbar(lag[1:Nplot], msd[1:Nplot], yerr=sigma[1:Nplot],   
               marker='.', markersize=1, alpha=0.4, label='MSD', color='k', linestyle='', zorder=0)
plt.plot(lag_model, anomalous_diff(lag_model, *popt_an),
            'c-', label=r'MSD($\tau$)= MSD$_0$+6D$\tau^\alpha$')
            
plt.title('Anomalous diffusion model')            
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'MSD ($\mu m^2$)')
plt.legend()


ax3  = fig.add_subplot(133)
plt.errorbar(lag[1:Nplot], msd[1:Nplot], yerr=sigma[1:Nplot],   
               marker='.', markersize=1, alpha=0.4, label='MSD', color='k', linestyle='', zorder=0)

if model=='Brownian diffusion':
    plt.plot(lag_model, brownian_diff(lag_model, *popt_br),
             'r-', label=r'MSD($\tau$)= MSD$_0$+6D$\tau$')
    plt.title('Brownian diffusion model')  
    
if model=='Confined diffusion':
    plt.plot(lag_model, confined_diff(lag_model, *popt_cf),
             'g-', label=r'MSD($\tau$)= MSD$_0$+$\frac{L^2}{3}(1-exp[\frac{-18 D_m \tau}{L^2}])$')
    plt.title('Confined diffusion model')  

if model=='Directed motion and diffusion':
    plt.plot(lag_model, directed_diff(lag_model, *popt_dm),
             'b-', label=r'MSD($\tau$)= MSD$_0$+6D$\tau$+(V$\tau$)$^2$')
    plt.title('Directed motion and diffusion model')  
    
            
          
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'MSD ($\mu m^2$)')
plt.legend()

fig.tight_layout()

print(f'''MSD analysis of {filename}
{percentage}% of total points

Anomalous diffusion fit: 
 alpha = {alpha:.5f} ± {alpha_err:.5f}
 ''')

    
if model=='Brownian diffusion':
    print(f'Brownian diffusion fit: \n D = ({D:.5f} ± {D_err:.5f}) um2/s')  
    
if model=='Confined diffusion':
    print(f'Confined diffusion fit: \n L = ({L:.5f} ± {L_err:.5f}) um \n Dm = ({Dm:.5f} ± {Dm_err:.5f}) um^2/s')  

if model=='Directed motion and diffusion':
    print(f'Directed motion and diffusion fit: \n V = ({V:.5f} ± {V_err:.5f}) um/s  \n D = ({D:.5f} ± {D_err:.5f}) um^2/s')  
    

    

    
    


