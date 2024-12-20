#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:32:46 2023

@author: ckettmayer
"""

###  3D trajectory instantaneous velocity ###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Load trajectory data
filename = 'data_example.txt'
data = np.loadtxt(filename, delimiter=None, skiprows=1, usecols=(0,1,2)) # Use columns with (x,y,z) data

# Parameters
pixel_size = 0.056       #(um)
point_time = 0.032768    #(s)

# Process coordinates
x = (data[:,0]- data[0,0])*pixel_size
y = (data[:,1]- data[0,1])*pixel_size
z = (data[:,2]- data[0,2])*pixel_size
t = np.arange(0, len(x)*point_time, point_time)   


# Calculate velocity
v = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2) / point_time



# 3D Plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(8,6))
# ax.view_init(90,270)   #xy plane
# ax.view_init(0,270)    #xz plane
# ax.view_init(0,0)      #yz plane

# Colormap
cm = plt.colormaps['cividis']         
for i in range(len(v) - 1):
    color = cm((v[i] - min(v)) / (max(v) - min(v)))
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=color)
    
# Colorbar
norm = mpl.colors.Normalize(vmin=min(v), vmax=max(v))
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.2, shrink=0.7, label=r'Velocity ($\mu$m/s)')

# Labels and plot adjustments
plt.title(f'File : {filename}')
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
ax.set_zlabel(r'z ($\mu$m)') 
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()

plt.show()
