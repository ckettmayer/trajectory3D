#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:54:47 2023

@author: ckettmayer
"""

###  3D trajectory visualization  ###

import numpy as np
import matplotlib.pyplot as  plt
import matplotlib as mpl


# Load trajectory data
filename = 'data_example.txt'
data = np.loadtxt(filename, delimiter=None, skiprows=1, usecols = (0,1)) # Use columns with (x,y) data

# Parameters 
pixel_size = 0.056       #(um)
point_time = 0.032768    #(s)

# Centering
x = (data[:,0] - data[0,0])*pixel_size
y = (data[:,1] - data[0,1])*pixel_size
t = np.arange(0, len(x)*point_time, point_time)   



# Plot
fig, ax = plt.subplots(figsize=(7,6))

# Colormap
cm = plt.colormaps['viridis']
for i in range(len(t)-1):
    ax.plot(x[i:i+2], y[i:i+2], color=cm((i+1)/len(t)))

# Colorbar
norm= mpl.colors.Normalize(vmin=min(t), vmax= max(t))
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.07, shrink=0.7, label='t (s)')

# Labels and plot adjustments
plt.title(f'File : {filename}')
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')

plt.tight_layout()


plt.show()

