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
# import cmasher as cmr      #more perceptually uniform colormaps


filename = 'outside_3.txt'   #SimFCS tracking .txt file
raw_data = open(filename)
data = np.loadtxt(raw_data, delimiter=None, skiprows=1, usecols = (0,1,2)) 
 
pixel_size = 0.056       #(um)
point_time = 0.032768    #(s)

x = (data[:,0]- data[0,0])*pixel_size
y = (data[:,1]- data[0,1])*pixel_size
z = (data[:,2]- data[0,2])*pixel_size

N = len(x)
t = np.arange(0, N*point_time, point_time)  




fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(6,4))
# ax.view_init(90,270)   #xy plane

cm = plt.cm.get_cmap('viridis')
# cm = cmr.gem                   # CMasher cmap

for i in range(len(t)-1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cm((i+1)/len(t)))

norm= mpl.colors.Normalize(vmin=min(t), vmax= max(t))
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
plt.colorbar(sm, pad=0.2, label='t(s)')
plt.title(filename)

plt.xlabel('x (um)')
plt.ylabel('y (um)')
ax.set_zlabel('z (um)') 
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.tight_layout()





