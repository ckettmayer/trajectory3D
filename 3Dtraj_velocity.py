#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:32:46 2023

@author: ckettmayer
"""


###  3D trajectory instantaneous velocity ###

import numpy as np
import matplotlib.pyplot as  plt
import matplotlib as mpl
# import cmasher as cmr          #more perceptually uniform colormaps


filename = 'outside_3.txt'
raw_data = open(filename)
data = np.loadtxt(raw_data, delimiter=None, skiprows=1, usecols = (0,1,2)) #Con usecols elijo que columnas quiero guardar en data

pixel_size = 0.056   #tamaño de pixel en um
point_time = 0.032768    #tiempo de adquisición por punto de la trayectoria en s

x = (data[:,0]- data[0,0])*pixel_size
y = (data[:,1]- data[0,1])*pixel_size
z = (data[:,2]- data[0,2])*pixel_size


N = len(x)
t = np.arange(0, N*point_time, point_time)   #tiempo en segundos


v = np.zeros(N-1)

for i in range(N-1):
    v[i] = np.sqrt( (x[i+1]-x[i])**2  + (y[i+1]-y[i])**2 + (z[i+1]-z[i])**2)/point_time




fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(6,4))
# ax.view_init(90,270)   #xy plane

cm = plt.cm.get_cmap('cividis')
# cm = cmr.bubblegum          

for i in range(len(v) - 1):
    color = cm((v[i] - min(v)) / (max(v) - min(v)))
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=color)
    

norm = mpl.colors.Normalize(vmin=min(v), vmax=max(v))
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, pad=0.3, label='Velocity (um/s)')
plt.title(filename)

plt.xlabel('x (um)')
plt.ylabel('y (um)')
ax.set_zlabel('z (um)') 
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

fig.tight_layout()
