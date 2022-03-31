# -*- coding: utf-8 -*-
"""
compute_reference_fractional_built_data.py

Krishna Kumar Perikamana
30.03.2022 
https://www.researchgate.net/profile/Krishna-Kumar-Perikamana

Computing fractional built information at 30m scale from  ESA Sentinel-2 LULC. This will be
used to train a model along with Landsat-8 image.

"""


import numpy as np
from numpy import *
import sys
import csv
from pyrsgis import raster
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction import image


ds, yourRaster = raster.read(r'''.\Data\43P_20190101-20200101_Bangalore_ra_10m_utm43n.tif''')#ESA Sentinel-2 LULC classified image
print("Raster cell size is", yourRaster.shape)

Res1 = 30 #Resolution of Landsat-8
Res2 = 10 #Resolution of Sentinel-2

H = yourRaster.shape[0]
W = yourRaster.shape[1] 

M = int(Res1/Res2)
M2 = M ** 2
Tr = int(H*W/M2) 
R1 = np.random.randn(Tr,M2)

print(M2)
print(Tr)
raise SystemExit

i,j=-1,-1
for x in range(0,H,M):
    for y in range(0,W,M):
        i=i+1
        j=-1
        for m in range(0,M):
            for n in range(0,M):
                j=j+1
                R1[i][j] = yourRaster[x+m][y+n]
                
#Computing fractional values of built from R1
# Built-up   : 7

P = np.random.randn(Tr)
for i in range(0,Tr):
    c1=0 # initializing counter
    for j in range(0,M2):
        if int(R1[i,j])==7:
            c1 = c1+1
    P[i] = c1

P = P/M2
T1 = P *100#scaling to compute percentage
T1.flatten()

savetxt('./Data/frac_landcover_built_2019.dat', T1 ,fmt='%4.2f')



