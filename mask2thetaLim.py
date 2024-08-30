import fabio
import numpy as np
import os
from glob import glob
direc = r'Z:\visitor\a311217\bm31\20240129\pylatus\gainmap'
os.chdir(direc)
min2theta = 1.4
max2theta = 48
mapdir = 'maps/'

maskList = glob('pos*mask.edf')
thetamaps = glob(f'{mapdir}/pos*2thmap.edf')
if not os.path.exists(f'{direc}/thlimMask/'):
    os.makedirs(f'{direc}/thlimMask/')

for mask, thetamap in zip(maskList,thetamaps):
    maskarray = fabio.open(mask).data
    thetamaparray = fabio.open(thetamap).data
    newmask = np.where(thetamaparray > max2theta, 1, maskarray)
    newmask = np.where(thetamaparray < min2theta, 1, newmask)
    fabioimage = fabio.edfimage.EdfImage(newmask)
    fabioimage.save(f'{direc}/thlimMask/{mask}')
