# The script is to save 2-theta map, Azm map, pixel distance map, and polarization scale map using pyFAI
#requires poni files made with pyfai-calib(2)

import os
from glob import glob
import fabio
import pyFAI.geometry
import numpy as np
def updatelist(ext):			# search for tif files in the main directory
    filelist=glob(ext)
    filelist.sort(key=lambda x: os.path.getctime(x), reverse=False) #sort files by creation time in ascending order
    return filelist


if __name__ == "__main__":
    os.chdir(r'C:\Users\kenneth1a\Documents\beamlineData\May2023/') #input working directory
    
    cwd = os.getcwd()				# get the current path

    PathWrap = lambda fil: os.path.join(cwd,fil)
    
    polarisation = 0.99
    
    newdir = 'maps'					# make a subfolder to store integrated images
    path = os.path.join(cwd,newdir)
    if not os.path.exists(path):			
        os.mkdir(path)
    pathmaps = path
    cbfs = updatelist('*.cbf')
    ponifiles = updatelist('*.poni')
    if cbfs:
        cbf = fabio.open(cbfs[0]).data
        shape = cbf.shape
        
    geometry = pyFAI.geometry.Geometry() #detector should be in poni, specify if needed
    #geometry = pyFAI.geometry.Geometry(detector = shape)
    if not ponifiles:
        raise RuntimeError("need at least one poni in the folder!")
    else:
        for file in ponifiles:
            geometry.load(file)
            twothetaMap = geometry.twoThetaArray()*180/np.pi
            polmap = geometry.polarization(factor = polarisation)
            solidAngleMap = geometry.solidAngleArray()
            ttm_image = fabio.edfimage.EdfImage(twothetaMap)
            name = file.replace('.poni','')
            tthfname = f'{path}/{name}_2thmap.edf'
            ttm_image.save(tthfname)
            polimage = fabio.edfimage.EdfImage(polmap)
            polfname = f'{path}/{name}_polmap.edf'
            polimage.save(polfname)
            chimap = geometry.chiArray()*180/np.pi
            chiimage = fabio.edfimage.EdfImage(chimap)
            chiname = f'{path}/{name}_azim.edf'
            chiimage.save(chiname)
            solidangleimage = fabio.edfimage.EdfImage(solidAngleMap)
            solidanglename = f'{path}/{name}_solidAngle.edf'
            solidangleimage.save(solidanglename)
            print(tthfname)
            print(polfname)
            print(chiname)
            print(solidanglename)