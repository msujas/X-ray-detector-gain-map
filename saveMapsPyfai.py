# The script is to save 2-theta map, Azm map, pixel distance map, and polarization scale map using GSAS-II scriptable.
# Use python installed with GSAS-II to execute the script.  Best way: use run_savemaps.bat (edit paths inside)
# An image and an *.imctrl file have to be present in the directory containing this script and the bat file.
# C:\Users\17bmuser\AppData\Local\Continuum\gsas2full\python.exe

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
    os.chdir(r'C:\Users\kenneth1a\Documents\beamlineData\March2023_gainMap/') #input working directory
    
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
        
    geometry = pyFAI.geometry.Geometry(detector = 'pilatus2mcdte') #input detector geometry, or put in shape if unavailable
    #geometry = pyFAI.geometry.Geometry(detector = shape)
    if not ponifiles:
        raise runTimeError("need at least one poni in the folder!")
    else:
        for file in ponifiles:
            geometry.load(file)
            twothetaMap = geometry.twoThetaArray()*180/np.pi
            polmap = geometry.polarization(factor = polarisation) 
            ttm_image = fabio.edfimage.EdfImage(twothetaMap)
            name = file.replace('.poni','')
            tthfname = f'{path}/{name}_2thmap.edf'
            ttm_image.save(tthfname)
            polimage = fabio.edfimage.EdfImage(polmap)
            polfname = f'{path}/{name}_polmap.edf'
            polimage.save(polfname)
            print(tthfname)
            print(polfname)