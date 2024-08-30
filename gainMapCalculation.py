import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
from glob import glob
from PIL import Image
from datetime import datetime
from skimage.exposure import equalize_hist
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import os,fabio, re, math
from pprint import pprint

# Import files from each position
direc = r'Z:\visitor\ch6987\bm31\20240326\pylatus\gainmap'
#direc = r'C:\Users\kenneth1a\Documents\beamlineData\Feb2023_gainMap'
os.chdir(direc)
cbfdir = 'glassRod'
mapdir = r'maps/'
tthetamapfiles = glob(f'./{mapdir}/*_2thmap.edf')
polmapfiles = glob(f'./{mapdir}/*_polmap.edf')
SAmapfiles = glob(f'./{mapdir}/*_solidAngle.edf')
tthetamapfiles.sort()
polmapfiles.sort()
SAmapfiles.sort()
solidAngleCorrection = True
avdir = 'average'
start, stop =46,48 # straight section at end of data for extrapolating values
filenamescbf = {}
subdirs = glob(f'{cbfdir}/pos*/')
subdirs.sort()
for c,subdir in enumerate(subdirs,1):
    if not os.path.isdir(subdir):
        continue
    
    filenamescbf[c] = [f'{subdir}/{avdir}/{file}' for file in os.listdir(f'{subdir}/{avdir}/')
                       if 'average.cbf' in file and not 'gainCorrected' in file]
  
tmaps = {}
for c,file in enumerate(tthetamapfiles,1):
    tmaps[c] = fabio.open(file).data


#polmapfiles.sort(key = lambda x: int(x.split('_')[0].replace(f'./{mapdir}\\pos','')))

polmaps = {}
for c,file in enumerate(polmapfiles,1):
    polmaps[c] = fabio.open(file).data
    #polmaps.append(fabio.open(file).data)

if solidAngleCorrection:
    
    SAmaps = {}
    for c,file in enumerate(SAmapfiles,1):
        SAmaps[c] = fabio.open(file).data

def read_files(filenames):
    data = []
    for i in range(0, len(filenames)):
        #data.append(io.imread(filenames[i]))
        data.append(fabio.open(filenames[i]).data)
    data = np.asarray(data)
    data[data <= -1] = -1
    av = np.sum(data, axis = 0)/data.shape[0]
    return av
def nan_mask(data):
    mask = np.copy(data)
    mask[mask>=0] = 1
    mask[mask<0] = np.nan
    return mask

data = {}
dataMasked = {}
for key in filenamescbf:
    data[key] = read_files(filenamescbf[key])
    masked = np.multiply(data[key], nan_mask(data[key])) #set masked pixels = nan

    masked = np.divide(masked, polmaps[key]) #apply polarisation correction
    if solidAngleCorrection:
        masked = np.divide(masked,SAmaps[key])
    dataMasked[key] = masked

pprint(filenamescbf)
pprint(polmapfiles)
pprint(tthetamapfiles)

def bin_map(tmap, nbins):
    """
    Bin 2-theta map
    Input (2-theta map, total number of bins)
    Return bin, map of bin index
    """
    bins = np.linspace(np.min(tmap), np.max(tmap), num = nbins)
    binned = np.digitize(tmap, bins) -1 # Digitize index starts at 1
    return bins, binned

def unwrap_image(data, datamap, bins = 800):
    """
    Unwraps a 2D diffraction image
    Takes input of diffraction pattern, 2-theta map, and number of bins
        diffraction pattern = data, 2-theta map = datamap, number of bins = bins
    Defaults to 800 bins if no input is provided
    Return bins and unwrapped radial array where rows correspond to each 2-theta bin
    """
    bins, binmap = bin_map(datamap, bins) # Bin input 2-theta map (bins, map of bins)
    values = np.array((binmap.ravel(), data.ravel())) # Values from diffraction pattern (bin, values)
    rind = np.unique(values[0]) # Unique indicies 
    
    avals = [] # Empty array 
    lengths = np.zeros(len(rind)) # Total count of values at each 2-theta
    for i in range (0, len(rind)):
        pos = np.where(values[0] == rind[i])
        vals = np.ndarray.flatten(values[:, pos][1])
        lengths[i] = len(vals)
        avals.append(vals)
   
    radial_array = np.zeros((len(avals), int(np.max(lengths)))) # Output array for unwrapped values
    for i in range (0, len(avals)):
        nan_array = np.empty( int( np.max(lengths) - len(avals[i]) ) ) # Initialize empty array for values not in measurement
        nan_array[:] = np.nan # Set all values in array to NaN
        radial_array[i] = np.concatenate( (avals[i], nan_array) )
    # Return 2-theta bins and unwrapped array
    return bins, radial_array

def bootstrap_resample(rad_array):
    """
    --- UNUSED FUNCTION ---
    Bootstrap resample unwrapped radial array along horizontal
    Returned array has same number of samples along horizontal
    Subsamples are 10 for no particular reason
    """
    out_arr = np.empty(rad_array.shape) # Initialize output array
    for i in range (rad_array.shape[0]): # Loop over horizontal rows
        data = np.copy(rad_array[i]) # Copy current row
        data = data[~np.isnan(data)] # Remove nans
        resamples = np.random.choice(data, size = (10, rad_array.shape[1]), replace = True) # 10 x [input horizontal] random samples with replacement 
        medians = np.nanmedian(resamples, axis = 0) # Take median across axis 0 to return [input horizontal] medians
        out_arr[i] = medians # Put medians in output array
    return out_arr


def interpolate_bstrap(bmap, bins, min_val, max_val):
    """
    --- UNUSED FUNCTION ---
    Interpolate the bootstrapped data map and return interpolating function
    Pick range of 2-theta values at high 2-theta to use for extrapolation at high 2-theta
    Pass bootstrapped map, bins from unwrapping function ( unwrap_image() ), min 2theta, and max 2theta
    Extrapolate to twice the input 2-theta value
    """
    median_vals = np.nanmedian(bmap, axis = 1) # Take median of bootstrapped array
    f = interp1d(bins, median_vals, kind = 'slinear', fill_value = 'extrapolate') # Interpolate data using order 1 splines
    x_extrap = np.linspace(min_val, max_val, 100) # Get x-values for extrapolation at high 2-theta
    y_extrap = f(x_extrap) # Get y-values for extrapolation from previous interpolation
    #f_extrap = interp1d(x_extrap, y_extrap, kind = 'slinear', fill_value = 'extrapolate') # Function for extrapolating high 2-theta using order 1 splines
    # Add median filter to extrapolation section
    f_extrap = interp1d(x_extrap, medfilt(y_extrap, 5), kind = 'slinear', fill_value = 'extrapolate') # Function for extrapolating high 2-theta using order 1 splines
    
    # Calculate new values including extrapolated region
    # Use 5000 points for interpolation for no particular reason
    x_new = np.linspace(bins[0], bins[-1]*2, 5000) # X values
    y_new = [] # Array for y values
    for i in range (0, len(x_new)):
        if x_new[i] <= min_val:
            y_new.append(f(x_new[i])) # Below cutoff use intepolation of data
        if x_new[i] > min_val:
            y_new.append(f_extrap(x_new[i])) # Above cutoff use extrapolation from cutoff range specified
    y_new = np.array(y_new)
    # Calculate interpolating function for new y-values that include extrapolation at high 2-theta
    # Use order 1 splines for interpolation
    f_out = interp1d(x_new,y_new, kind = 'slinear', fill_value = 'extrapolate')
    return f_out

def interpolate_uwrap(umap, bins, min_val, max_val):
    """
    Interpolate the unwrapped data map and return interpolating function
    Pick range of 2-theta values at high 2-theta to use for extrapolation at high 2-theta
    Pass bootstrapped map, bins from unwrapping function ( unwrap_image() ), min 2theta, and max 2theta
    Extrapolate to twice the input 2-theta value
    """
    median_vals = np.nanmedian(umap, axis = 1) # Take median of bootstrapped array
    f = interp1d(bins, median_vals, kind = 'slinear', fill_value = 'extrapolate') # Interpolate data using order 1 splines
    x_extrap = np.linspace(min_val, max_val, 100) # Get x-values for extrapolation at high 2-theta
    y_extrap = f(x_extrap) # Get y-values for extrapolation from previous interpolation
    f_extrap = interp1d(x_extrap, y_extrap, kind = 'slinear', fill_value = 'extrapolate') # Function for extrapolating high 2-theta using order 1 splines
    # Add median filter to extrapolation section
    f_extrap = interp1d(x_extrap, medfilt(y_extrap, 5), kind = 'slinear', fill_value = 'extrapolate') # Function for extrapolating high 2-theta using order 1 splines
    
    # Calculate new values including extrapolated region
    # Use 5000 points for interpolation for no particular reason
    x_new = np.linspace(bins[0], bins[-1]*2, 5000) # X values
    y_new = [] # Array for y values
    for i in range (0, len(x_new)):
        if x_new[i] <= min_val:
            y_new.append(f(x_new[i])) # Below cutoff use intepolation of data
        if x_new[i] > min_val:
            y_new.append(f_extrap(x_new[i])) # Above cutoff use extrapolation from cutoff range specified
    y_new = np.array(y_new)
    # Calculate interpolating function for new y-values that include extrapolation at high 2-theta
    # Use order 1 splines for interpolation
    f_out = interp1d(x_new,y_new, kind = 'slinear', fill_value = 'extrapolate')
    return f_out

def gain_map_all_pos(positions, tmaps, e_min, e_max, nbins = 800):
    """
    Calculate a gain map from five positions, return calculated map at each position
    Input set of average images at three positions (pos1, pos2, pos3) and corresponding 2-theta maps (tmap1, tmap2, tmap3)
    Provide min/max for 2-theta region to use for extrapolation for high 2-theta (e_min, e_max). This provides an estimated gain at high 2-theta
    Specify number of bins to use for radial average, defaults to 800
    Gain map at edges where information is estimated by extrapolation is probably not super reliable
    Gain map inside beamstop is also not correct
    """
    # Unwrap images to generate maps of 2-theta vs number of counts
    bins = {}
    maps = {}
    ifuns = {}
    for n in positions:
        bins[n], maps[n] = unwrap_image(positions[n], tmaps[n], nbins)

        # Get interpolated 1D function corresponding to 1D average of input diffraction patterns
        ifuns[n] = interpolate_uwrap(maps[n], bins[n], e_min, e_max)

    gmaps = np.empty(shape = (*positions[1].shape,len(positions)))
    for n in positions:
        signal = 0
        for f in ifuns:
            signal += ifuns[f](tmaps[n]) # Generate estimated signals at position and then average
        signal = signal/len(tmaps)
        gmap = positions[n]/signal
        gmap = gmap/np.nanmedian(gmap) #normalise by dividing by median
        #gmap = np.where(gmap < 0.7, -1, gmap)
        #gmap = np.where(gmap > 1.4, -1, gmap)
        gmap = np.where(np.isnan(positions[n]), -1, gmap)
        gmap = np.where(gmap > 10**10, -1, gmap)
        gmap = np.where(np.isnan(gmap),-1,gmap)

        #gmap = np.where(positions[n] < 0, -1, gmap)
        gmaps[:,:,n-1] = gmap
        
    return gmaps # Return gainmap

# Get gain mask of undamaged and damaged pixels (umap, dmap)
# Define damaged pixels as those with responses more than 4 sigma away from the mean
# Define sigma using 1.4826*MAD
def damaged_pixels(data):
    median = np.nanmedian(data) # Calculate median
    mad = np.nanmedian(np.abs(data-median)) # Calculate MAD
    sigma = 1.4826*mad # Calculate sigma
    dmap = np.copy(data) # Damaged pixels
    dmap[dmap<(median+4*sigma)] = 0
    umap = np.copy(data) # Undamaged pixels
    umap[umap>(median+4*sigma)] = np.nan
    return umap, dmap
# Corrects gain map containing damaged pixels
# needs median_filter from scipy.ndimage
def map_correction(gainmap, window_size):
    udmap, damap = damaged_pixels(gainmap) # Get map of undamaged and damaged pixels
    mask = np.copy(udmap) # Mask for damaged pixels
    mask[~np.isnan(mask)] = 1 # Set non nan in mask (undamaged pixels) to one
    mask = np.nan_to_num(mask) # Set nan in mask (damaged pixels) to zero
    filtered = median_filter(gainmap, window_size) # Apply 3x3 median filter to undamaged pixel map
    filtered = filtered*mask # Set damaged pixels to zeros 
    return filtered+damap # Add back in gains for damaged pixels

fig, ax = plt.subplots(len(data),4, figsize = (2.2*4,2.4*len(data)))
for i in data:
    ax[i-1][0].imshow(data[i])
    ax[i-1][1].imshow(tmaps[i])
    ax[i-1][2].imshow(polmaps[i])
    if solidAngleCorrection:
        ax[i-1][3].imshow(SAmaps[i])
plt.show()


bins = 1000
tx = {}
ty = {}
y = {}
data_select = {1:dataMasked[1],4:dataMasked[4],5:dataMasked[5],6:dataMasked[6],9:dataMasked[9]}
print('binning data')

for n in dataMasked:
    tx[n],ty[n] = unwrap_image(dataMasked[n], tmaps[n], bins = bins)
    y[n] = np.nanmedian(ty[n], axis = 1)

plt.figure()
#plt.rcParams["figure.figsize"] = (15, 5)
for n in tx:
    plt.plot(tx[n], y[n],label = n)
plt.legend()
plt.xlabel('2$\\theta$')
plt.ylabel('Counts')
plt.show()

bins = 1000
print('calculating maps')
maps = gain_map_all_pos(dataMasked, tmaps, start, stop, bins)

columns = 3
rows = int(2*math.ceil(len(data)/columns))
fig,ax = plt.subplots(rows,columns, dpi = 150, figsize = (2.2*columns, 2.4*rows))
#plt.rcParams["figure.figsize"] = (15, 20)
k = np.percentile(np.where(np.isnan(y[1]),-2,y[1]), 99) # Scale images to 99th percentile of 1D integration
usedAxes = np.empty(shape = (rows,columns))
for n in data:
    row = 2*((n-1) // columns)
    col = (n-1) % columns
    ax[row,col].imshow(np.nan_to_num(data[n]), vmax = k)
    ax[row,col].set_title(f'Position {n}')
    ax[row+1,col].imshow(maps[:,:,n-1], vmin = 0.85, vmax = 1.15)
    usedAxes[row,col] = True
    usedAxes[row+1,col] = True
usedAxes = np.where(usedAxes == True, True, False)
for r in range(len(usedAxes)):
    for c in range(len(usedAxes[0])):
        if not usedAxes[r,c]:
            fig.delaxes(ax[r,c])
fig.tight_layout()
plt.show()
plt.figure(dpi = 150)
median_map = np.nan_to_num(np.nanmedian(np.where(maps<0, np.nan, maps), axis = 2))

#plt.imshow(median_map, vmin = 0.85, vmax = 1.15)
# Apply correction to map
window_size = 3
corrmap = map_correction(median_map, window_size)
plt.imshow(corrmap, vmin = 0.85, vmax = 1.15)
plt.colorbar()
plt.title('median gain map')
plt.tight_layout()
plt.show()
# Export calculated gain map
gainmapfilename = 'gainMap'
# Define gain map format
g_form = 1 # Set to 1 if the gain map is to be divided by measured images, 0 if it is to be multiplied (GSAS wants multiply)
dtstring = datetime.today().strftime('%Y-%m-%d')
# Change to whatever format and then remove any nans which may be present
if g_form == 1:
    nmap = np.nan_to_num(corrmap)
if g_form == 0:
    nmap = np.nan_to_num(1/corrmap)
else:
    nmap = np.nan_to_num(corrmap) # Format parameter to something other than 0 returns a multiplied map
# Remove any inf/very large values created by any divide by zeros
#nmap[nmap>10**10] = 0
#nmap[nmap<-10**10] = 0

# Export map as 32 bit tif
im = Image.fromarray(np.float32(nmap), mode='F') # float32
#im.save(gainmapfilename+'_'+dtstring+'.tif', 'TIFF')

## Used for detector diagnostics ##
# Export calculated gain map without median filter applied
gainmapfilename = 'gainMap'
# Define gain map format
g_form = 1 # Set to 1 if the gain map is to be divided by measured images, 0 if it is to be multiplied (GSAS wants multiply)

# Change to whatever format and then remove any nans which may be present
if g_form == 1:
    nmap = np.nan_to_num(median_map)
if g_form == 0:
    nmap = np.nan_to_num(1/median_map)
else:
    nmap = np.nan_to_num(median_map) # Format parameter to something other than 0 returns a multiplied map
# Remove any inf/very large values created by any divide by zeros
nmap[nmap>10**10] = -1
nmap[nmap<-10**10] = -1

# Export map as 32 bit tif
im = Image.fromarray(np.float32(nmap), mode='F') # float32
gainmapfilenameUnfiltered = gainmapfilename+'_unfiltered_'+dtstring+'.tif'
edfUnfilteredName = gainmapfilenameUnfiltered.replace('.tif','.edf')
#im.save(gainmapfilenameUnfiltered, 'TIFF')

edfimage = fabio.edfimage.EdfImage(data = nmap)
edfimage.save(edfUnfilteredName)

# Change gain map into format expected by GSAS-II and save
# GSAS-II expects a gain map of integers where 1 corresponds to a value of 1000, 0.95 -> 950, etc
#
# GSAS-II has a display error where if the values are too large and are multiplied by a gain map they will 
# overflow and show negative values in the program
gsas_format = np.copy(nmap)
gsas_format = np.floor(gsas_format*1000)
gsas_format[gsas_format>2000] = 0 # Any pixel that needs more than a 200% correction is set to 0, comment out if undesired
# Create image for export from gain map
im = Image.fromarray(gsas_format.astype('I'), mode = 'I')
# Export calculated gain map

gainmapfilename = 'gainMap_gsasformat'
gainmapfilenameFull = gainmapfilename+'_'+dtstring+'.tif'
#im.save(gainmapfilenameFull, 'TIFF')
gainmapfilename = 'gainMap'
gainmapfilenameFull = f'{gainmapfilename}_filtered_{dtstring}.edf'
gainmapfilenameFull2 = f'{gainmapfilename}_filtered_kpm_{dtstring}.edf'

filteredIm = np.where(nmap <= 0, 1, nmap)
filteredIm = np.where(filteredIm >=2, -1, filteredIm)
filteredIm2 = np.where(nmap > 1.5, -1, nmap)
filteredIm2 = np.where(filteredIm2 == 0, -1, filteredIm2)
filteredIm2 = np.where(filteredIm2 < 0.7, -1, filteredIm2)
edfimage = fabio.edfimage.EdfImage(data = filteredIm)
edfimage2 = fabio.edfimage.EdfImage(data = filteredIm2)
edfimage.save(gainmapfilenameFull)
edfimage2.save(gainmapfilenameFull2)

plt.figure(dpi = 150)
plt.imshow(filteredIm2,vmin = 0.9,vmax = 1.1)
plt.colorbar()
plt.title('final map')
plt.show()