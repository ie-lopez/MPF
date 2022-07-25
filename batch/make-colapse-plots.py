#%%
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
from numpy import exp, loadtxt, pi, sqrt, arange, interp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits 
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.table import Table

from specutils import Spectrum1D
#%%
data='../data/sings/'
sources = Table.read(data+'sources.csv', format='ascii')

#%%
def make_colapse(args):
    targname = args
    cubeSL1 = fits.open(data+targname+'_DR5_SL1_cube') 
    #errorsSL1 = fits.open(data+targname+'_SL1_cube_unc')
    cubeSL2 = fits.open(data+targname+'_DR5_SL2_cube') 
    #errorsSL2 = fits.open(data+targname+'_SL2_cube_unc')
    hdr1 = cubeSL1[0].header                        
    #er_hdr1=errorsSL1[0].header
    hdr2 = cubeSL2[0].header                               
    #er_hdr2=errorsSL2[0].header
    #print(repr(hdr1))

    #Flux Data
    data_cube1 = cubeSL1[0].data[:,:,42:74]
    #error_cube1 = errorsSL1[0].data[:,:,42:74]
    data_cube2 = cubeSL2[0].data[:,:,0:32]
    #error_cube2 = errorsSL2[0].data[:,:,0:32]

    #Wavelength Data
    xwave1 = cubeSL1[1].data
    xwave1 = xwave1.field(0)[0]
    xwave2 = cubeSL2[1].data
    xwave2 = xwave2.field(0)[0]
    x=[]
    for line in xwave2:
        x.append(float(line))
    for line in xwave1:
        x.append(float(line))
    x=np.array(x)

    #Change the units from MJy/sr to Jy/pix
    correct_unit1=abs(hdr1['CDELT1'])*abs(hdr1['CDELT2'])*0.0003046118*(10**6)    
    data_cube1=data_cube1*correct_unit1
    #error_cube1=error_cube1*correct_unit1
    correct_unit2=abs(hdr2['CDELT1'])*abs(hdr2['CDELT2'])*0.0003046118*(10**6)    
    data_cube2=data_cube2*correct_unit2
    #error_cube2=error_cube2*correct_unit2

    #Concatenate SL1 and SL2
    data_cube=np.concatenate((data_cube2,data_cube1),axis=0)
    #error_cube=np.concatenate((error_cube2,error_cube1),axis=0)

    #Reorder x and data_cube
    xcor = x.argsort()
    data_cube = data_cube[xcor]
    #error_cube = error_cube[xcor]
    x=np.sort(x)

    #Cube dimensions and trimming 
    xsize, ysize, zsize = data_cube.shape
    ytrim=0; ysize=ysize-ytrim
    ztrim=0; zsize=zsize-ztrim
    #print('Trimmed cube dimensions:', xsize, ysize, zsize)

    #Collapsed 2D image
    cube_2dflux=np.nansum(data_cube,axis=0)[0:ysize,0:zsize]

    #Collapsed 1D spectrum
    cube_1dflux=np.nansum(data_cube,axis=(1,2))

    f,(ax1,ax2)=plt.subplots(1,2, figsize=(11,3))
    plt.suptitle(targname,fontsize=18, y=1.1)
    ax1.set_title('log Flux (Sum over Wavelength)')
    ax1.imshow(cube_2dflux, origin='lower', cmap='gray', norm=LogNorm())
    ax2.set_title('Flux (Sum over Spaxels)')
    ax2.plot(x,cube_1dflux)
    ax2.set_xlabel(r"Wavelength $(\mu m)$")
    ax2.set_ylabel('Flux (Jy)')
    plt.savefig('./plots/'+targname+'.png', bbox_inches='tight', dpi=120)
    return 

pooldata=[]
for target in sources['target_names']:      
    pooldata.append(target)

#Launch Multiprocessing Pool
start_time = time.time()
if __name__ == '__main__':
    with Pool(mp.cpu_count() - 1) as pool:
        pool.map(make_colapse,pooldata)    

print('Time count')
print("--- %s seconds ---" % (time.time() - start_time))

# %%
