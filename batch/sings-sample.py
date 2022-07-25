#%%
import time
import multiprocessing as mp
from multiprocessing import Pool
import pickle

import numpy as np
from numpy import exp, loadtxt, pi, sqrt, arange, interp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits 
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.visualization import make_lupton_rgb
from astropy.visualization import quantity_support
from astropy.modeling import models

from specutils import Spectrum1D

import lmfit
from lmfit import Model, Minimizer, Parameters
from lmfit.model import save_modelresult

#%%
def Modelcero(x):
    """Simple model of 0 to initialize some models"""
    return 0*x

#powerlaw, zblackbody, gaussian and drude functions wrap existing astropy models.
#lmfit requires wavelength (x) to be passed as function parameter.

def powerlaw(x, c1, c2):
    """power law function (c1*nu^c2) normalized at 1 micron, with positive index convention"""
    x0=1.0  #Normalize at  1 micron
    c2=-c2
    return models.PowerLaw1D.evaluate(x,c1,x0,c2)

def zblackbody(x, OnePlusZ, T, scale):
    """black body function"""
    xs=x*u.um/OnePlusZ
    bb=models.BlackBody(T*u.K, scale)
    return bb(xs)

def gaussian(x, amplitude, xcen, std):
    """1-d gaussian"""
    return models.Gaussian1D.evaluate(x,amplitude,xcen,std)   

def drude(x, amplitude, peakx, frac_FWHM):
    """dust emission"""
    FWHM=peakx*frac_FWHM
    return models.Drude1D.evaluate(x,amplitude,peakx,FWHM)

#pahdust and sidust functions do not exist in astropy

def pahdust(x, OnePlusZ, amplitude_76, amplitude_113):
    """PAH dust emission"""
    
    #Ionized PAH features
    PAH_peakx=     [ 5.27, 5.70, 6.22, 6.69, 7.42, 7.60, 7.85, 8.33, 8.61]
    PAH_frac_FWHM= [0.034,0.035, 0.030,0.07, 0.126,0.044,0.053,0.050, 0.039]
    PAH_rel_amplitude= [0.0,  0.0,   0.8,  0.0,  0.0,  1.0,  0.6,  0.0,   0.5]
    PAH_amplitude=[]
    for ampl in PAH_rel_amplitude: PAH_amplitude.append(amplitude_76*(ampl/PAH_rel_amplitude[5]))
 
    #Neutral PAH features
    PAH_peakx+=    [10.68, 11.23,11.33]
    PAH_peakx+=    [11.99,12.62,12.69,13.48,14.04,14.19,15.90, 16.45,17.04,17.375,17.87,18.92]
    PAH_frac_FWHM+=[0.020, 0.012,0.022]
    PAH_frac_FWHM+=[0.045,0.042, 0.013,0.040,0.016,0.025,0.020,0.014,0.065, 0.012, 0.016,0.019]   
    PAH_rel_amplitude= [0.0,  0.6, 1.0]
    PAH_rel_amplitude+=[0.2,  0.3,   0.1,  0.0,  0.1,  0.1,  0.0,   0.0,  0.0,  0.0,   0.0,  0.0]
    for ampl in PAH_rel_amplitude: PAH_amplitude.append(amplitude_113*(ampl/PAH_rel_amplitude[2]))
    
    pahflux=x-x
    for peakx, frac_FWHM, ampl in zip(PAH_peakx,PAH_frac_FWHM,PAH_amplitude):
        pahflux=pahflux+drude(x,ampl,peakx*OnePlusZ,frac_FWHM)

    return pahflux    #Amplitude unit

def sidust(x, T, amplitude):
    """Silicate dust emission"""
    
    #Custom extinction curve built from weighted sum of three components: 
    #two Drude functions and an exponent 1.7 power-law.
    d1=drude(x,0.80,10.0,0.25)
    d2=drude(x,0.25,11.1,0.25)
    #d3=drude(x,0.40,17.0,0.40)  #Outside of wavelength range.
    ext=d1+d2  #+d3

    # Form linear combination of modified silicate and powerlaw.
    beta=0.1
    ext=(1. - beta)*ext + beta*(9.7/x)**1.7
    si_em=amplitude*1.0E-6*3.97289E13/x**3/(exp(1.4387752E4/x/T)-1.)*ext
    
    return si_em

# Enable tabulated extinction functions. Existing options in dust_extinction are insufficient
# at mid-IR wavelengths

def extinction_a(x, tau, a):
    """Extinction from table data"""
    ext=np.interp(x, wave, a)      
    return exp(-tau*ext)

#Read extinction data from Chiar & Tielens (2005)
#Note, this is normalized to extinction in the K-band (1.004 at 2.14 um)
#agal is for Galactic Center, alocal is for local ISM extinction

wave, agal, alocal = np.loadtxt('../resources/chiar+tielens_2005.dat', skiprows=14, usecols=(0, 1, 2), unpack=True)

#%%
def setpar(pars, name, value, vary, minus):
    """Set any parameter"""
    pars[name].set(value,vary=vary,min=minus)
    
def gaussian_defpar(name, amplitude, mean, stddev, pars):
    """set the parameters for a gaussian"""
    std=stddev
    name=str(name)
    xcen=mean*OneZ
    setpar(pars,name+'_std',std,False,None)
    #setpar(pars,name+'_std',std,True,0)
    setpar(pars,name+'_xcen',xcen,False,None)
    setpar(pars,name+'_amplitude',amplitude,True,0)
    return #pars[name+'_std'].set(std,vary=False),pars[name+'_xcen'].set(xcen,vary=False),pars[name+'_amplitude'].set(amplitude)

def drude_defpar(name, amplitude, peakx, frac_FWHM, pars):
    """set the parameters for a drude"""
    peakx=peakx*OneZ
    setpar(pars,name+'_frac_FWHM',frac_FWHM,False,None)
    #setpar(pars,name+'_frac_FWHM',frac_FWHM,True,0)
    setpar(pars,name+'_peakx',peakx,False,None)
    setpar(pars,name+'_amplitude',amplitude,True,0)    
    return #pars[name+'_frac_FWHM'].set(frac_FWHM,vary=False),pars[name+'_peakx'].set(peakx,vary=False),pars[name+'_amplitude'].set(amplitude)

def pahdust_defpar(name, OnePlusZ, amplitude_76, amplitude_113, pars):
    """set the parameters for pdr dust"""
    setpar(pars,name+'_amplitude_76',amplitude_76,True,0) 
    setpar(pars,name+'_amplitude_113',amplitude_113,True,0) 
    setpar(pars,name+'_OnePlusZ',OnePlusZ,False,None)
    return #pars[name+'_frac_FWHM'].set(frac_FWHM,vary=False),pars[name+'_peakx'].set(peakx,vary=False),pars[name+'_amplitude'].set(amplitude)

def sidust_defpar(name, T, amplitude, pars):
    setpar(pars,name+'_amplitude',amplitude,True,0)
    setpar(pars,name+'_T',T,False,0)
    return

def gaussian_extractpars(prefix, result):
    """extract the parameters for a gaussian"""
    amp = result.params[prefix+'amplitude'].value
    cen = result.params[prefix+'xcen'].value
    std = result.params[prefix+'std'].value
    return amp, cen, std

def drude_extractpars(prefix, result):
    """extract the parameters for a drude"""
    ampl = result.params[prefix+'amplitude'].value
    peak = result.params[prefix+'peakx'].value         
    frac = result.params[prefix+'frac_FWHM'].value
    return ampl, peak, frac

def pahdust_extractpars(prefix, result):
    """extract the parameters for pahdust model"""
    ampl_76 = result.params[prefix+'amplitude_76'].value
    ampl_113 = result.params[prefix+'amplitude_113'].value
    return ampl_76, ampl_113  

def sidust_extractpars(prefix, result):
    """extract the parameters for pahdust model"""
    T = result.params[prefix+'T'].value
    ampl = result.params[prefix+'amplitude'].value
    return T,ampl 
#%%
#NOTE that these are ANALYTIC estimates of line fluxes,
#NOT available in astropy modeling or specutils

def gaussianline_flux(amp,std,cen):
    """calculate the integrated flux for a gaussian line"""
    #Units= amp:Jy cen:microns
    c= 29979.2458*10**10 #c in microns/s
    gaussianfactor=sqrt(2*pi)
    return amp*gaussianfactor*c*std/(cen**2)*10**(-23)      #erg s^{-1} cm^{-2}

def drudeline_flux(amp,frac,peak):
    """calculate the integrate flux for a drude line"""
    #Units= amp:Jy peak:microns
    c= 29979.2458*10**10 #c in microns/s
    drudefactor=pi
    return (amp*frac*drudefactor*c/((2*peak)))*10**(-23)   #erg s^{-1} cm^{-2}
#%%
def extract_spec(a, b):
    """extract spec and errors from a,b coordinates"""
    spec_pix=data_cube[:,a,b]
    err_pix=error_cube[:,a,b]
    return spec_pix, err_pix

def model_comps(model_result):
    """Evaluate model components"""
    comps = model_result.eval_components()
    plcomp=[]
    h2comp=[]
    pahcomp=[]
    sidustcomp=[]
    #print(comps.keys())
    for key in comps.keys(): 
        keyl=key.lower()
        if keyl[0:3]=='pwl': plcomp.append(key)
        if keyl[0]=='h': h2comp.append(key)
        if keyl[0:3]=='pah': pahcomp.append(key)
        if keyl[0:3]=='sid': sidustcomp.append(key)
       
    plaw_model=x-x
    h2_model=x-x
    ion_model=x-x
    pah_model=x-x
    sidust_model=x-x
    for comp in plcomp: plaw_model+=comps[comp]
    for comp in h2comp: h2_model+=comps[comp]
    for comp in pahcomp: pah_model+=comps[comp]
    for comp in sidustcomp: sidust_model+=comps[comp]
    atomiclines=['ArII','SIV','NeII']
    for comp in atomiclines:
        ion_model+=comps[comp+'_']
        
    return([plaw_model,h2_model,ion_model,pah_model,sidust_model],['PL','H2','Ions','PAHs','Si Dust'])

def plot_fit(x, spec, specerr, model_result):
    """plot spectrum, model components, and residual"""

    #Model Results
    fit_model = model_result.best_fit 
    fit_residual = spec - fit_model

    #Evaluate model components
    mod_comps,mod_labels=model_comps(model_result)
        
    #Spec-1D object
    spec1 = Spectrum1D(spectral_axis=x* u.um, flux=spec*u.Jy, uncertainty=StdDevUncertainty(specerr))
    with quantity_support():    
        f, ax = plt.subplots()  
        #ax.step, ax.plot, ax.scatter all work, but ax.scatter doesn't autscale correctly
        ax.plot(spec1.spectral_axis, spec1.flux,label='Data', color='k')
        #ax.set_xlabel(r"$Wavelength\ (\mu m)$")
        ax.grid(linestyle=':')
        #plt.errorbar only works on unitless data
        #ax.errorbar(spec1.spectral_axis, spec1.flux, yerr=specerr,label='Data', color='k')
        #ax.plot(spec1.spectral_axis, spec1.flux, yerr=specerr,label='Data', color='k')
        
        plt.show()   

    #Plot results
    fig1=plt.figure(figsize=(15, 6), dpi= 150)
    
    frame1=fig1.add_axes((.1,.3,.8,.6))
    frame1.set_xticklabels([])
    plt.ylabel(r"$Flux\ (Jy)$")
    plt.grid(linestyle=':')
    plt.errorbar(x, spec, yerr=specerr,label='Data', color='k')
    plt.plot(x, fit_model,label='Model',c='red')
    mod_colors=['brown','g','b','magenta','orange']
    for mcomp, mlabel, mcolr in zip(mod_comps,mod_labels,mod_colors):
        plt.plot(x, mcomp, label=mlabel, c=mcolr)        
    plt.legend()

    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(x, 0.*x, '-',c='r')
    plt.xlabel(r"$Wavelength\ (\mu m)$")
    plt.grid(linestyle=':')
    plt.errorbar(x,fit_residual,yerr=specerr,c='k')

    plt.show()     
    return 
#%%
# Spitzer IRS (CUBISM) cube loader does not exist in specutils

#Target 
targname='M58'

#Redshift
z=0.005060
OneZ=1.+z

#Download and open the data cubes and their uncertainties
BoxPath="https://data.science.stsci.edu/redirect/JWST/jwst-data_analysis_tools/cube_fitting/"
cubeSL1 = fits.open(BoxPath+targname+'_SL1_cube.fits') 
errorsSL1 = fits.open(BoxPath+targname+'_SL1_cube_unc.fits')
cubeSL2 = fits.open(BoxPath+targname+'_SL2_cube.fits') 
errorsSL2 = fits.open(BoxPath+targname+'_SL2_cube_unc.fits')

#Cube Info and Headers
cubeSL1.info()
cubeSL2.info()
hdr1 = cubeSL1[0].header                        
er_hdr1=errorsSL1[0].header
hdr2 = cubeSL2[0].header                               
er_hdr2=errorsSL2[0].header
#print(repr(hdr1))

#Flux Data
data_cube1 = cubeSL1[0].data
error_cube1 = errorsSL1[0].data
data_cube2 = cubeSL2[0].data
error_cube2 = errorsSL2[0].data

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
error_cube1=error_cube1*correct_unit1
correct_unit2=abs(hdr2['CDELT1'])*abs(hdr2['CDELT2'])*0.0003046118*(10**6)    
data_cube2=data_cube2*correct_unit2
error_cube2=error_cube2*correct_unit2

#Concatenate SL1 and SL2
data_cube=np.concatenate((data_cube2,data_cube1),axis=0)
error_cube=np.concatenate((error_cube2,error_cube1),axis=0)

#Reorder x and data_cube
xcor = x.argsort()
data_cube = data_cube[xcor]
error_cube = error_cube[xcor]
x=np.sort(x)

#Cube dimensions and trimming 
xsize, ysize, zsize = data_cube.shape
ytrim=0; ysize=ysize-ytrim
ztrim=3; zsize=zsize-ztrim
print('Trimmed cube dimensions:', xsize, ysize, zsize)

#Collapsed 2D image
cube_2dflux=np.sum(data_cube,axis=0)[0:ysize,0:zsize]

#Collapsed 1D spectrum
cube_1dflux=np.zeros(xsize,)
for a in arange(0,ysize):
    for b in arange(0,zsize):
        spec_pix,err_pix=extract_spec(a,b)
        cube_1dflux=cube_1dflux+spec_pix

f,(ax1,ax2)=plt.subplots(1,2, figsize=(10,5))
ax1.set_title('log Flux (Sum over Wavelength)')
ax1.imshow(cube_2dflux, origin='lower', cmap='gray', norm=LogNorm())
ax2.set_title('Flux (Sum over Spaxels)')
ax2.plot(x,cube_1dflux)
ax2.set_xlabel(r"Wavelength $(\mu m)$")
ax2.set_ylabel('Flux (Jy)')
plt.show()
#%%