# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 2016

@author: rachel, rachel.white@cantab.net

Note that pressure is non-dimensional, i.e. p = p/1000mb

Quasi-geostrophic formulation. Could calculate quasi-geostrophic streamfunction
from geopotential height. But we've shown that the solution is very close to being geostrophic, 
so it should be fine to use streamfunction calculated from wind components.

"""

import os, errno
from netCDF4 import Dataset
import netCDF4
import numpy as np
import datetime as dt
import Ngl

a = 6.37122e06  # radius of Earth
PI = 3.14159265358979
omega =  7.2921e-5
g = 9.80616
P0 = 1000.0
cp = 1.00464e3
Rd = 287.0
kappa = (Rd/cp)


def calcEP(indata,indataPS,BWs = False,indataV=0):
    try:
        THETA = indata['TH']
    except AttributeError:
        THETA = indata['T'] * np.power(indata['T'].lev_p/1000.0,(-1.0 * kappa))

    U = indata['U'] #.mean(dim='time')
    if indataV == 0:
        V = indata['V'] #.mean(dim='time')
    else: # U and V are in separate files
        V = indataV['V']

    if BWs: # account for the fact that U is now backwards
        U = -1.0 * U

    PS = indataPS['PS'] #.mean(dim='time')
    # mask where below ground
    nlevs = len(U.lev_p)
    nlons = len(U.lon)
    nlats = len(U.lat)
    ntimes = len(U.time)

    # mask if PS above level for any point, but

    for ilev in range(0,nlevs):
        plev = U.lev_p[ilev]
        U[:,ilev,...] = np.where(PS[...]/100. < plev,np.nan,U[:,ilev,...])
        V[:,ilev,...] = np.where(PS[...]/100. < plev,np.nan,V[:,ilev,...])
        THETA[:,ilev,...] = np.where(PS[...]/100. < plev,np.nan,THETA[:,ilev,...])

    THETAzm = THETA.mean(dim='lon')

    # Calculate zonal mean TH
    lat = THETAzm.lat
    level = THETAzm.lev_p

    # Calculate d(THETA)/dp on time mean fields from vertical finite differences in
    # log-pressure coordinates
    # noting that dT/dp = (1/p) * dT/d(lnp)
    loglevel = np.log(level)

    # find index that corresponds to levels    # find index that corresponds to latitude
    for idx in range(0,len(THETAzm.dims)):
        if THETAzm.dims[idx] == 'lev_p':
            idxlev = idx
    THETAp = np.gradient(THETAzm,axis=idxlev)/np.gradient(loglevel)[:,None]

    #THETAp = THETAzm.differentiate('lev_p')/np.gradient(loglevel)[:,None]

    THETAp = THETAp/(level.values[:,None]*100.0)

    if len(THETAzm.dims) == 2:
        THETAp = xr.DataArray(THETAp, coords=[THETA.lev_p, THETA.lat], dims=['lev_p', 'lat'])
    elif len(THETAzm.dims) == 3:
        THETAp = xr.DataArray(THETAp, coords=[THETA.time, THETA.lev_p, THETA.lat], dims=['time','lev_p', 'lat'])
    else:
        sys.exit('Unexpected number of dimensions in inputdata')

    Uzm = U.mean(dim='lon')
    Uza = U - Uzm
    Vza = V - V.mean(dim='lon')

    THETAza = THETA - THETAzm

    UV = Uza * Vza
    UVzm = UV.mean(dim='lon')
    VTHETA = Vza * THETAza
    VTHETAzm = VTHETA.mean(dim='lon')

    phi = np.deg2rad(U.lat)     # Get latitude in radians
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    acphi = a * cphi
    asphi = a * sphi
    f = 2*omega*sphi

    latfac = acphi * cphi

    Fphi = (-UVzm*latfac).mean(dim='time')

    Fp = ((f.values*acphi.values *VTHETAzm)/THETAp).mean(dim='time')

    # find index that corresponds to latitude
    for idx in range(0,len(Fphi.dims)):
        if Fphi.dims[idx] == 'lat':
            idxlat = idx

    EPdiv1 = np.gradient(Fphi, axis=idxlat)/np.gradient(asphi)

    # find index that corresponds to level
    for idx in range(0,len(Fphi.dims)):
        if Fphi.dims[idx] == 'lev_p':
            idxlev = idx
    EPdiv2 = np.gradient(Fp,axis=idxlev)/np.gradient(level*100.0)[:,None]

    # put into dataarrays
    EPdiv1 = xr.DataArray(EPdiv1, coords=[THETA.lev_p, THETA.lat], dims=['lev_p', 'lat'])
    EPdiv2 = xr.DataArray(EPdiv2, coords=[THETA.lev_p, THETA.lat], dims=['lev_p', 'lat'])

    # Add together derivative components
    EPdiv = EPdiv1 + EPdiv2

    # Compute acceleration from divF

    dudt = 86400.0 * EPdiv/acphi
    
    EP_CTL = xr.Dataset({'Fp': (['lev_int', 'lat'],  Fp),
                         'Fphi': (['lev_int', 'lat'],  Fphi),
                         'dudt': (['lev_int', 'lat'],  dudt),
                          'EPdiv':(['lev_int','lat'],EPdiv),
                          'EPdiv_H':(['lev_int','lat'],EPdiv1),
                          'EPdiv_z':(['lev_int','lat'],EPdiv2)},
                          coords={'lev_int': (['lev_int'], THETA.lev_p),
                                  'lat': (['lat'], THETA.lat),
                            })

    
    return(EP_CTL)



def EPfluxes_old(datain,datainPS):

    THETA = datain['TH'].mean(dim='time')
    U = datain['U'].mean(dim='time')
    V = datain['V'].mean(dim='time')

    PS = datainPS['PS'].mean(dim='time')

    # mask where below ground
    nlevs = len(U.lev_p)
    for ilev in range(0,nlevs):
        plev = U.lev_p[ilev]
        U[ilev,...] = np.where(PStoplot[...]/100. < plev,np.nan,U[ilev,...])
        V[ilev,...] = np.where(PStoplot[...]/100. < plev,np.nan,V[ilev,...])

        THETA[ilev,...] = np.where(PStoplot[...]/100. < plev,np.nan,THETA[ilev,...])

    # Calculate zonal mean TH
    THETAzm = THETA.mean(dim='lon')

    lat = THETAzm.lat
    level = THETAzm.lev_p

    # Calculate d(THETA)/dp on time mean fields from vertical finite differences in
    # log-pressure coordinates
    # noting that dT/dp = (1/p) * dT/d(lnp)
    loglevel = np.log(level)

    THETAp = np.gradient(THETAzm,axis=0)/np.gradient(loglevel)[:,None]

    THETAp = THETAp/(level.values[:,None]*100.0)

    THETAp = xr.DataArray(THETAp, coords=[THETA.lev_p, THETA.lat], dims=['lev_p', 'lat'])

    Uza = U - U.mean(dim='lon')
    Vza = V - V.mean(dim='lon')

    THETAza = THETA - THETA.mean(dim='lon')

    UV = Uza * Vza
    UVzm = UV.mean(dim='lon')

    VTHETA = Vza * THETAza
    VTHETAzm = VTHETA.mean(dim='lon')

    # Calculate latitude values
    phi = np.deg2rad(U.lat)     # Get latitude in radians
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    acphi = a * np.cos(phi)
    asphi = a * np.sin(phi)
    f = 2*omega*np.sin(phi)

    latfac = acphi * np.cos(phi)

    Fphi = -UVzm*latfac

    Fp = (f.values*acphi.values *VTHETAzm)/THETAp

    EPdiv1 = np.gradient(Fphi, axis=1)/np.gradient(asphi)

    # take derivate with respect to pressure
    # Pressure in pascals

    EPdiv2 = np.gradient(Fp,axis=0)/np.gradient(level*100.0)[:,None]

    # put into dataarrays
    EPdiv1 = xr.DataArray(EPdiv1, coords=[THETA.lev_p, THETA.lat], dims=['lev_p', 'lat'])
    EPdiv2 = xr.DataArray(EPdiv2, coords=[THETA.lev_p, THETA.lat], dims=['lev_p', 'lat'])

    # Add together derivative components
    EPdiv = EPdiv1 + EPdiv2

    # Compute acceleration from divF

    dudt = 86400.0 * EPdiv/acphi

