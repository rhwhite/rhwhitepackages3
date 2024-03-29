
"""
Functions on gridded data files

Created 13 Aug 2019

Author: Rachel White rachel.white@cantab.net
"""

import os, errno
import netCDF4
import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr
import calendar
from rhwhitepackages3.CESMconst import *
from rhwhitepackages3.CESMmanip import *

def ddy_merc(invar):
    # based on Hoskins and Karoly:
    # https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281981%29038%3C1179%3ATSLROA%3E2.0.CO%3B2
    #  ddy = cos(phi)/a ddphi

    try:
        nlats = len(invar['lat'])
        latname = 'lat'
        lats = invar['lat']
    except KeyError:
        nlats = len(invar['latitude'])
        latname = 'latitude'
        lats = invar['latitude']

    phi = np.deg2rad(lats)
    cosphi = np.cos(phi).values

    dims = invar.shape

    dvardy = invar.copy(deep=True)

    if latname == 'lat':
        dims_var = invar.dims
        latidx_var = dims_var.index('lat')

        dims_lat = invar.lat.dims
        latidx_lat = dims_lat.index('lat')

        ddy = invar.copy(deep=True)
        ddy.isel(lat=0)[...] = 0

        dvar = np.gradient(invar,axis=latidx_var)
        dphi = np.gradient(phi,axis=latidx_lat)
        if len(dims_var) > 1:
            dvardy[...] = (dvar/dphi[:,None]) * (cosphi[:,None] / rearth)
        else:
            dvardy[...] = (dvar/dphi) * (cosphi / rearth)

    elif latname == 'latitude':
        dims_var = invar.dims
        latidx_var = dims_var.index('latitude')

        dims_lat = invar.latitude.dims
        latidx_lat = dims_lat.index('latitude')

        ddphi = invar.copy(deep=True)
        ddphi.isel(latitude=0)[...] = 0

        dvar = np.gradient(invar,axis=latidx_var)
        dphi = np.gradient(phi,axis=latidx_lat)

        if len(dims_var) > 1:
            dvardy[...] = (dvar/dphi[:,None]) * (cosphi[:,None] / rearth)
        else:
            dvardy[...] = (dvar/dphi) *(cosphi / rearth)

    return(dvardy)


def ddy_vect(invar):
    # based on https://www.ncl.ucar.edu/Document/Functions/Built-in/uv2dv_cfd.shtml
    # H.B. Bluestein [Synoptic-Dynamic Meteorology in Midlatitudes, 1992, Oxford Univ. Press p113-114]

    # Using this rather ugly approach to allowing different names for the latitude variable
    # Because of the way I have written the xarray code to use variable names, names are hard-coded in places
    try:
        nlats = len(invar['lat'])
        latname = 'lat'
    except KeyError:
        nlats = len(invar['latitude'])
        latname = 'latitude'

    dims = invar.shape
    if len(dims) == 1:
        # treat array values differently if it is a 1D array
        if latname == 'lat':
            ddy = invar.copy(deep=True)
            ddy[:] = np.nan

            ddy.isel(lat=0).values = 0

            for ilat in range(1,nlats-1):

                dy = getlatdist(invar['lat'].isel(lat = ilat+1),
                                    invar['lat'].isel(lat = ilat-1))
                dvar = invar.isel(lat = ilat+1) - invar.isel(lat = ilat-1)
                ddy.isel(lat = ilat).values = dvar/dy - (invar.isel(lat=ilat)/rearth) * np.tan(np.deg2rad(invar.lat.isel(lat=ilat)))

            ddy.isel(lat=nlats-1).values = 0

        elif latname == 'latitude':
            ddy = invar.copy(deep=True)
            ddy[:] = np.nan
            ddy[0] = 0

            for ilat in range(1,nlats-1):
                dy = getlatdist(invar['latitude'].isel(latitude = ilat+1),
                                    invar['latitude'].isel(latitude = ilat-1))
                dvar = invar.isel(latitude = ilat+1) - invar.isel(latitude = ilat-1)
                ddy[ilat] = dvar/dy - (invar.isel(latitude=ilat)/rearth) * np.tan(np.deg2rad(invar.latitude.isel(latitude=ilat)))

            ddy[nlats-1] = 0
    else:
        # more than 1D array:
        if latname == 'lat':
            ddy = invar.copy(deep=True)
            ddy.isel(lat=0)[...] = 0

            for ilat in range(1,nlats-1):

                dy = getlatdist(invar['lat'].isel(lat = ilat+1),
                                    invar['lat'].isel(lat = ilat-1))
                dvar = invar.isel(lat = ilat+1) - invar.isel(lat = ilat-1)
                ddy.isel(lat = ilat)[...] = dvar/dy - (invar.isel(lat=ilat)/rearth) * np.tan(np.deg2rad(invar.lat.isel(lat=ilat)))

            ddy.isel(lat=nlats-1)[...] = 0

        elif latname == 'latitude':
            ddy = invar.copy(deep=True)
            ddy[...] = np.nan
            ddy.isel(latitude=0)[...] = 0

            for ilat in range(1,nlats-1):
                dy = getlatdist(invar['latitude'].isel(latitude = ilat+1),
                                    invar['latitude'].isel(latitude = ilat-1))
                dvar = invar.isel(latitude = ilat+1) - invar.isel(latitude = ilat-1)
                ddy.isel(latitude = ilat)[...] = dvar/dy - (invar.isel(latitude=ilat)/rearth) * np.tan(np.deg2rad(invar.latitude.isel(latitude=ilat)))

            ddy.isel(latitude=nlats-1)[...] = 0
    return(ddy)


def ddphi(invar):
    # in spherical coordinate, del = 1/r d/dphi
    a = 6.3781E6

    try:
        lats_r = np.deg2rad(invar.latitude)
    except AttributeError:
        lats_r = np.deg2rad(invar.lat)

    try:
        lat_r = U_testing.lat
        dvar = U_testing.differentiate('lat') 
        dphi = lats_r.differentiate('lat')
        dvardphi = (dvar/dphi)*1/a 
    except AttributeError:
        lat_r = U_testing.latitude       
        dvar = U_testing.differentiate('latitude') 
        dphi = lats_r.differentiate('latitude')
        dvardphi = (dvar/dphi)*1/a     
        
    return(dvardphi)

def ddx(invar):
    # based on https://www.ncl.ucar.edu/Document/Functions/Built-in/uv2dv_cfd.shtml
    # H.B. Bluestein [Synoptic-Dynamic Meteorology in Midlatitudes, 1992, Oxford Univ. Press p113-114]
    nlons = len(invar['lon'])
    nlats = len(invar['lat'])
    ddx = invar.copy(deep=True)
    #ddx.isel(lat=0)[...] = 0

    for ilat in range(1,nlats-1):
        for ilon in range(1,nlons-1):
            dx = getlondist(invar['lon'].isel(lon = ilon+1),
                            invar['lon'].isel(lon = ilon-1),
                            invar['lat'].isel(lat=ilat))
            dvar = invar.isel(lon = ilon+1).isel(lat=ilat) - invar.isel(lon = ilon-1).isel(lat=ilat)
            ddx.isel(lat = ilat)[...] = dvar/dx

        # Now cover ilon = 0 and ilon = nlons-1
        ilon = 0
        dx = getlondist(invar['lon'].isel(lon = ilon+1),
                            invar['lon'].isel(lon = nlons-1),
                            invar['lat'].isel(lat=ilat))
        dvar = invar.isel(lon = ilon+1).isel(lat=ilat) - invar.isel(lon = nlons-1).isel(lat=ilat)
        ddx.isel(lat = ilat)[...] = dvar/dx

        ilon= nlons-1
        dx = getlondist(invar['lon'].isel(lon = 0),
                            invar['lon'].isel(lon = ilon),
                            invar['lat'].isel(lat=ilat))
        dvar = invar.isel(lon = 0).isel(lat=ilat) - invar.isel(lon = ilon-1).isel(lat=ilat)
        ddx.isel(lat = ilat)[...] = dvar/dx

    return(ddx)

def ddp(invar):
    dims = invar.dims
    try:
        levidx = dims.index('lev_p')
        levels = invar.lev_p * 100.0
    except ValueError:
        levidx = dims.index('lev_int')
        levels = invar.lev_int * 100.0

    dvar = np.gradient(invar,axis=levidx)
    dlev = np.gradient(levels)

    if len(invar.dims) == 2:
        da = xr.DataArray(dvar/dlev[:,None], coords=[levels, invar.lat], dims=['lev_p', 'lat'])

    elif len(invar.dims) == 3:
        da = xr.DataArray(dvar/dlev, coords=[invar.time,levels, invar.lat], dims=['time','lev_p', 'lat'])
    elif len(invar.dims) == 4:
        da = xr.DataArray(dvar/dlev, coords=[invar.time,levels, invar.lat,invar.lon], 
                          dims=['time','lev_p', 'lat','lon'])

    return(da)
