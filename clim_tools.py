"""
Functions to create, smooth, interpolate etc climatologies

Created 14 Nov 2017

Author: Rachel White rachel.white@cantab.net
"""
import os, errno
import netCDF4
import numpy as np
import datetime as dt
import calendar as cal
import pandas as pd
import xarray as xr
#import Ngl
import math
import scipy
from scipy import stats
from scipy.interpolate import interp1d
from scipy import interpolate

from rhwhitepackages3.readwrite import shiftlons
from rhwhitepackages3.readwrite import xrOpen, xrMfOpen
from rhwhitepackages3.stats import regressmaps

def smooth_clim_from_monthly(varin,ndaysyear,yearin,typein):
    nmonths = 36
    startmonth = 12
    ndays = np.zeros(nmonths)
    ndaypre = 0

    for imonth in range(0,12):
        month1 = imonth + 1
        nday1 = cal.monthrange(yearin,month1)[1]
        try:
            ndaypre = ndaypre + cal.monthrange(yearin,month1-1)[1]
        except ValueError:
            ndays[startmonth + imonth] = nday1/2.0
        else:
            ndays[startmonth + imonth] = ndaypre + nday1/2.0

    # Add on previous year
    for imonth in range(0,12):
        ndays[imonth] = ndays[imonth + 12] - ndaysyear
    # Add on next years months
    for imonth in range(24,36):
        ndays[imonth] = ndays[imonth-12] + ndaysyear

    clim_cont = xr.concat([varin, varin, varin],dim='month')

    f = interp1d(ndays,clim_cont,kind=typein,axis=0)

    days = np.arange(0,ndaysyear)
    return f(days)

def smooth_clim_from_daily(varin,ndaysyear,startmonth,endmonth):
    ## find index of first month and last month
    if varin.dims[1] not in ['lat','lats','latitude','latitudes']:
        print('expecting 2nd dimension to be latitude, it\'s not!')
        return(None,None)
    if varin.dims[2] not in ['lon','lons','longitude','longitudes']:
        print('expecting 3rd dimension to be longitude, it\'s not!')
        return(None,None)

    try:
        lats = varin.latitude
    except KeyError:
        try:
            lats = varin.lat
        except KeyError:
            lats = varin.lats

    try:
        lons = varin.longitude
    except KeyError:
        try:
            lons = varin.lon
        except KeyError:
            lons = varin.lons

    months = varin.time.values.astype('datetime64[M]').astype(int) % 12 + 1
    ndays = len(months)

    years = 0
    starts=[]
    ends=[]

    istart = 0
    # Loop through all full years, and add values to datasum (a dataarray)
    while istart < ndays-ndaysyear:
        while months[istart] != startmonth:
            istart += 1

        iend = istart + ndaysyear
        if years == 0:
            datasum = varin.isel(time=slice(istart,iend))
        else:
            datasum = datasum + varin.isel(time=slice(istart,iend)).values

        years += 1
        starts.append(istart)
        ends.append(iend)
        istart = iend

    # create climatology
    dataclim = datasum/np.float(years)

    # Select values used in climatology, starting with startmonth
    datasel = varin.isel(time=slice(starts[0],ends[-1]))

    # Create smooth climatology
    # Using a savgol filter with a window of 51, and a polynomial or order 3
    smooth_clim = np.zeros(dataclim.shape)

    nlats = dataclim.shape[1]
    nlons = dataclim.shape[2]

    for ilat in range(0,nlats):
        for ilon in range(0,nlons):
            smooth_clim[:,ilat,ilon] = scipy.signal.savgol_filter(
                            dataclim.isel(longitude=ilon,latitude=ilat),
                            window_length=51, polyorder=3)


    # Now run through each year, subtract this seasonal cycle
    data_years = {}
    for iyear in range(0,years):
        data_years[iyear] = datasel[iyear*365:(iyear+1)*365] - smooth_clim

    data_anom = xr.concat(data_years.values(), dim='time')

    # Make smooth_clim into a dataset
    xr_clim = xr.DataArray(smooth_clim,dims=('day','lat','lon'),
                            coords={'day':np.arange(1,ndaysyear+1),'lat':lats.values,'lon':lons.values})

    return data_anom,xr_clim


def smooth_clim_from_daily_1D(varin,ndaysyear,startmonth):
    ## find index of first month and last month
    months = varin.time.values.astype('datetime64[M]').astype(int) % 12 + 1
    ndays = len(months)

    years = 0
    starts=[]
    ends=[]

    istart = 0
    # Loop through all full years, and add values to datasum (a dataarray)
    while istart <= ndays-ndaysyear:
        while months[istart] != startmonth:
            istart += 1

        iend = istart + ndaysyear
        if years == 0:
            datasum = varin.isel(time=slice(istart,iend))
        else:
            datasum = datasum + varin.isel(time=slice(istart,iend)).values

        years += 1
        starts.append(istart)
        ends.append(iend)
        istart = iend
    # create climatology
    dataclim = datasum/np.float(years)

    # Select values used in climatology, starting with startmonth
    datasel = varin.isel(time=slice(starts[0],ends[-1]))

    # Create smooth climatology
    # Using a savgol filter with a window of 51, and a polynomial or order 3
    smooth_clim = scipy.signal.savgol_filter(dataclim,
                            window_length=51, polyorder=3)


    # Now run through each year, subtract this seasonal cycle
    data_years = {}
    for iyear in range(0,years):
        data_years[iyear] = datasel[iyear*ndaysyear:(iyear+1)*ndaysyear] - smooth_clim

    data_anom = xr.concat(data_years.values(), dim='time')

    return data_anom,smooth_clim

