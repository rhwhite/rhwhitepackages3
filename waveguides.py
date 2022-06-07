# Module identify waveguides in refractive index data
# Written by rhwhite rachel.white@cantab.net
import numpy as np
import xarray as xr
import math
from scipy.interpolate import interp1d

# Look for waveguides by searching for turning points at specific wavenumbers
def iden_waveguide_TPs_map(inKs2,inU,ninterp,Uthresh,WGwidth,WGdepth,wnstart,wnend,nwgs):
    orig_lats = inKs2.latitude
    # latitudes for cubic spline interpolation
    lats_li = np.linspace(np.amin(orig_lats),np.amax(orig_lats),ninterp)

    TPs = {}
    WGlatmin = np.ndarray([nwgs,wnend - wnstart+1])
    WGlatmax = np.ndarray([nwgs,wnend - wnstart+1])
    WGlatmin[...] = np.nan
    WGlatmax[...] = np.nan


    # define waveguide map so latitude indexing works easily
    nlats = 90
    nk = wnend - wnstart+1
    WGmap = np.ndarray([nk,nlats])

    wg_map = xr.DataArray(WGmap,coords={
                       'k':np.arange(wnstart,wnend+1),
                       'latitude':np.linspace(0,90,nlats)},
                        dims=('k','latitude'))
    wg_map[...] = 0

    if np.any(np.isfinite(inKs2.values)): # only if there are some non-nan values
        LI = interp1d(orig_lats[::-1],inKs2[::-1])
        LI_U = interp1d(orig_lats[::-1],inU[::-1])

        Ks2_li = xr.DataArray(LI(lats_li),coords={'latitude':lats_li},dims = ('latitude'))
        U_li = xr.DataArray(LI_U(lats_li),coords={'latitude':lats_li},dims = ('latitude'))

        # For each wavenumber (5,6,7,8) find the turning points

        for iwn in range(wnstart,wnend+1):
            Ks2_wn = (Ks2_li - iwn**2)
            # Find local minima
            TPs[iwn] = np.where(np.diff(np.sign(Ks2_wn)))[0]

            # For each pair of TPs, check if it's a waveguide
            iwg = 0
            nTPs = len(TPs[iwn])
            for iTP in np.arange(0,nTPs-1):
                # Width of waveguide criteria:
                TPlat1 = Ks2_li.latitude[TPs[iwn][iTP]]
                TPlat2 = Ks2_li.latitude[TPs[iwn][iTP+1]+1] # add 1 as the turning points finds the latitude before
                if np.abs(TPlat1 - TPlat2) >= WGwidth:
                    # U threshold criteria
                    if ~np.any(U_li[TPs[iwn][iTP]:TPs[iwn][iTP+1]].values < Uthresh):
                        # Waveguide depth critera:
                        if np.any(Ks2_li[TPs[iwn][iTP]+1:TPs[iwn][iTP+1]-1].values > (iwn+WGdepth)**2):
                            WGlatmin[iwg,iwn-wnstart] = TPlat1
                            WGlatmax[iwg,iwn-wnstart] = TPlat2
                            iwg +=1

                            wg_map.sel(k=iwn).sel(latitude=slice(TPlat1,TPlat2)).values[...] = 1

    return(WGlatmin,WGlatmax,wg_map.values)



# Look for waveguides by searching for turning points at specific wavenumbers
def iden_waveguide_TPs(inKs2,inU,ninterp,Uthresh,WGwidth,WGdepth,wnstart,wnend,nwgs):
    orig_lats = inKs2.latitude
    # latitudes for cubic spline interpolation
    lats_li = np.linspace(np.amin(orig_lats),np.amax(orig_lats),ninterp)

    TPs = {}
    WGlatmin = np.ndarray([nwgs,wnend - wnstart+1])
    WGlatmax = np.ndarray([nwgs,wnend - wnstart+1])
    WGlatmin[...] = np.nan
    WGlatmax[...] = np.nan

    if np.any(np.isfinite(inKs2.values)): # only if there are some non-nan values
        LI = interp1d(orig_lats[::-1],inKs2[::-1])
        LI_U = interp1d(orig_lats[::-1],inU[::-1])

        Ks2_li = xr.DataArray(LI(lats_li),coords={'latitude':lats_li},dims = ('latitude'))
        U_li = xr.DataArray(LI_U(lats_li),coords={'latitude':lats_li},dims = ('latitude'))

        # For each wavenumber (5,6,7,8) find the turning points

        for iwn in range(wnstart,wnend+1):
            Ks2_wn = (Ks2_li - iwn**2)
            # Find local minima
            TPs[iwn] = np.where(np.diff(np.sign(Ks2_wn)))[0]

            # For each pair of TPs, check if it's a waveguide
            iwg = 0
            nTPs = len(TPs[iwn])
            for iTP in np.arange(0,nTPs-1):
                # Waveguide depth critera:
                if np.any(Ks2_li[TPs[iwn][iTP]+1:TPs[iwn][iTP+1]-1].values > (iwn+WGdepth)**2):
                    # U threshold criteria
                    if ~np.any(U_li[TPs[iwn][iTP]:TPs[iwn][iTP+1]].values < Uthresh):
                        # Width of waveguide criteria:
                        TPlat1 = Ks2_li.latitude[TPs[iwn][iTP]]
                        TPlat2 = Ks2_li.latitude[TPs[iwn][iTP+1]+1] # add 1 as the turning points finds the latitude before
                        if np.abs(TPlat1 - TPlat2) >= WGwidth:
                            WGlatmin[iwg,iwn-wnstart] = TPlat1
                            WGlatmax[iwg,iwn-wnstart] = TPlat2
                            iwg +=1

    return(WGlatmin,WGlatmax)

# count_waveguides and write out magnitude and latitude
# Only looking for waveguides at 2 specific latitudes
def identify_waveguides_mag(datain,wavenumber):

    wnbrs = wavenumber * wavenumber

    temp = datain.copy(deep=True)

    # Subtropical waveguide
    # Look for a value of wnbrs between 30 and 36
    # Look for at least 6deg (3 consecutive gridboxes) positive values between 34 and 46
    # Look for a value of wnbrs between 44 and 60
    wguideST = np.zeros(len(temp.longitude))
    wguideML = np.zeros(len(temp.longitude))

    wguideSTlat = np.zeros(len(temp.longitude))
    wguideMLlat = np.zeros(len(temp.longitude))
    
    for ilon in range(0,len(temp.longitude)):
        tempsel = temp.isel(longitude=ilon).copy(deep=True)
        # Cut off all values below wnbrs:
        tempsel[:] = np.where(tempsel<wnbrs,wnbrs,tempsel)    
        if np.any(tempsel.sel(latitude=slice(36,30)) == wnbrs):
            for ilat in range(34,46):
                result = np.all(tempsel.sel(latitude=slice(ilat+6,ilat)) > wnbrs)
                if result:
                    if np.any(tempsel.sel(latitude = slice(60,44)) == wnbrs):
                        wguideST[ilon] = np.amax(tempsel.sel(latitude=slice(46,34))) - wnbrs
                        templats = tempsel.sel(latitude=slice(52,34))
                        wguideSTlat[ilon] = templats.latitude.isel(latitude = np.argmax(templats.values))
                        break
                    else:
                        wguideST[ilon] = 0
            else:
                wguideST[ilon] = 0
        else:
            wguideST[ilon] = 0

        if np.any(tempsel.sel(latitude=slice(50,40)) == wnbrs):
            for ilat in range(48,60):
                result = np.all(tempsel.sel(latitude=slice(ilat+6,ilat)) > wnbrs)
                if result:
                    if np.any(tempsel.sel(latitude = slice(70,58)) == wnbrs):
                        wguideML[ilon] = np.amax(tempsel.sel(latitude=slice(60,48))) - wnbrs
                        templats = tempsel.sel(latitude=slice(66,48))
                        wguideMLlat[ilon] = templats.latitude.isel(latitude = np.argmax(templats.values))
                        break
                    else:
                        wguideML[ilon] = 0
            else:
                wguideML[ilon] = 0
        else:
            wguideML[ilon] = 0

    return(wguideST,wguideSTlat,wguideML,wguideMLlat)

# Identify waveguides in data, only looking for waveguides at certain latitudes, can't stray from this!
def waveguide_analysis_fixedlat(Ks2in,wnb):
    maxKs = 200
    Ks2in[...] = np.where(Ks2in < maxKs,Ks2in,maxKs)
    Ks2in[...] = np.where(Ks2in > -maxKs,Ks2in,-maxKs)

    waveguide_freq_ST = np.zeros([len(Ks2in.time),len(Ks2in.longitude)])
    waveguide_freq_ML = np.zeros([len(Ks2in.time),len(Ks2in.longitude)])
    waveguide_lat_ST = np.zeros([len(Ks2in.time),len(Ks2in.longitude)])
    waveguide_lat_ML = np.zeros([len(Ks2in.time),len(Ks2in.longitude)])
    
    for itime in range(0,len(Ks2in.time)):
    #    if itime % 100 == 0: print itime
                   
        waveguide_freq_ST[itime],waveguide_lat_ST[itime],waveguide_freq_ML[itime],waveguide_lat_ML[itime] = identify_waveguides_mag(Ks2in.isel(time=itime),wnb)
    
    DA_ST = xr.DataArray(waveguide_freq_ST, coords=[Ks2in.time, Ks2in.longitude], dims=['time', 'longitude'])
    DA_ML = xr.DataArray(waveguide_freq_ML, coords=[Ks2in.time, Ks2in.longitude], dims=['time', 'longitude'])
    DA_STlat = xr.DataArray(waveguide_lat_ST, coords=[Ks2in.time, Ks2in.longitude], dims=['time', 'longitude'])
    DA_MLlat = xr.DataArray(waveguide_lat_ML, coords=[Ks2in.time, Ks2in.longitude], dims=['time', 'longitude'])

    DS_ST = DA_ST.to_dataset(name='waveguide_freq_ST')
    DS_ML = DA_ML.to_dataset(name='waveguide_freq_ML')
    DS_STlat = DA_STlat.to_dataset(name='waveguide_lat_ST')
    DS_MLlat = DA_MLlat.to_dataset(name='waveguide_lat_ML')
    
    towrite = xr.merge([DS_ST, DS_ML,DS_STlat,DS_MLlat])
    del towrite.time.encoding["contiguous"]
    
    return(towrite) 
