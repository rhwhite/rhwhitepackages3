# Module identify waveguides in refractive index data
# Written by rhwhite rachel.white@cantab.net
import numpy as np
import xarray as xr
import math
from scipy.interpolate import interp1d
from datetime import date

# Look for waveguides by searching for turning points at specific wavenumbers
def iden_waveguide_TPs_map(inKs2,inU,Uthresh,minwidth,mindepth,wnstart,wnend,nwgs,toprint=False,SH=False):
    # Check that latitudes are the correct way round!
    if inKs2.latitude[0] > inKs2.latitude[1]:
        exit('latitudes must be south to north')

    TPs = {}
    WGlatmin = np.ndarray([nwgs,wnend - wnstart+1])
    WGlatmax = np.ndarray([nwgs,wnend - wnstart+1])
    WGdepth = np.ndarray([nwgs,wnend - wnstart+1])
    WGwidth = np.ndarray([nwgs,wnend - wnstart+1])
   
    WGlatmin[...] = np.nan
    WGlatmax[...] = np.nan
    WGdepth[...] = np.nan
    WGwidth[...] = np.nan

    # define waveguide map so latitude indexing works easily
    nlats = 90
    nk = wnend - wnstart+1

    if SH:
        wg_map = xr.DataArray(np.ndarray([nk,nlats]),coords={
                       'k':np.arange(wnstart,wnend+1),
                       'latitude':np.arange(-90.0,0.0)},
                        dims=('k','latitude'))
    else:
        wg_map = xr.DataArray(np.ndarray([nk,nlats]),coords={
                       'k':np.arange(wnstart,wnend+1),
                       'latitude':np.arange(0.0,90.0)},
                        dims=('k','latitude'))

    wg_map[...] = 0

    if np.any(np.isfinite(inKs2.values)): # only if there are some non-nan values

        # For each wavenumber find the turning points
        for iwn in range(wnstart,wnend+1):
            if toprint: print('wavenumber:' + str(iwn))
            Ks2_wn = (inKs2 - iwn**2)
            # Find local minima
            TPs[iwn] = np.where(np.diff(np.sign(Ks2_wn)))[0]
            # these are always the points BEFORE the change in sign
            # So for TP2, we want to add 1 to the index
            if toprint: print(TPs[iwn])

            # For each pair of TPs, check if it's a waveguide
            iwg = 0
            nTPs = len(TPs[iwn])
            for iTP in np.arange(0,nTPs-1):
                if toprint:
                    print('turning point ' + str(iTP))
                # check Ks2 exceeds threshold everywhere between TPs:
                # Add 1 from first TP because by definition the values AT the 
                # turning points are less than or equal to the iwn**2
                # Add 1 from second TP because slice doesn't include the last value 
                if ~np.any(inKs2[TPs[iwn][iTP]+1:TPs[iwn][iTP+1]+1].values < iwn**2):
		    #Set inside edge TPs for this waveguide:
                    TP1 = TPs[iwn][iTP] + 1 # add 1 as the code find the point BEFORE it turns INTO waveguide
                    TP2 = TPs[iwn][iTP + 1] # don't add one, as this is BEFORE it turns OUT of waveguide
                    
                    # Calculate waveguide depth in Ks2
                    max_Ks2 = np.amax(inKs2[TP1:TP2 + 1].values) # add 1 to include last value
                    if toprint:
                        print("Ks2 and threshold:")
                        print(max_Ks2)
                        print(iwn+mindepth)
                    # Apply waveguide depth critera:
                    if max_Ks2 >= (iwn+mindepth)**2:
                        # U threshold criteria - WITHIN waveguide (i.e. don't include outside edges)
                        if toprint: 
                            print('U values for threshold check')
                            print(inU[TP1:TP2+1].values)
                        if ~np.any(inU[TP1:TP2+1].values < Uthresh):

                            # Width of waveguide criteria, in degrees latitude:
                            # Strict definition: width of waveguide is at least this width
                            TPlat1 = inKs2.latitude[TP1].values
                            TPlat2 = inKs2.latitude[TP2].values # 

                            if np.abs(TPlat1 - TPlat2) >= minwidth:
                                if toprint: 
                                    print('WAVEGUIDE!!')
                                    print('TPs:')
                                    print(TPlat1,TPlat2)
                                    print((TPlat1 - TPlat2))
                                WGlatmin[iwg,iwn-wnstart] = TPlat1
                                WGlatmax[iwg,iwn-wnstart] = TPlat2
                                WGdepth[iwg,iwn-wnstart] = np.sqrt(max_Ks2) - iwn
                                WGwidth[iwg,iwn-wnstart] = np.abs(TPlat1 - TPlat2)

                                iwg +=1 

                                # .sel includes the end value, so we don't need to add anything here (isel does not)
                                Ks2_inwg = inKs2.sel(latitude = slice(TPlat1,TPlat2))
                                depth = np.sqrt(Ks2_inwg) - iwn
                                if np.any(depth < 0):
                                    exit('something has gone wrong, depth is negative')
                                if toprint: print(TPlat1,TPlat2)
                                
                                wg_map.sel(k=iwn).sel(latitude=slice(TPlat1,TPlat2)).values[...] = depth
                                    
                                
                            else:
                                if toprint: print('not a waveguide, didn\'t pass width test')
                        else:
                            if toprint: print('not a waveguide, didn\'t pass U threshold test')

                else:
                    if toprint: print('not a waveguide, didn\'t pass Ks exceeds everywhere threshold')

    return(WGlatmin,WGlatmax,WGdepth,WGwidth,wg_map.values)

def calc_waveguide_map(Uin,Ksin,Ks2in,latstart,latend,
                        Uthresh,wnstart,wnend,minwidth,mindepth,
                        Uname,interp =False,toprint=False,SH=False):

    # Get data to make files
    ntimes = len(Ks2in.time)
    try:
        nlons = len(Ks2in.longitude)
        ZM=False
    except AttributeError:
        ZM=True
        nlons=1
        
    # Set maximum number of waveguides that can exist at any longitude:
    nwgs = 10

    WGlatmin = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])
    WGlatmax = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])
    WGdepth = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])
    WGwidth = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])

    WGlatmin[...] = np.nan
    WGlatmax[...] = np.nan
    WGdepth[...] = np.nan
    WGwidth[...] = np.nan

    # Check which way the latitudes are and flip indata if necessary
    if Ks2in.isel(latitude=0).latitude.values > Ks2in.isel(latitude=1).latitude.values:
        Ks2in = Ks2in.isel(latitude=slice(None, None, -1))
    if Ksin.isel(latitude=0).latitude.values > Ksin.isel(latitude=1).latitude.values:
        Ksin = Ksin.isel(latitude=slice(None, None, -1))
    if Uin.isel(latitude=0).latitude.values > Uin.isel(latitude=1).latitude.values:
        Uin = Uin.isel(latitude=slice(None, None, -1))

    lat1 = latstart
    lat2 = latend
    
    # define waveguide map so latitude indexing works easily
    nlats = 90
    nk = wnend - wnstart+1

    WGmap = np.ndarray([ntimes,nk,nlats,nlons])

    # define map as xarray so we can reference latitudes more easily
    if SH:
        xr_wg_map = xr.DataArray(WGmap,coords={
                       'time':Ks2in.time,'k':np.arange(wnstart,wnend+1),
                       'longitude':Ks2in.longitude,'latitude':np.linspace(-90,0,nlats)},
                        dims=('time','k','latitude','longitude'))
    else:
        xr_wg_map = xr.DataArray(WGmap,coords={
                       'time':Ks2in.time,'k':np.arange(wnstart,wnend+1),
                       'longitude':Ks2in.longitude,'latitude':np.linspace(0,90,nlats)},
                        dims=('time','k','latitude','longitude'))
    xr_wg_map[...] = 0
    
    for timein in np.arange(0,ntimes):
        if timein%100 == 0: print(timein)

        for ilon in np.arange(0,nlons):
            if toprint: print('longitude: ' + str(ilon))
            if ZM: # then zonal mean
                Ks2_sel = Ks2in.isel(time=timein).sel(latitude=slice(lat1,lat2)).copy(deep=True).sel(level=250)
                U_sel = Uin.isel(time=timein).sel(latitude=slice(lat1,lat2)).copy(deep=True).sel(level=250)

            else: # select longitude
                Ks2_sel = Ks2in.isel(time=timein).isel(longitude=ilon).sel(latitude = slice(lat1,lat2)).squeeze()
                U_sel = Uin.isel(time=timein).isel(longitude=ilon).sel(latitude = slice(lat1,lat2)).squeeze()

                
            (WGlatmin[:,:,ilon,timein],WGlatmax[:,:,ilon,timein],WGdepth[:,:,ilon,timein],WGwidth[:,:,ilon,timein],
                xr_wg_map.isel(time=timein).isel(longitude=ilon).values[...]) = iden_waveguide_TPs_map(Ks2_sel,U_sel,
                    Uthresh=Uthresh,minwidth=minwidth,mindepth=mindepth,
                    wnstart=wnstart,wnend=wnend,nwgs=nwgs,toprint=toprint,SH=SH)
            
    ## Make dataset
    if ZM:
        longitudes_out = [0]
    else:
        longitudes_out = Ks2in.longitude

    xr_wg_minlat = xr.DataArray(WGlatmin,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))

    xr_wg_maxlat = xr.DataArray(WGlatmax,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))    

    xr_wg_depth = xr.DataArray(WGdepth,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))    
    xr_wg_width = xr.DataArray(WGwidth,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))    
             
    # Convert to datasets and append
    ds = xr_wg_minlat.to_dataset(name = 'WG_min_lat')
    ds['WG_max_lat'] = xr_wg_maxlat
    ds['WG_depth'] = xr_wg_depth
    ds['WG_width'] = xr_wg_width

    ds.attrs['history'] = ('Created on ' + str(date.today()) + ' by calc_waveguide_map in rhwhitepackages3')
    ds.attrs['input data'] = ('Input files = ' + Uname)
    ds.attrs['parameters'] = ('waveguide minimum width = ' + str(minwidth) + '; minimum zonal wind in waveguide = ' + str(Uthresh) +
                              '; waveguide minimum depth = ' + str(mindepth))

    dsmap = xr_wg_map.to_dataset(name = 'WG_map')
    dsmap.attrs['history'] = ('Created on ' + str(date.today()) + ' by calc_waveguide_map in rhwhitepackages3')
    dsmap.attrs['input data'] = ('Input files = ' + Uname)
    dsmap.attrs['parameters'] = ('waveguide minimum width = ' + str(minwidth) + '; minimum zonal wind in waveguide = ' + str(Uthresh) +
                              '; waveguide minimum depth = ' + str(mindepth))

    return(ds,dsmap)



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
