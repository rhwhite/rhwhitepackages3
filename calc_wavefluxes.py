# EP fluxes, Plumb fluxes and Tak Nak fluxes

# From equation 5.7 from 
# https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281985%29042%3C0217%3AOTTDPO%3E2.0.CO%3B2


def calc_Plumb(SF,plev,latname='lat',lonname='lon'):
    # From equation 5.7 from 
    # https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281985%29042%3C0217%3AOTTDPO%3E2.0.CO%3B2

    # define constants
    a = 6.37122e06  # radius of Earth

    # get latitudes and longitudes
    lat = SF.coords[latname]
    lon = SF.coords[lonname]

    # get dimensions and order
    SFdims = SF.dims
    for idx in 0,len(SFdims)-1:
        if SFdims[idx] == latname: idxlat = idx
        elif SFdims[idx] == lonname: idxlon = idx

    dlat = np.gradient(np.deg2rad(lat))
    dlon = np.gradient(np.deg2rad(lon))

    phi = np.deg2rad(lat)     # Get latitude in radians
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    acphi = a * cphi

    # Check that pressure is in mb
    if plev > 1000.0:
        print("plev not in mb?")
        sys.exit()


    # Calculate differentations
    PSIanom = SF - SF.mean(dim=lonname)

    dPSIanomdLON = np.gradient(PSIanom,axis=idxlon)/dlon
    ddPSIanomddLONLON = np.gradient(dPSIanomdLON,axis=idxlon)/dlon

    dPSIanomdLAT = np.gradient(PSIanom,axis=idxlat)/dlat[:,None]
    ddPSIanomddLONLAT = np.gradient(dPSIanomdLON,axis=idxlat)/dlat[:,None]

    # plev normalized to 1000mb as in Plumb (1985)
    const = (plev/1000.0) * 1.0/(2.0 * a*a)

    Fx = const * 1.0/cphi * (dPSIanomdLON * dPSIanomdLON - PSIanom * ddPSIanomddLONLON)
    Fy = const * (dPSIanomdLON * dPSIanomdLAT - PSIanom * ddPSIanomddLONLAT)

    return(Fx,Fy)

