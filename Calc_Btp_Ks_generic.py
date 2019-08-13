# -*- coding: utf-8 -*-
"""
Created on Aug 13 2019

@author: rachel, rachel.white@cantab.net

"""

import os, errno
from netCDF4 import Dataset
import netCDF4
import numpy as np
import datetime as dt
import Ngl


a = 6.37122e06

PI = 3.14159
P0 = 100000.0	# "Pa"
secs = 86400.0

def Calc_Btp_Ks(U,lons,lats):

    # set up some3 trig constants
    phi = np.deg2rad(lats)
    cphi = np.cos(phi)
    c2phi = cphi * cphi
    acphi = a * cphi
    asphi = a * np.sin(phi)
    f = 2.0 * omega * np.sin(phi)
    a2 = a*a
    f2 = f*f
#######################

    for ilat in range(0,len(lats)):
	    for ilon in range(0,len(lons)):
		    dthetadz(...,ilat,ilon) = np.array(np.gradient(TH(...,ilat,ilon),z(...,ilat,ilon)


