# -*- coding: utf-8 -*-
"""
Created on 13 Aug 2019

@author: rachel, rachel.white@cantab.net

Functions to calculation the barotropic refractive index
There are slightly different ways of calculating this in the literature,
potentially due to different ways of approximating to a Mercator projection.
Each method gives similar results

"""

import os, errno
import netCDF4
import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr
#import Ngl
import math

from copy import copy

from scipy import stats
from scipy.interpolate import interp1d
import rhwhitepackages3
from rhwhitepackages3.readwrite import *
from rhwhitepackages3.stats import regressmaps
from rhwhitepackages3.griddata_functions import *
from rhwhitepackages3.CESMconst import *

def calc_Ks(Uin):
    ## Calculate BetaM
    ## Hoskins and Karoly (see also Vallis (page 551) and Petoukhov et al 2013
    ## and Hoskins and Ambrizzi (1993))

    OMEGA = 7.2921E-5
    a = 6.3781E6
    try:
        lats_r = np.deg2rad(Uin.latitude)
    except AttributeError:
        lats_r = np.deg2rad(Uin.lat)

    coslat = np.cos(lats_r)

    betaM1 = 2.0 * OMEGA * coslat * coslat / a

    Um = Uin / coslat

    cos2Um = Um * coslat * coslat

    # first differentiation
    ddy_1 = ddy_merc(cos2Um)
    # divide by cos2phi
    ddy_1_over_cos2p = ddy_1 * (1.0/(coslat * coslat))

    # second differentiation

    ddy_2 = ddy_merc(ddy_1_over_cos2p)

    betaM = betaM1 - ddy_2
    # Now calculate Ks from BetaM
    # Hoskins and Ambrizzi and Hoskins and Karoly now agree
    Ks2 = a * a * betaM/Um

    Ks = np.sqrt(Ks2)

    return(Ks,Ks2) #,betaM)


