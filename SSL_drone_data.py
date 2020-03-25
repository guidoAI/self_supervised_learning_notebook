# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:07:21 2020

Load data and process it for an SSL experiment.

@author: guido
"""

import pickle
from matplotlib import pyplot as plt
import numpy as np

def pressure_to_height(pressure, base_height = 0):
    """ Transform pressure values from a Paparazzi log to height in meters.
        https://en.wikipedia.org/wiki/Barometric_formula
        h = R T ln (P / Pb) / - g M
    """
    
    PPRZ_ISA_GAS_CONSTANT = 8.31447 # R
    PPRZ_ISA_MOLAR_MASS = 0.0289644 # M
    PPRZ_ISA_GRAVITY = 9.80665 # g
    PPRZ_ISA_SEA_LEVEL_TEMP = 288.15 # T
    PPRZ_ISA_SEA_LEVEL_PRESSURE = 101325.0 # P
    PPRZ_ISA_AIR_GAS_CONSTANT = (PPRZ_ISA_GAS_CONSTANT/PPRZ_ISA_MOLAR_MASS)
    PPRZ_ISA_M_OF_P_CONST = (PPRZ_ISA_AIR_GAS_CONSTANT *PPRZ_ISA_SEA_LEVEL_TEMP / PPRZ_ISA_GRAVITY)
    height = PPRZ_ISA_M_OF_P_CONST * np.log(PPRZ_ISA_SEA_LEVEL_PRESSURE / pressure)
    if(base_height != 0):
        height += -np.mean(height) + base_height
    return height


pickle_in = open("drone_data.pickle","rb")
drone_data = pickle.load(pickle_in)

plt.figure()
plt.plot(drone_data['optitrack'])
plt.plot(drone_data['sonar'])
plt.plot(pressure_to_height(drone_data['pressure'], np.mean(drone_data['sonar'])))
plt.legend(['optitrack', 'sonar', 'pressure-based height'])


