# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:50:01 2020

Load the data matrix as used in the MATLAB scripts for the article and store it as a pickle.
    
@author: guido
"""

import scipy.io
import pickle

data = scipy.io.loadmat('data.mat')
sonar = data['data'][0][1]
pressure = data['data'][0][2]
optitrack = data['data'][0][3]

drone_data = {'sonar' : sonar, 'pressure' : pressure, 'optitrack' : optitrack}

with open('drone_data.pickle', 'wb') as f:
    pickle.dump(drone_data, f)