# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:07:21 2020

Load data and process it for an SSL experiment.

@author: guido
"""

import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

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

def get_drone_data():
    pickle_in = open("drone_data.pickle","rb")
    drone_data = pickle.load(pickle_in)

    return drone_data

def plot_drone_data(drone_data):
    plt.figure()
    plt.plot(drone_data['sonar'])
    plt.plot(pressure_to_height(drone_data['pressure'], np.mean(drone_data['sonar'])))
    plt.plot(drone_data['optitrack'], 'k--')
    plt.ylabel('Height [m]')
    plt.xlabel('Time [-]')
    plt.legend(['sonar', 'pressure-based height', 'optitrack'])

def plot_pressure(drone_data):
    plt.figure()
    plt.plot(drone_data['pressure'])
    plt.xlabel('Time [-]')
    plt.ylabel('Pressure [Pa]')

def train_regression_KNN(training_data, target_values, k = 5, wghts = 'uniform'):
    """
    - training_data: [n_samples, n_features]
    - target_values: [n_samples, n_outputs]
    - k: number of neighbors
    - wghts:    'uniform' = all neighbors contribute equally, 
                'distance' = weights points by the inverse of their distance
    """
    kNN = KNeighborsRegressor(n_neighbors = k, weights = wghts)
    kNN.fit(training_data, target_values)
    return kNN
    
def predict_regression_KNN(kNN, test_data):
    
    outputs = kNN.predict(test_data)
    
    return outputs

def map_pressure_to_sonar(drone_data, training_ratio=0.8, method = 'knn'):

    train_ind = int(training_ratio * len(drone_data['pressure']))
    training_data = drone_data['pressure'][:train_ind]
    training_targets = drone_data['sonar'][:train_ind]
    test_data = drone_data['pressure'][train_ind:]
    test_targets = drone_data['sonar'][train_ind:]
    
    if(method == 'knn'):
        kNN = train_regression_KNN(training_data, training_targets)
        training_outputs = predict_regression_KNN(kNN, training_data)
        test_outputs = predict_regression_KNN(kNN, test_data)
    else:
        lr = LinearRegression().fit(training_data, training_targets)
        training_outputs = lr.predict(training_data)
        test_outputs = lr.predict(test_data)
    
    plt.figure()
    plt.plot(range(len(test_data)), test_outputs)
    plt.plot(range(len(test_data)), test_targets)
    plt.legend(['Pressure-based estimates','Sonar measurements'])
    
    plt.figure()
    plt.hist(test_outputs-test_targets)
    plt.title('Histogram of $h_{pressure}$ - $h_{sonar}$')
    
    print('Mean absolute error = {:.4f}, mean error = {:.4f}'.format(np.mean(abs(test_outputs-test_targets)), np.mean(test_outputs-test_targets)))
    
    # Fusion:
    test_ground_truth = drone_data['optitrack'][train_ind:]
    training_ground_truth = drone_data['optitrack'][:train_ind]
    
    # SSL:
    var_pressure = np.var(training_outputs-training_targets)
    # A priori knowledge:
    # var_pressure = np.var(training_outputs - training_ground_truth)
    var_sonar = np.var(training_data - training_ground_truth)
    test_fused = (var_sonar * test_outputs + var_pressure * test_targets) / (var_sonar + var_pressure)
    print('MAE sonar: {:.4f}, MAE learned pressure = {:.4f}, MAE fused = {:.4f}'.format(
          np.mean(abs(test_targets - test_ground_truth)), np.mean(abs(test_outputs - test_ground_truth)),
          np.mean(abs(test_fused - test_ground_truth)) ))
    
    var_t = np.var(training_ground_truth)
    mean_t = np.mean(training_ground_truth)
    print('Mean t = {}'.format(mean_t))
    test_fused = (var_sonar * test_outputs + var_pressure * test_targets) / (var_sonar + var_pressure + var_t)
    print('MAE sonar: {:.4f}, MAE learned pressure = {:.4f}, MAE fused = {:.4f}'.format(
          np.mean(abs(test_targets - test_ground_truth)), np.mean(abs(test_outputs - test_ground_truth)),
          np.mean(abs(test_fused - test_ground_truth)) ))
    
    
if __name__ == "__main__":
    drone_data = get_drone_data()
    map_pressure_to_sonar(drone_data, method = 'knn')
    

