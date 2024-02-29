#!/usr/bin/env python
# coding: utf-8

# ## Fenton Karma Numerical Solution

in_cannon = True
jupyter = False
debug = True
cannon_path = <path-to-cluster>

import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
from scipy import integrate
import pickle
import glob

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
 
import statsmodels.api as sm
import pandas as pd

import matplotlib
from matplotlib import ticker, pyplot, cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from IPython.display import HTML

if in_cannon: PATH_TO_ROOT = cannon_path 
    
PATH_TO_DATA = PATH_TO_ROOT + 'data/'
PATH_TO_RESULTS = PATH_TO_ROOT + 'tests/results/'
PATH_TO_SOURCE = PATH_TO_ROOT + 'src'
PATH_TO_FIGURES = PATH_TO_ROOT + 'tests/plots/'

print(PATH_TO_ROOT)

# custom routines
sys.path.append(PATH_TO_SOURCE)
import FK
import importlib
importlib.reload(FK);
from FK import last_run
from FK import find_pseudo_ECG_peaks
from FK import rebuild_results
from FK import append_pseudo_params
from FK import append_AP_params

from FK import advance, pseudo_ECG

#plt.style.use('seaborn-whitegrid')
## LATER - PUT all rcParams in separate file
import config
#plt.rcParams.update(config.rcparams)
sns.set_style("white")

import warnings
warnings.simplefilter("ignore")

from FK import J_fi, J_so, J_si, J_stim 
from FK import tau_v_minus, I_mA, J_ms_1, V_mV
from FK import calc_min_Nsteps, f_pulse, p

PRECISION = 3
test_dpi = 70
display_dpi = 100
paper_dpi = 300

last_run()


####### 
## If in Cannon, parse command line arguments
######

if (in_cannon & (not jupyter)):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-JOB_ID', '--JOB_ID', default=0.005, help="Job ID assigned")
    parser.add_argument('-D', '--D', default=0.005, type=float, help="Diffusion constant")
    parser.add_argument('-Jamp', '--Jamp', default=0.9, type=float, help="Amplitude of J stimulus")
    parser.add_argument('-tp', '--tp', default=5., type=float, help="Duration of J stimulus")
    parser.add_argument('-Lexc', '--num_cells', default=10, type=int, help="Space length of J stimulus")
    parser.add_argument('-V0', '--V0', default=0, type=int, help="Resting membrane potential")
    parser.add_argument('-Da', '--Da', default=0.25, type=float, help="Rel. amplitude of D in scar")
    parser.add_argument('-Lscar', '--Lscar', default=0.25, type=float, help="x length of scar tissue")
    parser.add_argument('-xscar', '--xscar', default=0.25, type=float, help="x start of scar tissue")
    parser.add_argument('-cake', '--cake', default='NONE', help="file distinguisher")

    args = vars(parser.parse_args())

    # Set up parameters
    JOB_ID = args['JOB_ID']
    D = args['D']
    Jamp = args['Jamp']
    tp = args['tp']
    Lexc_num_cells = args['num_cells']
    V_0 = args['V0']
    Da = args['Da']       # relative increase in D in scar tissue frm D
    Lscar = args['Lscar'] # x length of scar tissue
    xscar = args['xscar']
    cake = args['cake']

print(JOB_ID)

if debug: print(D, Jamp, tp, Lexc_num_cells)

## Create RUN ID
RUN_ID = FK.create_ID(JOB_ID, cake)
print(RUN_ID)

# Read in model parameters
params_df = pd.read_csv(PATH_TO_ROOT + 'parameters.csv', 
                         delimiter=',', index_col='model',
                         skip_blank_lines=True, comment='#') 
params_df

# 1 - Choose parameter model
FK_model = 'MBR'
params = pd.Series(params_df.loc[FK_model]) 
print(params)

# 2 - Add additional parameters
C_m = 1.;         # Capacitance (uF/cm^2), typical 1.-12.
V_fi = 15.        # Nernst potential (mV)

# Fast_inward_current (ms) = 0.25
tau_d = C_m/params['g_hat_fi']; 
print(f'tau_d={tau_d}')

# 3 - Time limits
Tp = 600.      # ms period between pulses - not utilised yet
T0 = 100.      # ms when the pulse hits
T = 400.       # ms total time 

# Add them in params
ser = pd.Series(data={'C_m': C_m, 'V_0': V_0, 'V_fi': V_fi, 
                      'tau_d':tau_d, 'Tp': Tp, 'T0':T0, 
                      'T': T, 'Jamp': Jamp, 'tp': tp, 'D': D, 
                      'ID': RUN_ID, 'Da': Da, 'Lscar': Lscar, 
                      'xscar': xscar, 'model': FK_model}) 

params = pd.concat([params, ser]) 

params['Dscar'] = params['D'] * params['Da']

if debug: print(params)

##### 
## FK model - Numerical solution using finite differences
#####

# test plot of p
if debug:
    fig = plt.subplots(1, 1, figsize=(5,3), dpi=test_dpi)
    tt = np.linspace(0, 2., 100)
    plt.plot(tt, p(tt, 0.14))
    plt.show()

####
## x
####

xmin = 0. 
xmax = 3. 
Nx = 400
x = np.linspace(xmin,xmax,Nx)
dx = x[1]-x[0]
print(f'check dx={dx}')  # should be dx = 0.0075 cm

# stimulus hits in first Lexc_num_cells (~15)
Lexc = Lexc_num_cells * dx  #0.01 is ~one cell #0.14
print(f'Lexc={Lexc}, dx={dx}')

## force certain dt
#dt = 0.004 # ms
#Nsteps = int((T-T0)/dt)
#print(f'{dt}, {Nsteps:_}') #should give 230_769

#### TESTING WITH SMALL TIMESTEP 
## force certain dt 
# dt = 0.05653230821414438 
# Nsteps = int((T-T0)/dt)
# print(f'{dt}, {Nsteps:_}')

## TEST: force certain dt
# dt = 1e-3
# Nsteps = int((T-T0)/dt)
# print(f'{dt}, {Nsteps:_}')

## TEST: force certain dt
#dt = 0.0013
#Nsteps = int((T-T0)/dt)
#print(f'{dt}, {Nsteps:_}') #should give 230_769

# TEST: force certain dt
#dt = 0.002
#Nsteps = int((T-T0)/dt)
#print(f'dt={dt}, Nsteps={Nsteps:_}') # should give 150_000

#### FORCE dt for Dtilde vs D exponential plot
dt = 0.0013  #ms, Nsteps = 2  
Nsteps = int((T-T0)/dt)
print(f'dt={dt}, Nsteps={Nsteps:_}') # should give 230769

# make sure dt is enough
dt_lim, Nsteps_lim = calc_min_Nsteps(D, T, T0, dx)
print(f'Min dt={dt_lim} - Min Nsteps required {Nsteps_lim}')
assert dt <= dt_lim, 'dt should be less that min dt'
 
# Add them in params
ser = pd.Series(data={'xmin': xmin, 'xmax': xmax, 'Nx': Nx, 
                      'dx': dx, 'dt': dt, 'Nsteps': Nsteps,
                     'Lexc':Lexc})

params = pd.concat([params, ser]) 

# Final params to dict
params = params.to_dict()
print(params)

# test plot of pulse
if debug:
    # Check t component 
    fig = plt.subplots(1, 1, figsize=(5,3), dpi=test_dpi)
    tt = np.linspace(0, 100., 500)
    a_t = 200
    b2_t = 10
    b1_t = b2_t + params['tp']

    plt.plot(tt, f_pulse(tt, a_t, b1_t, b2_t))
    plt.show()
    
    # Check x component 
    fig = plt.subplots(1, 1, figsize=(5,3), dpi=test_dpi)
    xx = np.linspace(0, 3.5, 100)
    a_x = 500
    b2_x = 0.02
    b1_x = b2_x + params['Lexc']
    plt.plot(xx, f_pulse(xx, a_x, b1_x, b2_x))
    plt.show()

print(f'xmin={xmin}, xmax={xmax}, Nx={Nx}, T0={T0}, T={T}, Nsteps={Nsteps:_}')

print(params)

#############
# main single
#############

t = np.linspace(T0, T, Nsteps)
x = np.linspace(xmin, xmax, Nx)  
print(xmin, xmax, Nx, T0, T, Nsteps)

u_2D, v_2D, w_2D = advance(x, t, params)
print(u_2D.shape)
print(last_run())

# ################

# Save u, w, v arrays
name = 'u'
path = f'{PATH_TO_DATA}{RUN_ID}_{name}.npy' 
print(f'Saving {name} to {path}')
with open(path, 'wb') as f:
    np.save(f, u_2D)
f.close()

name = 'v'
path = f'{PATH_TO_DATA}{RUN_ID}_{name}.npy' 
print(f'Saving {name} to {path}')
with open(path, 'wb') as f:
    np.save(f, v_2D)
f.close()
    
name = 'w'
path = f'{PATH_TO_DATA}{RUN_ID}_{name}.npy' 
print(f'Saving {name} to {path}')
with open(path, 'wb') as f:
    np.save(f, w_2D)
f.close()

# ###########################################
# #### Construct, plot, and save pseudo-EKG
# ###########################################

Phi, xstar = pseudo_ECG(u_2D, params)

params['xstar'] = np.round(xstar, 3)

name = 'Phi'
path = f'{PATH_TO_DATA}{RUN_ID}_{name}.npy' 
print(f'Saving {name} to {path}')
with open(path, 'wb') as f:
    np.save(f, Phi)
f.close()

# Find peaks and update params with pseudo attributes
params = append_pseudo_params(Phi, params)
print(params)

# Add APD to params
params = append_AP_params(u_2D, params)
print(params)

result = pd.DataFrame([params])
pathtocsv = PATH_TO_DATA + RUN_ID + '_params.csv'
result.to_csv(pathtocsv, index=False, header=True)
print(f'Saving params to {pathtocsv}')
print('SUCCESS!!!')
print(last_run())

############ FIN 










