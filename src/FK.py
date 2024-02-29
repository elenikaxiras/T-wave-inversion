import numpy as np
import pandas as pd
from pathlib import Path
from IPython.core.display import HTML
from datetime import date
from datetime import datetime
from tqdm import tqdm
from scipy import integrate
from scipy import signal
from scipy.signal import argrelextrema
import re
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.axis import Axis
import matplotlib.ticker as ticker 
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

paper_dpi = 300
sns.set_style("ticks")

# Get system date and time
def last_run():
    print(f'DONE\nLast Run: {datetime.now():%Y-%m-%d} {datetime.now().time():%H:%M:%S}')

def create_ID(jobID, cake):
    '''Creata a unique ID for each parameter run
    Is a combination of JobID and date
    '''
    DATESTAMP = f'{datetime.now():%m%d}{datetime.now().strftime("%H%M")}'
    return str(cake) + '_' + str(jobID) + '_' + DATESTAMP 

# Helper routines
def p(xx, thresh):
    ''' Implementation of the Heaviside functions
        H1 = lambda u: torch.heaviside(u-2, value))
        We substituted with sigmoid because of the need for 
        a derivative to optimize
        TRY RELU
        e.g. p(0.12, 0.14) should return almost 0
    '''
    a = 300
    return (np.tanh(a*(xx-thresh))+1.0)/2

def f_pulse(x,a,b1,b2):
    ''' A pulse signal
    '''
    f1 = (np.tanh(a*(b1-x))+1.0)
    f2 = (np.tanh(a*(x-b2))+1.0)
    return f1*f2/4.0

def calc_min_Nsteps(D, T, T0, dx):
    ''' Scan through values of D to find out max Nsteps needed
    '''
    # Choice of Nsteps to satisfy STABILITY CONDITION 
    dt_lim = dx**2/(2.0*D)
    Nsteps_lim = int((T-T0)/dt_lim)
    return dt_lim, Nsteps_lim

# Results
def load_results(filepath, cols):
    """
    Load the results Dataframe if it exists
    Else create it
    """  
    path = Path(filepath)
    if path.is_file():
        print(f'Loading saved results file: {filepath}')
        return pd.read_pickle(filepath)
    else:
        print(f'Creating results file: {filepath}')
        return pd.DataFrame(columns = cols) #, index=None)
    
def save_results(filepath, results):
    assert list(results.columns) == COLS_RESULTS, \
    f'Columns do not match results dataframe: {COLS_RESULTS}'
    results.to_pickle(filepath)
    print(f'Saved results in {filepath}')
    return 'DONE'

def create_labels(params, cols):
    label = []
    for col in cols:
        if col in params.keys():
            value = params[col]
            label.append(f'{col}:{params[col]}')
    label = ' '.join(str(e) for e in label)
    return label

def update_results(rs, r):
    '''
    Updates an existing results df by appending a new result
    Usage:
    ------------
    eda.update_results(PATH_TO_DATA + 'results.pkl', result)
    '''
    print('Updating results dataframe')
    rs = rs.append([r]).reset_index(drop=True)
    return rs

def save_image(fig, caption, path, figname, dpi=paper_dpi, tiff=False):
    '''Save this image in pathfig and its caption in pathcapt
    Input
    ------
    fig: the figure to save as .png and .pdf
    caption: its caption in text quirkly formated, e.g.
    caption = ('The diffusion coefficient $\tilde{D}$ versus position $x$ is shown for myocardial tissue with normal' 
           f' D = {D} containing a defect (scar) with the following parameter values:' 
           f'diffusion coefficient = {Dscar}, length = {Lscar}, and start position = {xscar}')
    path: PATH_TO_FIGURES
    figname: a unique name to add to all the files
    Output
    ------
    Just saves the files
    '''
    pathfig = f'{path}{figname}'
    print(f'Saving file to {pathfig}.png')
    fig.savefig(pathfig+'.png', dpi=dpi, format='png', edgecolor=fig.get_edgecolor())
    
    print(f'Saving file to {pathfig}.pdf')
    fig.savefig(pathfig+'.pdf', dpi=dpi, format='pdf', bbox_inches='tight', edgecolor=fig.get_edgecolor())
    
    # save high fidelity image
    if tiff:
        print(f'Saving file to {pathfig}.tiff')
        fig.savefig(pathfig+'.tiff', dpi=dpi, format='tiff', bbox_inches='tight', edgecolor=fig.get_edgecolor())

    print(f'Saving caption to {pathfig}.tex')
    latex = f'%!TEX root = ../nature.tex\n\\newcommand\\{figname}{ {caption} }\n'
    latex = latex.replace("'", "")
    with open(pathfig+'.tex', 'w') as f: f.write(latex)
    f.close()
        
def clean(text, cake):
    '''Clean a long filename
    '''
    pattern = f'({cake}_\w+)_params\.csv' #pattern = r'GEO_(\w+)_params\.csv'
    match = re.search(pattern, text)
    if match:
        extracted_string = match.group(1)
    else:
        extracted_string = 'No match found'
    return extracted_string

def pretty_print(df, heading='', head=False):
    title = "<style>h3{text-align: left;}</style><h3>" + f'{heading}' + "</h1>" 
    if head:
        return display(HTML(title), HTML(df.head().to_html().replace("\\n","<br>")))
    else: return display(HTML(title), HTML(df.to_html().replace("\\n","<br>")))

# FK currents
# network approximations of the ionic currents
def J_fi(u, v, puc, params):        
    tau_d = params['tau_d']
    u_c = params['u_c']
    return -(v/tau_d) * puc * (1-u) * (u - u_c)

def J_so(u, pu, puc, params):
    tau_0 = params['tau_0']
    u_c = params['u_c']
    tau_r = params['tau_r']
    return (u/tau_0)*(pu) + ((1/tau_r) * puc)

# k = 10. denotes the steepness, should be smooth, a.k.a. small
def J_si(u, w, params):
    tau_si = params['tau_si']
    u_csi = params['u_csi']
    return -(w/(2 * tau_si)) * (1 + np.tanh(10. * (u-u_csi)))

def J_stim(x_i, t_n, params):
     
    a_x = 300. # somewhat arbitrarily chosen
    b2_x = 0.  # instead of 0.02 (~ two cells length)
    b1_x = b2_x + params['Lexc']
    
    a_t = 300.
    b2_t = params['T0']
    b1_t = b2_t + params['tp'] 
    
    stim = f_pulse(x_i, a_x, b1_x, b2_x)\
                   * f_pulse(t_n, a_t, b1_t, b2_t) * params['Jamp'] 
    return stim

def J_stim_t(t_n, params):
    '''Jstim t component'''
    offset = 1
    a_t = t_n.max()
    #print(a_t)
    b2_t = params['T0'] + offset
    #print(b2_t)
    b1_t = b2_t + params['tp'] 
    #print(b1_t)
    
    stim = f_pulse(t_n, a_t, b1_t, b2_t) * params['Jamp'] 
    return stim

def J_stim_x(x_n, params):
    '''Jstim x component'''
    offset = 1
    a_x = 500
    b2_x = 0.02
    b1_x = b2_x + params['Lexc']
    
    stim = f_pulse(x_n, a_x, b1_x, b2_x) * params['Jamp'] 
    return stim

def tau_v_minus(u, params):
    ''' Gate variable using a step function, 
        basically assumes two values: 
        tau_v_minus=tau_v1 if u <= u_v
        tau_v_minus=tau_v2 if u > u_v
    '''
    return (1-p(u, params['u_v'])) * params['tau_v1_minus'] \
           + p(u, params['u_v']) * params['tau_v2_minus']

# variable D
def D_tilde(x, params):
    ''' Space dependent diffusion coeff.
    '''
    
    D = params['D']             # D effective value
    Da = params['Da']           # scar D value as % of D effective
    Dscar = D * Da              # scar D absolute value (negative or positive)
    a = 200                     # steepness
    b1 = params['xscar']        # start of scar tissue
    b2 = b1 + params['Lscar']   # length of scar tissue
    f1 = (np.tanh(a*(x-b1))+1.0)
    f2 = (np.tanh(a*(b2-x))+1.0)
    
    return D + Dscar * f1*f2/4.0

# Converting to and from normalized values
def I_mA(J, params, Va):
    ''' Calculate un-normalized ionic current in mA
        J is in ms-1
    '''
    Cm = params['C_m']
    V0 = params['V_0']
    Va = params['V_fi']
    return J * Cm * (Va - V0)

def J_ms_1(I, params):
    ''' Reverse calculate normalized current in ms-1
        I is in mA
    '''
    Cm = params['C_m']
    V0 = params['V_0']
    Va = params['V_fi']
    return I / (Cm * (Va - V0))

def V_mV(uu, params):
    ''' Calculate un-normalized potential in mV
    '''
    V0 = params['V_0']
    Vfi = params['V_fi']
    return V0 + (uu * (Vfi - V0))

def V_mV_1(V, params):
    ''' Calculate normalized potential
    '''
    V0 = params['V_0']
    Vfi = params['V_fi']
    return (V-V0)/(Vfi - V0)

def rebuild_results(ID, path, load_all=False):
    ''' Returns an array of variables for a given run.
    It also reconstructs the t and x variables
    Returns
    -------
    A dictionary with keys: u, Phi, params
    '''
    
    def show_result(result, load_all):
        if load_all: return 'u, v, w, Phi, params'
        else: return 'u, Phi, params'
        
    result = {}
    path = path+ID
    
    # load u, v, w
    to_load = ['u', 'Phi', 'v', 'w'] if load_all else ['u', 'Phi']
    
    for name in to_load:
        filename = f'{path}_{name}.npy'
        with open(filename, 'rb') as f:
            vars()[name] = np.load(f) 
        #print(f'Loading: {filename}')
        result[name] = vars()[name]
    
    # load parameters
    name = 'params'
    filename = f'{path}_{name}.csv'  
    prms = pd.read_csv(filename)
    prms = prms.iloc[0].to_dict()
    #print(f'Loading: {filename}')
    result['params'] = prms
    
    Nsteps = prms['Nsteps']
    #print(f'Successfully rebuilt {ID} (Nsteps={Nsteps}): {show_result(result, load_all)}')
    
    return result

def plot_currents(u_2D, v_2D, w_2D, params, rrange=[5], dpi=paper_dpi):
    fig,ax = plt.subplots(3, 2, figsize=(10,10), dpi=dpi)
    i = 0
    ax[0,0].axhline(y=params['u_c'], linestyle='dotted', 
                    linewidth=1., color='orange', label=r'$u_c$ threshold') 
    for ii in range(3):
        for jj in range(2):
            ax[ii,jj].tick_params(axis='both', which='major', labelsize=10)
            ax[ii,jj].tick_params(axis='both', which='minor', labelsize=8)
            ax[ii,jj].set_xlabel(r'Time$\;$(ms)');
            #ax[ii,jj].legend(fontsize=10)

    Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
    t = np.linspace(T0, T, Nsteps)  
    xmin, xmax, Nx = params['xmin'], params['xmax'], int(params['Nx'])
    x = np.linspace(xmin, xmax, Nx)

    for ix in rrange: #[5]: #np.arange(50, Nx, 100):
        puc = p(u_2D[ix,:], params['u_c'])
        pu = (1 - puc)
        uu = u_2D[ix,:]
        V = V_mV(uu, params)

        Jfi = J_fi(u_2D[ix,:], v_2D[ix,:], puc, params)
        Jso = J_so(u_2D[ix,:], pu, puc, params)
        Jsi = J_si(u_2D[ix,:], w_2D[ix,:], params)

        ax[0,0].plot(t, uu-i*0.01, linewidth=1) #, label='x=%5.3f'%x[ix]);
        ax[0,0].set_ylabel(r'$u$ (normalized)')
        ax[0,1].plot(t, v_2D[ix,:], linewidth=1,);
        ax[0,1].set_ylabel(r'$\nu$')
        ax[1,0].plot(t, w_2D[ix,:]);#, label='w');
        ax[1,0].set_ylim(0.5,1.1)
        ax[1,0].set_ylabel(r'$w$')
        ax[1,1].plot(t, Jfi)
        ax[1,1].set_ylabel(r'$J_fi$')
        ax[2,0].plot(t, Jso)
        ax[2,0].set_ylabel(r'$J_so$')
        ax[2,1].plot(t, Jsi)
        ax[2,1].set_ylabel(r'$J_si$')
        i +=1
    fig.tight_layout()
    return fig

def pseudo_ECG(u, params,  
               fudge=50., kappa = 0.0189, #0.02
               scale=0.25):
    ''' Calculate the pseudo-ECG (Gima and Rudy, 2002), 
    the extracellular potential at a distance x_star
    '''
    #t = np.linspace(T0, T, Nsteps)
    xmin, xmax, Nx, Nsteps = params['xmin'], params['xmax'], int(params['Nx']), int(params['Nsteps'])
    x = np.linspace(xmin, xmax, Nx)
    
    Phi = np.zeros(Nsteps)
    dx = x[1] - x[0]          # space discretization
    L = x[-1]                 # integration limits 0, L
    #xstar = L + 2.           # lazar xstar 
    xstar = L + fudge*dx      # point of measurement of 
                              # extracellular potential
    twodx = 2*dx
    print(xstar.shape)
    for n in range(Nsteps):
        V = V_mV(u[:, n], params)
        V_prime = np.zeros(Nx)
        V_prime[0] = (V[1] - V[0])/dx
        V_prime[-1] = (V[-1] - V[-2])/dx

        for ix in range(1, Nx-1):
            V_prime[ix] = (V[ix+1] - V[ix-1])/(twodx)

        V_int = V_prime/(xstar - x)**2

        Phi[n] = integrate.simpson(V_int, x)

    Phi = -kappa * Phi
    # optional to match the real EKG
    Phi = Phi * scale 
    
    print(Phi.shape)
    
    return Phi, xstar

def find_pseudo_ECG_peaks(pseudo, prominence=0.001, distance=4000, verbose=False):
    """ Find the 2 peaks of the signal using scipy
    Arguments
    -------------
    - signal : the 1D signal array
    - prominence: min amplitude value to be considered a peak. Default = 0.007 will
    set very low T (flat T) to zero.
    - distance: min horizontal distance between peaks. Default = 50000 gets rid of 
    little peaks close to R
             
    Returns
    ------------
    peaks :  an array of the peaks
    prominences : an array of the prominences for each peak
    
    NOTE: distance should be percentage of all T points not hardcoded ;)
    """
    print('inside FK')
    pseudo_R = 0.
    pseudo_T = 0.
    
    # first find the peaks and their prominence
    peaks, _ = signal.find_peaks(pseudo, prominence=prominence, distance=distance, height=0.0001)   
    if verbose: print(f'inside find: peaks={peaks}')
    promin = signal.peak_prominences(pseudo, peaks)[0]
    
    # remove start peaks
    thresh = 5000
    peaks = [num for num in peaks if num >= thresh]
    
    if len(peaks) > 1:
        pseudo_R = pseudo[peaks[1]]
        if peaks[0] > peaks[1]:
            pseudo_T = pseudo[peaks[0]]
    elif len(peaks) > 0:
        pseudo_R = pseudo[peaks[0]]

    # keep only the two most prominent ones
    #peaks = dict(zip(promin, peaks))
    #first2peaks = {k: peaks[k] for k in sorted(peaks.keys())[2:]}
    #promin = list(first2peaks.keys())[:2]
    #peaks = list(first2peaks.values())[:2]
   
    return peaks

#### DELETE
# def append_pseudo_params(Phi, params):
#     """ Calculate several attributes of the pseudo ECG and 
#     add them to the 'params' dictionary. Attributes are:
#     Arguments
#     -------------
#     Phi: the 1D signal array
#     params: old parameters dictionary
             
#     Returns
#     ------------
#     params: new parameters dictionary
#     """
    
#     Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
#     t = np.linspace(T0, T, Nsteps)
        
#     # Find peaks, all time points are in sample space
#     peaks = find_pseudo_ECG_peaks(Phi)    
#     widths, width_heights, left_ips, right_ips = signal.peak_widths(Phi, peaks, rel_height=0.95)
    
#     try: pseudo_R = Phi[peaks[0]]
#     except: pseudo_R = 0.
#     params['pseudo_R'] = pseudo_R
    
#     try: pseudo_T = Phi[peaks[1]]
#     except: pseudo_T = 0.
#     params['pseudo_T'] = pseudo_T
    
#     try: QRSWAVE_high = int(right_ips[0])
#     except: QRSWAVE_high = 0.
#     params['QRSWAVE_high'] = QRSWAVE_high
    
#     try: TWAVE_low = int(left_ips[1])
#     except: TWAVE_low = 0.
#     params['TWAVE_low'] = TWAVE_low
    
#     try: TWAVE_high = int(right_ips[1])
#     except: TWAVE_high = 0.
#     params['TWAVE_high'] = TWAVE_high
    
#     try: pseudo_T_dur = t[TWAVE_high] - t[TWAVE_low] 
#     except: pseudo_T_dur = 0
#     params['pseudo_T_dur'] = pseudo_T_dur
    
#     try: pseudo_ST = t[TWAVE_low] - t[QRSWAVE_high] 
#     except: pseudo_ST = 0
#     params['pseudo_ST_dur'] = pseudo_ST
    
#     try: 
#         mode = stats.mode(Phi[QRSWAVE_high:TWAVE_low], axis=None)
#         #print(mode[0], type(mode[0]))
#         mode = float(mode[0])
#         pseudo_ST_elev = mode
#     except: pseudo_ST_elev = 0
#     params['pseudo_ST_elev'] = pseudo_ST_elev

#     try: RPEAK = peaks[0]
#     except: RPEAK = 0
#     params['RPEAK'] = RPEAK
    
#     try: TPEAK = peaks[1]
#     except: TPEAK = 0
#     params['TPEAK'] = TPEAK

#     return params

def calc_APD(X, params, percent=90, plot=False):
    '''Calculate action potential duration
    '''
    Nsteps, T0, T = int(params['Nsteps']), int(params['T0']), params['T']
    t = np.linspace(T0, T, Nsteps) 
    
    # find peak
    if len(X.shape) > 1: X = X[5,:]
    AP = np.argmax(X)
    #print(AP, t[AP])
    # target is percent % of peak
    target_value = X[AP] - (X[AP] * percent/100)
    #print(target_value)
    tol = 0.1  
    # Use np.isclose to find values within the specified tolerance
    indices = np.where(np.isclose(X[AP:], target_value, atol=tol))
    # If values are found, it will return an array of indices
    if len(indices[0]) > 0:
        AP90 = indices[0][0]# correct for start of peak
        APD = t[AP+AP90]-t[T0]
        #print(f'AP90={AP90:.4f}')
        #print(f'APD={APD}')
    else: 
        AP90=0
        APD=0
        
    indices = np.where(np.isclose(X[:AP], target_value, atol=tol))
    if len(indices[0]) > 0:
        start_AP90 = indices[0][0]
        #print(f'startAP90={start_AP90:.4f}')
    else: AP90=0
        
    if plot:
        fig, ax = plt.subplots(dpi=300) #figsize=(10,3), 
        connectionstyle="arc3,rad=0."
        ax.plot(t, X)
        ax.plot(t[AP], X[AP], "*", color='red')
        ax.plot(t[AP:][AP90], X[AP:][AP90], "*", color='blue')
        # make fancy connector
        yoffset = 0.03
        x1, y1 = t[start_AP90], X[AP+AP90]
        x2, y2 = t[AP+AP90], X[AP+AP90]
        #ax.plot([x1, x2], [y1, y2], ".", color='green')
        ax.text((x2-x1)/2+T0, y2+yoffset, 'APD', ha='center', fontsize=12)
        ax.annotate("",
                    xy=(x1, y1), xycoords='data',
                    xytext=(x2, y2), textcoords='data',
                    arrowprops=dict(arrowstyle="<->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle=connectionstyle,
                                    ),
               )
        fig.show()
    
    return APD

def append_AP_params(X, params, percent=90, plot=False):
    if len(X.shape) > 1: X = X[10,:]
    try: APD = calc_APD(X, params, percent=percent, plot=plot)
    except: APD = 0
    params['APD'] = APD
    
    return params

def append_pseudo_params(Phi, params, verbose=False):
    """ Calculate several attributes of the pseudo ECG and 
    add them to the 'params' dictionary. Attributes are:
    Arguments
    -------------
    Phi: the 1D signal array
    params: old parameters dictionary
             
    Returns
    ------------
    params: new parameters dictionary
    """
    
    pseudo_R = 0.
    pseudo_T = 0.
    alt_pseudo_T = 0.
    
    Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
    t = np.linspace(T0, T, Nsteps)
        
    # Find peaks, all time points are in sample space
    peaks = find_pseudo_ECG_peaks(Phi)    
    widths, width_heights, left_ips, right_ips = signal.peak_widths(Phi, peaks, rel_height=0.95)
    if verbose: print(peaks)
    
    try: pseudo_R = Phi[peaks[0]]
    except: pseudo_R = 0
    params['pseudo_R'] = pseudo_R
    
    try: pseudo_T = Phi[peaks[1]]
    except: pseudo_T = 0
    params['pseudo_T'] = pseudo_T
    
    try: QRSWAVE_high = int(right_ips[0])
    except: QRSWAVE_high = 0
    params['QRSWAVE_high'] = QRSWAVE_high
    
    try: TWAVE_low = int(left_ips[1])
    except: TWAVE_low = 0
    params['TWAVE_low'] = TWAVE_low
    
    try: TWAVE_high = int(right_ips[1])
    except: TWAVE_high = 0
    params['TWAVE_high'] = TWAVE_high
    
    try: pseudo_T_dur = t[TWAVE_high] - t[TWAVE_low] 
    except: pseudo_T_dur = 0
    params['pseudo_T_dur'] = pseudo_T_dur
    
    try: pseudo_ST = t[TWAVE_low] - t[QRSWAVE_high] 
    except: pseudo_ST = 0
    params['pseudo_ST_dur'] = pseudo_ST
    
    try: 
        mode = stats.mode(Phi[QRSWAVE_high:TWAVE_low], axis=None)
        mode = float(mode[0])
        pseudo_ST_elev = mode
    except: pseudo_ST_elev = 0
    params['pseudo_ST_elev'] = pseudo_ST_elev

    try: RPEAK = int(peaks[0])
    except: RPEAK = 0
    params['RPEAK'] = RPEAK
    
    try: TPEAK = int(peaks[1])
    except: TPEAK = 0
    params['TPEAK'] = TPEAK
    
    # Search for negative or biphasic T
    params['alt_pseudo_T'] = 0. # save the other peak in biphasic
    if np.any(Phi < 0.): 
        # save existing positive pseudo t
        pos_pseudo_T = pseudo_T
        ID = params['ID']
        if verbose: print(f'ID={ID}: possible negative T')
        peaks = find_pseudo_ECG_peaks(-Phi, prominence=0., distance=50000)
        widths, width_heights, left_ips, right_ips = signal.peak_widths(-Phi, peaks)
        #try:
         #   print(f'Negative peak at: {peaks} or t={t[peaks[0]]} and Phi={Phi[peaks[0]]}') 
        #except: print('negative but zero?')
        
        try: 
            if np.abs(Phi[peaks[0]]) > np.abs(pseudo_T): 
                if verbose: print('negative wins!')
                try: pseudo_T = Phi[peaks[0]]
                except: pseudo_T = 0
                params['pseudo_T'] = pseudo_T  # is negative
                try: TWAVE_low = int(left_ips[0])
                except: TWAVE_low = 0
                params['TWAVE_low'] = TWAVE_low
                try: TWAVE_high = int(right_ips[0])
                except: TWAVE_high = 0
                params['TWAVE_high'] = TWAVE_high
                try: pseudo_T_dur = t[TWAVE_high] - t[TWAVE_low] 
                except: pseudo_T_dur = 0
                params['pseudo_T_dur'] = pseudo_T_dur
                # twinkle, twinkle, little star
                try: TPEAK = int(peaks[0])
                except: TPEAK = 0
                params['TPEAK'] = TPEAK
                
                # keep the pos too
                params['alt_pseudo_T'] = pos_pseudo_T
            else:
                params['alt_pseudo_T'] = Phi[peaks[0]]
                      
        except: print('out')
    return params

def plot_figduo(u_2D, Phi, params, dpi=paper_dpi):
    
    labelsize = 10
    ID, D, Da, Lscar, xscar = params['ID'], params['D'], params['Da'], params['Lscar'], params['xscar']
    pseudo_T = params['pseudo_T']
    pseudo_R = params['pseudo_R']
    
    T0, T, Nsteps = params['T0'], params['T'], int(params['Nsteps'])
    xmin, xmax, Nx = params['xmin'], params['xmax'], int(params['Nx'])
    t = np.linspace(T0, T, Nsteps)
    x = np.linspace(xmin, xmax, Nx)
    
    figduo, ax = plt.subplots(1, 2, figsize=(15,5), dpi=dpi)
    V_thresh = V_mV(params['u_c'], params)
    ax[0].axhline(y=V_thresh, linestyle='dotted', 
                  linewidth=1., color='orange', 
                  label=f'{V_thresh} activation threshold')
    ax[0].legend(fontsize=labelsize)
    
    start = np.where(x==params['xscar'])
    if start[0].size>0: 
        start = start[0][0].astype(int)
    else: start=100
    i=0
    if Da==0: 
        title = f'ID={ID},D={D},Da={Da},T={pseudo_T:.5f},R={pseudo_R:.5f}'
    else:
        title = f'ID={ID},D={D},Da={Da},Lscar={Lscar},xscar={xscar},T={pseudo_T:.5f},R={pseudo_R:.5f}'

    # should choose two x, one outside scar and one inside
    for ix in [100, start, 200, 270]:
        puc = p(u_2D[ix,:], params['u_c'])
        pu = (1 - puc)
        uu = u_2D[ix,:]
        V = V_mV(uu, params)

        tol = 0.001 # tolerance
        #R,C = np.where(np.abs(V[:,None] - V_thresh)<=tol)
        #print(R,C)
        ax[0].plot(t, V, linewidth=1, label='x=%5.3f cm'%x[ix]);
        ax[0].legend()
        i +=1
    figduo.suptitle(title)
    ax[0].set_ylabel('u')
    ax[0].set_xlabel('$t(ms)$');
    ax[1].set_xlabel('$t(ms)$');
    ax[1].set_ylim(-1., 2.5)
    ax[1].plot(t, Phi, label=f'pseudo EKG');
    ax[1].legend()
    figduo.tight_layout()
    return figduo #, caption


# def plotly(x, y, ax=None):
#     if ax is None:
#         ax = plt.gca()
#     line, = ax.plot(x, y)
#     ax.set_ylabel('Yabba dabba do!')
#     return line

def plot_pseudo_ECG(Phi, params, ax=None,
                    xrange=None, yrange=None,
                    figsize=(12,5),  
                    markers=False, marker_R=False, dpi=150, 
                    ):
    
    Nsteps, T0, T, Lscar = int(params['Nsteps']), params['T0'], params['T'], params['Lscar']
    Da, pseudo_T, pseudo_R, alt_T = params['Da'], params['pseudo_T'], params['pseudo_R'], params['alt_pseudo_T']
    ID = params['ID']
    D = params['D']
    ST_elev = params['pseudo_ST_elev']
    title = f'T={pseudo_T:.3f}, alt_T={alt_T:.3f} (mV), D={D:.5f}, ID={ID}'  
    t = np.linspace(T0, T, Nsteps)
    labelsize = 10
    fontsize = 15
    ticksize = 10
    markersize = 13
    palette = sns.color_palette("tab10")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if ax is None:
        ax = plt.gca()
    
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        
    #plt.yticks(fontsize=ticksize)
    #plt.xticks(fontsize=ticklsize)
    ax.set_ylabel('V (mV)', fontsize=fontsize)
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    yoffset = 0.2
    
    #plt.plot(t, Phi)
    #plt.show()
    
    line, = ax.plot(t, Phi, linewidth=2) #, label=f'Lscar={Lscar} cm, Da={Da}, pseudo_T={pseudo_T:.3f}');
    if xrange: ax.set_xlim(xrange)
    if yrange: ax.set_ylim(yrange)
    ax.axhline(y=0., linestyle='--', linewidth=0.5, color='orange') #, label='0 mV')
    if markers:
        ax.axhline(y=params['pseudo_ST_elev'], linestyle='--', linewidth=0.5, color='blue', label='ST elevation')
        ax.axhline(y=Phi[params['TPEAK']], linestyle='--', linewidth=0.5, color='green', label='T peak')
        ax.vlines(x=t[params['TWAVE_low']], linestyle='--', ymin=Phi[params['TWAVE_low']]-yoffset, 
                   ymax=Phi[params['TWAVE_low']]+yoffset, linewidth=0.5, color='green')
        ax.vlines(x=t[params['TWAVE_high']], linestyle='--', ymin=Phi[params['TWAVE_high']]-yoffset, 
                   ymax=Phi[params['TWAVE_high']]+yoffset, linewidth=0.5, color='green')
        ax.vlines(x=t[params['QRSWAVE_high']], linestyle='--', ymin=Phi[params['QRSWAVE_high']]-yoffset, \
                   ymax=Phi[params['QRSWAVE_high']]+yoffset, 
                   linewidth=0.5, color='magenta')       
        ax.vlines(x=t[params['RPEAK']], linestyle='--', ymin=Phi[params['RPEAK']]-yoffset, 
                   ymax=Phi[params['RPEAK']]+yoffset, linewidth=0.5, color='green')
        ax.vlines(x=t[params['TPEAK']], linestyle='--', ymin=Phi[params['TPEAK']]-yoffset, 
                   ymax=Phi[params['TPEAK']]+yoffset, linewidth=0.5, color='green')

    # Plot anyway
    ax.vlines(x=t[params['TPEAK']], linestyle='-', ymin=0., ymax=Phi[params['TPEAK']], 
                 linewidth=2., color='green', label='T wave amplitude')
    #ax.plot(t[params['RPEAK']], Phi[params['RPEAK']], "*", color='magenta', label='R wave top')
    ax.scatter(t[params['RPEAK']], Phi[params['RPEAK']], marker="*", c=palette[0], s=100, label=f'R wave peak')
    ax.scatter(t[params['TPEAK']], Phi[params['TPEAK']], marker="*", c=palette[2], s=100, label=f'T wave peak')
    ax.legend(fontsize=labelsize)
    if marker_R==True:
        ax.vlines(x=t[params['RPEAK']], linestyle='-', ymin=0., ymax=Phi[params['RPEAK']], 
                  linewidth=2., color='magenta', label='R wave amplitude')  
    

    #ax.legend(fontsize=labelsize);
    ax.set_title(title)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_yticks([0, 0.5, 1, 1.5, 2], labels=[0, 0.5, 1, 1.5, 2],
                      fontsize=ticksize)
    ax.set_xticks([100, 200, 300, 400], labels=[100, 200, 300, 400],
                      fontsize=ticksize)
    #ax.set_ylim(-0.2, 2.)
    #ax.set_xlim(80, 400)
    
    return line

def plot_APD(X, params, percent=90, figsize=(10,3), dpi=100):
    '''Calculate action potential duration
    '''
    Nsteps, T0, T = int(params['Nsteps']), int(params['T0']), params['T']
    t = np.linspace(T0, T, Nsteps) 
    
    if len(X.shape) > 1: X = X[5,:]
    AP = np.argmax(X)
    target_value = X[AP] - (X[AP] * percent/100)
    tol = 0.1  
    indices = np.where(np.isclose(X[AP:], target_value, atol=tol))
    if len(indices[0]) > 0:
        AP90 = indices[0][0]# correct for start of peak
        APD = t[AP+AP90]-t[T0]
    else: 
        AP90=0
        APD=0
        
    indices = np.where(np.isclose(X[:AP], target_value, atol=tol))
    if len(indices[0]) > 0:
        start_AP90 = indices[0][0]
    else: AP90=0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
    connectionstyle="arc3,rad=0."
    ax.plot(t, X)
    ax.plot(t[AP], X[AP], "*", color='red')
    ax.plot(t[AP:][AP90], X[AP:][AP90], "*", color='blue')
    # make fancy connector
    yoffset = 0.03
    x1, y1 = t[start_AP90], X[AP+AP90]
    x2, y2 = t[AP+AP90], X[AP+AP90]
    #ax.plot([x1, x2], [y1, y2], ".", color='green')
    ax.text((x2-x1)/2+T0, y2+yoffset, 'APD', ha='center', fontsize=12)
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="<->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
           )

    return fig

def make_label(Da, Lscar, pseudo_T):
    Lscar_text = r'$L_{scar}$'
    #text = r'$\text{scar}$'
    #Lscar_text = r'$L_{{}}$'.format(text)
    if Da!=0: return f'{Lscar_text} = {Lscar:.2f} cm'
    else: return f'Control'
    
def make_width(Da):
    if Da!=0: return 1
    else: return 3
    
def make_color(Da):
    if Da!=0: return None
    else: return 'black'

## Revised as Axes
def ax_multi_plot_pseudo_ECG(Phi_list, params_list, ax=None,
                              xrange=(100, 400), yrange=None, 
                              figsize=(12,5), markers=False, 
                              label=True, fill=True, dpi=paper_dpi):
    
    labelsize = 16
    fontsize = 18
    #fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if ax is None:
        ax = plt.gca()

    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    ax.set_ylabel('V (mV)', fontsize=fontsize)
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    ax.axhline(y=0., linestyle='--', linewidth=0.5, color='orange')
    if xrange: ax.set_xlim(xrange)
    if yrange: ax.set_ylim(yrange) 
    yoffset = 0.05
    xoffset = 10
    i = 0
    for Phi, params in list(zip(Phi_list, params_list)):
        Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
        Lscar, Da, pseudo_T, pseudo_R = params['Lscar'], params['Da'], params['pseudo_T'], params['pseudo_R']
        xscar = params['xscar']
        ST_elev = params['pseudo_ST_elev']
        t = np.linspace(T0, T, Nsteps)
        ID = params['ID']
        Nsteps = params['Nsteps']
        line, = ax.plot(t, Phi, linewidth=make_width(Da), color=make_color(Da), 
                label=make_label(Da, Lscar, pseudo_T));
        i +=1
        if markers:
            ax.axhline(y=params['pseudo_ST_elev'], linestyle='--', linewidth=0.5, color='blue', label='ST elevation')
            plt.vlines(x=t[params['TWAVE_low']], linestyle='--', ymin=Phi[params['TWAVE_low']]-yoffset, \
                       ymax=Phi[params['TWAVE_low']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['QRSWAVE_high']], linestyle='--', ymin=Phi[params['QRSWAVE_high']]-yoffset, \
                       ymax=Phi[params['QRSWAVE_high']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['TWAVE_high']], linestyle='--', ymin=Phi[params['TWAVE_high']]-yoffset, \
                       ymax=Phi[params['TWAVE_high']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['RPEAK']], linestyle='--', ymin=Phi[params['RPEAK']]-yoffset, \
                       ymax=Phi[params['RPEAK']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['TPEAK']], linestyle='--', ymin=Phi[params['TPEAK']]-yoffset, \
                       ymax=Phi[params['TPEAK']]+yoffset, linewidth=0.5, color='green')

            plt.plot(t[params['RPEAK']], Phi[params['RPEAK']], "*", color='red', label='R wave')
            plt.plot(t[params['TPEAK']], Phi[params['TPEAK']], "*", color='blue', label='T wave')
    if label: 
        ax.legend(loc='upper right', fontsize=labelsize);
    if fill:
        ax.fill_between(t, -0.2, 0.2, alpha=0.1)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.AutoLocator())
    return line
   
def multi_plot_pseudo_ECG(Phi_list, params_list, 
                          xrange=(100, 400), yrange=None, 
                          figsize=(12,5), markers=False, 
                          label=True, fill=True, dpi=paper_dpi):
    
    labelsize = 14
    fontsize = 18
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    ax.set_ylabel('V (mV)', fontsize=fontsize)
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    ax.axhline(y=0., linestyle='--', linewidth=0.5, color='orange')
    if xrange: ax.set_xlim(xrange)
    if yrange: ax.set_ylim(yrange) 
    yoffset = 0.05
    xoffset = 10
    i = 0
    for Phi, params in list(zip(Phi_list, params_list)):
        Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
        Lscar, Da, pseudo_T, pseudo_R = params['Lscar'], params['Da'], params['pseudo_T'], params['pseudo_R']
        xscar = params['xscar']
        ST_elev = params['pseudo_ST_elev']
        t = np.linspace(T0, T, Nsteps)
        ID = params['ID']
        Nsteps = params['Nsteps']
        ax.plot(t, Phi, linewidth=make_width(Da), color=make_color(Da), 
                label=make_label(Da, Lscar, pseudo_T));
        i +=1
        if markers:
            ax.axhline(y=params['pseudo_ST_elev'], linestyle='--', linewidth=0.5, color='blue', label='ST elevation')
            plt.vlines(x=t[params['TWAVE_low']], linestyle='--', ymin=Phi[params['TWAVE_low']]-yoffset, \
                       ymax=Phi[params['TWAVE_low']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['QRSWAVE_high']], linestyle='--', ymin=Phi[params['QRSWAVE_high']]-yoffset, \
                       ymax=Phi[params['QRSWAVE_high']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['TWAVE_high']], linestyle='--', ymin=Phi[params['TWAVE_high']]-yoffset, \
                       ymax=Phi[params['TWAVE_high']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['RPEAK']], linestyle='--', ymin=Phi[params['RPEAK']]-yoffset, \
                       ymax=Phi[params['RPEAK']]+yoffset, linewidth=0.5, color='green')
            plt.vlines(x=t[params['TPEAK']], linestyle='--', ymin=Phi[params['TPEAK']]-yoffset, \
                       ymax=Phi[params['TPEAK']]+yoffset, linewidth=0.5, color='green')

            plt.plot(t[params['RPEAK']], Phi[params['RPEAK']], "*", color='red', label='R wave')
            plt.plot(t[params['TPEAK']], Phi[params['TPEAK']], "*", color='blue', label='T wave')
    if label: 
        ax.legend(loc='upper right', fontsize=labelsize);
    if fill:
        ax.fill_between(t, -0.2, 0.2, alpha=0.1)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.AutoLocator())
    return fig

def find_prime(phi, params):
    '''Make an array with all 1st derivatives for array y with step k
    
    Input
    ---------
        phi: array of signal points 
        params: parameters  
    Returns
    '''
    Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
    t = np.linspace(T0, T, Nsteps)
    dt = t[1] - t[0]
    
    phi_prime = np.zeros(Nsteps)
    phi_prime[0] = (phi[1] - phi[0])/dt
    phi_prime[-1] = (phi[-1] - phi[-2])/dt

    for it in range(1, Nsteps-1):
        phi_prime[it] = (phi[it+1] - phi[it-1])/(2*dt)
    
    return phi_prime

def plot_integrand_t(u, phi, params, numsteps=30, dpi=paper_dpi):
    '''Plot AP with integrads and derivatives in time
    Inputs
    ------
    u : action potential in (x,t)
    params : parameters
    '''
    
    Nsteps = int(params['Nsteps'])
    Nx = int(params['Nx'])
    xmax = params['xmax']
    x = np.linspace(params['xmin'], xmax, Nx)
    t = np.linspace(params['T0'], params['T'], Nsteps)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    fig, ax = plt.subplots(2, 2, figsize=(10,8), dpi=100)
    i = 0
    disp = 0.01
    fudge=10.
    x_star = xmax + fudge*dx 
    print(f'x*={x_star:.4f}')

    # printing
    frames = []
    V_primes = []
    tt = []
    
    numsteps = numsteps
    ax[0,0].axhline(y=params['u_c'], linestyle='--', linewidth=0.5, color='orange', label=r'$u_c$')
    for it in np.arange(1000, Nsteps, int(Nsteps/numsteps)):
        puc = p(u[:, it], params['u_c'])
        pu = (1 - puc)
        uu = u[:, it]

        V = u[:, it]
        V_prime = np.zeros(Nx)
        V_prime[0] = (V[1] - V[0])/dx
        V_prime[-1] = (V[-1] - V[-2])/dx

        for ix in range(1, Nx-1):
            V_prime[ix] = (V[ix+1] - V[ix-1])/(2*dx)
            
        # Phi prime
        Phi_prime = np.zeros(Nsteps)
        Phi_prime[0] = (phi[1] - phi[0])/dt
        Phi_prime[-1] = (phi[-1] - phi[-2])/dt

        for it in range(1, Nsteps-1):
            Phi_prime[it] = (phi[it+1] - phi[it-1])/(2*dt)

        V_int = V_prime/(x_star - x)**2

        ax[0,0].plot(x, uu+i*disp, linewidth=1); #uu-i*disp
        ax[0,0].set_ylabel('u normalized (mV)')
        ax[0,0].set_xlabel(r'$x(cm)$')
        ax[0,0].legend()
#         ax[0,1].plot(x, V_int+i*100*disp, linewidth=1,); #V_int-i*100*disp
#         ax[0,1].set_ylabel('V integrand')
#         ax[0,1].set_xlabel(r'$x(cm)$')
        ax[1,0].plot(x, V_prime+i*100*disp) #, label='t=%5.2f'%t[it]); #V_prime-i*100*disp
        ax[1,0].set_ylabel('V prime')
        ax[1,0].set_xlabel(r'$x(cm)$')
        i +=1
        # printing
        frames.append(uu.copy())
        V_primes.append(V_prime.copy())
        tt.append(t)
    ax[1,1].plot(t, Phi_prime, linewidth=1,); 
    #ax[1,1].set_ylabel('Phi prime')
    #ax[1,1].set_xlabel(r'$t$')
    ax[1,1].plot(t, phi)
    ax[1,1].set_ylabel('pseudo EKG')
    ax[1,1].set_xlabel(r'$t(ms)$')
    ax[1,1].set_ylim(-0.02,0.02)
    ax[1,1].axhline(y=0., linestyle='--', linewidth=0.5, color='orange', alpha=1., label=r'$u_c$')
    fig.tight_layout()
    return fig


def advance_x(x, t, u0, v0, w0, Dtilde, dx, dt, params):
    '''Advance solution over all space for one timestep
       Implemented by central differences
    '''
        
    # Propagate with forward-difference in time, 
    # central-difference in space
    Nx=len(u0)
    Du=np.zeros(Nx)
    Dv=np.zeros(Nx)
    Dw=np.zeros(Nx)
    
    # Impose Neumann boundary conditions :
    # end-point derivatives are equal to 0
    u0[0]=u0[1]
    u0[-1]=u0[-2]
    
    tau_w_minus = params['tau_w_minus']
    tau_w_plus = params['tau_w_plus']
    tau_v_plus = params['tau_v_plus']
    u_c = params['u_c']
    
    for i in range(1, Nx-1):
        
        u_xx = (u0[i+1]-2*u0[i]+u0[i-1])/(dx**2)
        
        u_x = (u0[i+1] - u0[i-1])/(2*dx)
        Dtilde_x = (Dtilde[i+1] - Dtilde[i-1])/(2*dx)

        Jstim = J_stim(x[i], t, params)

        puc = p(u0[i], u_c)
        pu = (1 - puc)

        Jfi = J_fi(u0[i], v0[i], puc, params)

        Jso = J_so(u0[i], pu, puc, params)
        Jsi = J_si(u0[i], w0[i], params)

        Du[i] =  (Dtilde_x  * u_x) + (Dtilde[i] * u_xx) - (Jfi + Jso + Jsi - Jstim)
        #Du[i] =  (Dtilde * u_xx) - (Jfi + Jso + Jsi - Jstim)

        Dw[i] = pu * (1 - w0[i])/tau_w_minus\
                - puc * w0[i]/tau_w_plus

        Dv[i] = pu * (1 - v0[i])/tau_v_minus(u0[i], params)\
                 - puc * v0[i]/tau_v_plus

    return Du, Dw, Dv

def advance(x, t, params):
    '''Advance solution one time step
       Implemented by RK4
    '''
    
    Nsteps = t.size
    Nx = x.size
    T0 = params['T0']
    
    tau_v2_minus = params['tau_v2_minus']
    tau_w_minus = params['tau_w_minus']

    # constant
    #D_tilde = params['D'] 
    
    # x dependant
    Dtilde = D_tilde(x, params)
    
    u_2D = np.zeros((Nx, Nsteps))
    v_2D = np.zeros((Nx, Nsteps))
    w_2D = np.zeros((Nx, Nsteps))

    # Set initial conditions
    v_2D[:, 0] = (1-np.exp(-T0/tau_v2_minus))
    w_2D[:, 0] = (1-np.exp(-T0/tau_w_minus))

    dx = x[1]-x[0]
    print(f'inside advance dx={dx}')
    dt = t[1]-t[0]
    print(f'inside advance dt={dt}')
    dt2 = 0.5 * dt
    dt3 = (1/6) * dt
    
    # Iterate through all timesteps

    
    for n in tqdm(range(Nsteps-1)):

        if t[n] < T0:
            v_2D[:, n] = (1-np.exp(-t[n]/tau_v2_minus))
            w_2D[:, n] = (1-np.exp(-t[n]/tau_w_minus))
        else:   
            u0=u_2D[:,n]
            v0=v_2D[:,n]
            w0=w_2D[:,n]

            # Iterate through all space x for one timestep    
            Ku1, Kw1, Kv1 = advance_x(
                                     x, t[n], 
                                     u0, v0, w0,
                                     Dtilde, dx, dt, 
                                     params)

            u1 = u0 + dt2 * Ku1
            v1 = v0 + dt2 * Kv1
            w1 = w0 + dt2 * Kw1

            Ku2, Kw2, Kv2 = advance_x(
                                     x, t[n]+dt2, 
                                     u1, v1, w1,
                                     Dtilde, dx, dt, 
                                     params)

            u2 = u0 + dt2 * Ku2
            v2 = v0 + dt2 * Kv2
            w2 = w0 + dt2 * Kw2

            Ku3, Kw3, Kv3 = advance_x(
                                     x, t[n]+dt2, 
                                     u2, v2, w2,
                                     Dtilde, dx, dt, 
                                     params)

            u3 = u0 + dt2 * Ku3
            v3 = v0 + dt2 * Kv3
            w3 = w0 + dt2 * Kw3

            Ku4, Kw4, Kv4 = advance_x(
                                     x, t[n]+dt, 
                                     u3, v3, w3,
                                     Dtilde, dx, dt, 
                                     params) 

            u_2D[:, n+1] = u0 + dt3 * (Ku1 + 2*Ku2 + 2*Ku3 + Ku4)
            v_2D[:, n+1] = v0 + dt3 * (Kv1 + 2*Kv2 + 2*Kv3 + Kv4)
            w_2D[:, n+1] = w0 + dt3 * (Kw1 + 2*Kw2 + 2*Kw3 + Kw4)
        
    return u_2D, v_2D, w_2D 

#### delete
# def append_pseudo_params2(Phi, params):
#     """ Calculate several attributes of the pseudo ECG and 
#     add them to the 'params' dictionary. Attributes are:
#     Arguments
#     -------------
#     Phi: the 1D signal array
#     params: old parameters dictionary
             
#     Returns
#     ------------
#     params: new parameters dictionary
#     """
    
#     def find_root(f:np.array, low, high):
#         roots = []
#         for n in range(low,high-1):
#             fprod=f[n]*f[n+1]
#             if fprod<=0: 
#                 roots.append(n)
#                 #print(f'found prod={fprod} at {n}')
#         root = roots[-1] if roots else low  
#         return root    

#     def find_first_root(f:np.array, low, high):
#         roots = []
#         for n in range(low,high-1):
#             fprod=f[n]*f[n+1]
#             if fprod<=0: 
#                 roots.append(n)
#                 #print(f'found prod={fprod} at {n}')
#         root = roots[0] if roots else high 
#         return root   

#     def calc_TWAVE(X:np.array, low, high):
#         '''
#         Arguments
#         ---------
#         X   : a 1D EKG array of shape (timesteps), SHOULD BE A SINGLE LEAD
#         low : effectively the 'QRSWAVE_high' since we have no 'TWAVE_low'
#         high: end of T wave, 'TWAVE_high'

#         Returns
#         -------
#         T_low   : start of T as indicated by slope 
#         T_peak  : a number indicating the time of the T peak from start of 
#         the whole signal
#         T_area: area under T (positive or negative) - under construction
#         T_lslope:
#         T_ratio = (T_peak - T_low)/(T_high - T_peak) indicating accelaration 
#                                                     and descelaration duration
#         '''  

#         if X.ndim > 1: 
#             print(f'ERROR: input array must to be 1D but it\'s {X.ndim}D')
#             return 0

#         # smooth the signal 
#         sigma = 5
#         gauss = gaussian_filter1d(X, sigma, mode='constant') 
#         X = gauss

#         start = int(((high-low)/2)+low)
#         # pinpoint area to look for T peak
#         wave = X[start:high].copy()
#         T_peak_pos = np.argmax(wave) + start  # correction from T_peak = np.argmax(X) 
#                                               # so it does not look outside of T
#                                               # addition of `low` is because we loose the
#                                               # generality when going into wave vs. X
#         T_peak_neg = np.argmin(wave) + start 
#         T_peak = T_peak_pos if (np.abs(X[T_peak_pos]) > np.abs(X[T_peak_neg])) \
#                     else T_peak_neg
#         T_height = X[T_peak] 

#         T_halfmax = T_height/2
#         T_midtime = find_root(X-T_halfmax, low, T_peak)

#         T_root = (2*T_midtime)-T_peak
#         T_low = T_root if T_root>low else low
#         T_high = high

#         return T_low, T_peak
    
#     ###### main
#     Nsteps, T0, T = int(params['Nsteps']), params['T0'], params['T']
#     t = np.linspace(T0, T, Nsteps)
        
#     # Find R peak
#     peaks = find_pseudo_ECG_peaks(Phi)    
#     widths, width_heights, left_ips, right_ips = signal.peak_widths(Phi, peaks, rel_height=0.95)
    
#     try: pseudo_R = Phi[peaks[0]]
#     except: pseudo_R = 0.
#     params['pseudo_R'] = pseudo_R
    
#     tol = 1e-5
#     low = np.where(np.isclose(t, 200, tol))[0][0]
#     high = np.where(np.isclose(t, 350, tol))[0][0]
#     print(low, high)
    
#     #try:
#     T_low, pseudo_T = calc_TWAVE(Phi, low, high)
#     #except: pseudo_T = 0.
#     params['pseudo_T'] = pseudo_T
        
#     return params

# def multi2_plot_pseudo_ECG(Phi_list, params_list, 
#                           xrange=None, yrange=None, 
#                           FIGSIZE=10, markers=True, dpi=paper_dpi):
    
#     labelsize = 10
#     num_channels = len(Phi_list)
#     fig, ax = plt.subplots(num_channels, 1, 
#                            figsize=(FIGSIZE*2., 
#                                     FIGSIZE*num_channels), 
#                            dpi=dpi)

#     # change all spines
#     for axis in ['top','bottom','left','right']:
#         ax.spines[axis].set_linewidth(1)

#     plt.yticks(fontsize=labelsize)
#     plt.xticks(fontsize=labelsize)
#     ax.set_ylabel('Pseudo V(mV)', fontsize=labelsize)
#     ax.set_xlabel('t(ms)', fontsize=labelsize)
#     ax.axhline(y=0., linestyle='--', linewidth=0.5, color='orange', label='zero')
#     if xrange: ax.set_xlim(xrange)
#     if yrange: ax.set_ylim(yrange) 
#     yoffset = 0.05
#     xoffset = 10
#     n = 0
#     for Phi, params in list(zip(Phi_list, params_list)):
#         if num_channels!=1: ax = axes[n]
#         else: ax = axes
#         Nsteps, T0, T, Lscar = int(params['Nsteps']), params['T0'], params['T'], params['Lscar']
#         t = np.linspace(T0, T, Nsteps)
#         ax.plot(t+i*xoffset, Phi, label=f'Lscar={Lscar} cm');
#         i +=1
#         if markers:
#             ax.axhline(y=params['pseudo_ST_elev'], linestyle='--', linewidth=0.5, color='blue', label='ST elevation')
#             plt.vlines(x=t[params['TWAVE_low']], linestyle='--', ymin=Phi[params['TWAVE_low']]-yoffset, \
#                        ymax=Phi[params['TWAVE_low']]+yoffset, linewidth=0.5, color='green')
#             plt.vlines(x=t[params['QRSWAVE_high']], linestyle='--', ymin=Phi[params['QRSWAVE_high']]-yoffset, \
#                        ymax=Phi[params['QRSWAVE_high']]+yoffset, linewidth=0.5, color='green')
#             plt.vlines(x=t[params['TWAVE_high']], linestyle='--', ymin=Phi[params['TWAVE_high']]-yoffset, \
#                        ymax=Phi[params['TWAVE_high']]+yoffset, linewidth=0.5, color='green')
#             plt.vlines(x=t[params['RPEAK']], linestyle='--', ymin=Phi[params['RPEAK']]-yoffset, \
#                        ymax=Phi[params['RPEAK']]+yoffset, linewidth=0.5, color='green')
#             plt.vlines(x=t[params['TPEAK']], linestyle='--', ymin=Phi[params['TPEAK']]-yoffset, \
#                        ymax=Phi[params['TPEAK']]+yoffset, linewidth=0.5, color='green')

#             plt.plot(t[params['RPEAK']], Phi[params['RPEAK']], "*", color='red', label='R wave')
#             plt.plot(t[params['TPEAK']], Phi[params['TPEAK']], "*", color='blue', label='T wave')
#         n +=1
#     ax.legend(fontsize=labelsize);
#     tick_spacing = 100
#     ax.xaxis.set_major_locator(ticker.AutoLocator())
    
#     return fig

# # def multi2_plot_pseudo_ECG(Phi_list, params_list, 
# #                           xrange=None, yrange=None, 
# #                           FIGSIZE=10, markers=True, dpi=paper_dpi):
    
# #     labelsize = 10
# #     num_channels = len(Phi_list)
# #     fig, ax = plt.subplots(num_channels, 1, 
# #                            figsize=(FIGSIZE*2., 
# #                                     FIGSIZE*num_channels), 
# #                            dpi=dpi)

# #     # change all spines
# #     for axis in ['top','bottom','left','right']:
# #         ax.spines[axis].set_linewidth(1)

# #     plt.yticks(fontsize=labelsize)
# #     plt.xticks(fontsize=labelsize)
# #     ax.set_ylabel('Pseudo V(mV)', fontsize=labelsize)
# #     ax.set_xlabel('t(ms)', fontsize=labelsize)
# #     ax.axhline(y=0., linestyle='--', linewidth=0.5, color='orange', label='zero')
# #     if xrange: ax.set_xlim(xrange)
# #     if yrange: ax.set_ylim(yrange) 
# #     yoffset = 0.05
# #     xoffset = 10
# #     n = 0
# #     for Phi, params in list(zip(Phi_list, params_list)):
# #         if num_channels!=1: ax = axes[n]
# #         else: ax = axes
# #         Nsteps, T0, T, Lscar = int(params['Nsteps']), params['T0'], params['T'], params['Lscar']
# #         t = np.linspace(T0, T, Nsteps)
# #         ax.plot(t+i*xoffset, Phi, label=f'Lscar={Lscar} cm');
# #         i +=1
# #         if markers:
# #             ax.axhline(y=params['pseudo_ST_elev'], linestyle='--', linewidth=0.5, color='blue', label='ST elevation')
# #             plt.vlines(x=t[params['TWAVE_low']], linestyle='--', ymin=Phi[params['TWAVE_low']]-yoffset, \
# #                        ymax=Phi[params['TWAVE_low']]+yoffset, linewidth=0.5, color='green')
# #             plt.vlines(x=t[params['QRSWAVE_high']], linestyle='--', ymin=Phi[params['QRSWAVE_high']]-yoffset, \
# #                        ymax=Phi[params['QRSWAVE_high']]+yoffset, linewidth=0.5, color='green')
# #             plt.vlines(x=t[params['TWAVE_high']], linestyle='--', ymin=Phi[params['TWAVE_high']]-yoffset, \
# #                        ymax=Phi[params['TWAVE_high']]+yoffset, linewidth=0.5, color='green')
# #             plt.vlines(x=t[params['RPEAK']], linestyle='--', ymin=Phi[params['RPEAK']]-yoffset, \
# #                        ymax=Phi[params['RPEAK']]+yoffset, linewidth=0.5, color='green')
# #             plt.vlines(x=t[params['TPEAK']], linestyle='--', ymin=Phi[params['TPEAK']]-yoffset, \
# #                        ymax=Phi[params['TPEAK']]+yoffset, linewidth=0.5, color='green')

# #             plt.plot(t[params['RPEAK']], Phi[params['RPEAK']], "*", color='red', label='R wave')
# #             plt.plot(t[params['TPEAK']], Phi[params['TPEAK']], "*", color='blue', label='T wave')
# #         n +=1
# #     ax.legend(fontsize=labelsize);
# #     tick_spacing = 100
# #     ax.xaxis.set_major_locator(ticker.AutoLocator())
    
# #     return fig

