import numpy as np
import h5py
import dask.array as da
import dask.dataframe as dd
import random

from scipy.optimize import curve_fit


def sumchan_helper(raw_fims,rois):
    '''
    Function. that does the actual summing of fim and crix channels

    Parameters
    ----------
    raw_fims : array
        containing the actual data, either pre-processed or raw - these cases will be distinguished below

    rois : array
        describing the area containing background and the region of interest, 
        as well as the channels that should be used for reduction.
        Defined in the config yaml file.

    Returns
    -------
    fimsum : 1D array containing sum over ROI and channels

    '''
    # raw_fims=raw_fims.compute_chunk_sizes()
    if raw_fims.ndim == 3:    
        bg = (raw_fims[...,rois['bg_roi'][0]:rois['bg_roi'][1]]).mean(axis=-1)
        # print(raw_fims.shape)
        bgf = raw_fims - bg[...,None]
        fim_sum = np.zeros([bgf.shape[0],len(rois['channels'])])
        i=0
        #TODO: is this for loop the most efficient way - probably yes, since ROIs 
        # may be channel dependent ; confirm channel numbers match
        for c in rois['channels']:
            fim_sum[:,i] = np.nansum(np.abs(bgf[:,c-1,rois['roi'][i][0]:rois['roi'][i][1]]),axis = 1)
            i=i+1
        fimsum =np.nansum(fim_sum, axis=1)
    
    elif raw_fims.ndim == 2:
        fimsum = np.zeros([raw_fims.shape[0]])
        for c in rois['channels']:
            fimsum = fimsum + (raw_fims[:,c-1]).squeeze()
    
    elif raw_fims.ndim==1:
        fimsum = raw_fims

    return fimsum

def sum_channels(obj,fyaml): #dict includes {fim0: fim_0,....}
    '''
    Function to handle singleshot or integrating fims and crix 
    
    Different cases depending if object from singleshot or array from 
    integrating detectors is being parsed

    Parameters
    ----------
    obj : object
        detector object containing fim or crixs data
    channel_dict : dictionary
        dict containing list of which detectors to process, 
        and how the attribute should be called
    fymal : dictionary
        Dict from yml file that contains information of ROIs for the different detectors
    '''
    rois = fyaml
    channel_dict = fyaml['channels_to_integrate']
    
    for key in channel_dict: 
        #if we are parsing from the integrating class this is an array
        # try:
        a=getattr(obj,key)
        # print(a)
        if hasattr(getattr(obj,key), "__len__"):
            summed = sumchan_helper(getattr(obj,key), rois[channel_dict[key]]) 
            setattr(obj, channel_dict[key], summed)
        #if we are parsing from the singleshot class this is an object
        #FIXME: implement option for getting channels from preproc or full area
        else:
            summed = sumchan_helper(getattr(getattr(obj,key),'preproc'),rois[channel_dict[key]]) 
            setattr(obj, channel_dict[key], summed)
        # except:
        #     print(f'{key} does not exits ')

def normalise(dat, I0):
    '''
    Function to normalise a given signal to a given reference 
    
    Mostly normalisation of an n-dimensional detector signal by the X-ray intensities 

    Parameters
    ----------
    dat : array
        n-dimensional array with n e{1,2,3}, with one dimension being N = number of images
    I0 : dictionary
        1D array containing I0 for each corrected image

    Returns
    -------
    norm : normalised detector array
    '''
    #FIXME: not sure if I need to implement different cases for when dimensions are in a different order
    # dat.compute_chunk_sizes()
    # I0.compute_chunk_sizes()
    if dat.ndim == 1:
        norm = dat/I0
    elif dat.ndim == 2:
        norm = dat/I0[:,np.newaxis]
    elif dat.ndim == 3:
        norm = dat/I0[:,np.newaxis,np.newaxis]
    else:
        raise ValueError('Dimension do not match for normalising detector')
    return norm


def bin_data(data,bin_axis,bins,scantype='fly'):
    
    
    ## JW edit: instead of being Nbins, bin_wdith, or bin_edges, we can just have 2 settings, uniform or bin_edges
    # bin_range 
    # unifrom requires #bins. 
    # bin_edges requires a list of the bin edges that are already custom ready.
    
    # the scantype for "step" and "fly" are still good. 
    '''
    Function binning a given data set along a given binning variable
    
    Binning can be done for different scan types (fly or step scan), and independent of what the 
    scanvariable is. Different ways of binning can be determined

    Parameters
    ----------
    data : array
        n-dimensional array with n e{1,2,3}, with one dimension being N = number of images
    bin_axis : array
        1xN array containing the values of the scanvariable for each image 
    bins : list
        bins[0] determines the type of binning ('Nbins','bin_width','bin_edges')
        bin[1] defines for 
            'Nbins': number of bins
            'bin_width': the width of each bin
            'bin_edges': [start value, end value, #steps]

    Returns
    -------
    bin_centers: array
        binned scanvariable 
    binned_dat_sum : array
        n D array containing the data summed per bin
    binned_dat_mean : array
        n D array containing the data averaged per bin
    binned_dat_std : array
        n D array containing the standard deviation of the data per bin
    '''
    #FIXME: do I call this function for each detector somewhere else, or do I loop through the detectors here
    # for now writing this for an individual detector, do on and off stuff outside this funciton too
    bin_axis = bin_axis.squeeze()
    if bin_axis.ndim > 1:
        raise ValueError('scanvar too many dimensions')
    if False:
        idx = da.argsort(bin_axis)
    else:
        idx = np.argsort(bin_axis)

    #FIXME: for single shot we will need an approximate / chunked sort 

    
    bin_axis = bin_axis[idx]
    data = data[idx]

    #Create bins depending on type of scan
    if scantype == 'fly':
        print('fly)')
        if bins[0] == 'Nbins':
            print('Nbins')
            bin_counts, bin_edges = np.histogram(bin_axis, bins=bins[1], density=False)
        elif bins[0] == 'bin_width':
            print('bin_width')
            Nbins = int((np.max(bin_axis)-np.min(bin_axis))/bins[1])
            bin_counts, bin_edges = np.histogram(bin_axis, bins=Nbins, density=False)
        #FIXME: option for bins with equal number of data points
        elif  bins[0] == 'bin_edges':
            print('bin_edges')
            bin_edges = np.linspace(float(bins[1][0]),float(bins[1][1]),int(bins[1][2]))
            bin_counts, bin_edges1 = np.histogram(bin_axis, bin_edges, density=False)


        else:
            raise ValueError('binning type unclear')
    
        # print('bin_counts', len(bin_counts))
        # print('bin_edges', len(bin_edges))
        
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
    elif scantype == 'step':
        # this is assuming that there is no jitter in the mono calibration.
        # if there is jitter, then we will end up with bins that only have 1 or 2 counts. 
        scanvar,bin_counts = np.unique(bin_axis,return_counts=True)
        print(f'scanvar in bin_data {len(scanvar)}')
        bin_edges = scanvar
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    elif scantype == 'static':
        bin_centers = np.mean(bin_axis)
        bin_edges = [np.mean(bin_axis)]
        bin_counts = len(bin_axis)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
         
    else:
        raise ValueError('scan type for binning not defined')
    bin_edges = np.asarray(bin_edges)
    # print(bin_edges)
    #FIXME: by using digitize are we excluding data points at both ends?

    ######
    #FIXME: does not seem to be working for delay scans
    # |
    # V
    #######
    # print('bin_axis', np.min(bin_axis),bin_axis.max())
    # print('bin_edges',np.min(bin_edges),bin_edges.max())
    # print('bin_width', bin_widths)

    inds = np.digitize(bin_axis,bin_edges)
    # print(inds)

    if data.ndim == 1:
        binned_dat_sum  = np.zeros(bin_edges.shape[0])
        binned_dat_mean = np.zeros(bin_edges.shape[0])
        binned_dat_std  = np.zeros(bin_edges.shape[0])
        
    elif data.ndim == 2:
        binned_dat_sum  = np.zeros([bin_edges.shape[0],data.shape[1]])
        binned_dat_mean = np.zeros([bin_edges.shape[0],data.shape[1]])
        binned_dat_std  = np.zeros([bin_edges.shape[0],data.shape[1]])
    else:
        raise ValueError('Detector shape not known')
    #FIXME: normalisation by bin counts missing
    # print('bin_counts',bin_counts)
    # bin_counts =np.append(1,bin_counts[:])
    for i in np.arange(len(bin_edges)):
        if not sum((inds==i))==0:
            binned_dat_sum[i,:]  = np.nansum(data[inds==i],0)#/bin_counts[i]
            binned_dat_mean[i,:] = np.nanmean(data[inds==i],0)
            binned_dat_std[i,:]  = np.nanstd(data[inds==i],0)


    return bin_centers, binned_dat_sum[1:,:], binned_dat_mean[1:,:], binned_dat_std[1:,:], bin_counts

def myround(x, base=5):
    return base * np.round(x/base)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_premirror_pitch(premirror_pitch):
    try:
        return np.nanmean(myround(premirror_pitch,1))
    except:
        # This needs improving
        print('Add SP1K1:MONO:MMS:M_PI.RBV to epics config')
        return 144506
    
def mono_energy(pitchG,pitchM2,stateG = 'LRG', fname='../mono_calib.yml'):
    '''Calculator for RIXS mono calibration. This is the same function as 
    what is saved to the mono eV epics variable. 
    pitchG: Grating pitch in urad
    pitchM2: Pre-mirror pitch in urad.
    '''
    import yaml
    with open(fname, 'r') as file:
        fy = yaml.safe_load(file)
    if stateG=='LRG':
        D0 = fy['D0_LRG']
        offsetG = fy['offsetG_LRG']
    elif stateG=='LEG':
        D0 = fy['D0_LEG']
        offsetG = fy['offsetG_LEG']
    elif stateG=='MEG':
        D0 = fy['D0_MEG']
        offsetG = fy['offsetG_LRG']
    elif stateG=='HEG':
        D0 = fy['D0_HEG']
        offsetG = fy['offsetG_HEG']

     # constants
    eVmm = 0.001239842 # Wavelenght[mm] = eVmm/Energy[eV]
    m = 1 # diffraction order

    pG = pitchG*1e-6 - offsetG
    pM2 = pitchM2*1e-6 - fy['offsetM2']
    alpha = np.pi/2 - pG + 2*pM2 - fy['thetaM1']
    beta = -np.pi/2 - pG + fy['thetaES']
    E = m*D0*eVmm/(np.sin(alpha) + np.sin(beta))
    Cff = np.cos(beta)/np.cos(alpha)

    #print('Calculated photon energy {0:6.2f} eV, Cff {1:3.2f}'.format(E, Cff))
    return E
 

def avg_data(runs : list, proc_folder : str = '',proc_path : str = ''):
    avg = {}
    with h5py.File(proc_folder+f'{runs[0]:04d}.h5','r') as tmp:
        keys = list(tmp.keys())
        # print(keys)
        for key in keys:
            avg[key] = np.zeros(tmp[key].shape)

    i=0
    for run in runs:
        i=i+1
        file = proc_folder+f'{run:04d}.h5'
        # FIXME: 
        for key in keys:    
            with h5py.File(file,'r') as f:
                if avg[key].shape==f[key].shape:
                    avg[key] = avg[key] + np.asarray(f[key])
                    # print(f[key])
                else:
                    #FIXME: Here implement stuff for stitching
                    print(f'Run {run} {key} shapes do not match')
            
    for key in keys:
        avg[key] = avg[key]/i
    return avg

# ### **Uncertainty Formula for unweighted Average**:
# [\sigma_{\bar{x}} = \frac{1}{\sqrt{N}} \sqrt{\sum_{i=1}^{N} \sigma_i^2}]
   

# def avg_data_count(runs : list, proc_folder : str = '',proc_path : str = ''):
#     avg = {}
#     with h5py.File(proc_folder+f'{runs[0]:04d}.h5','r') as tmp:
#         keys = list(tmp.keys())
#         if 'counts_on' in keys:
#             counts_on = np.zeros(tmp['counts_on'].shape)
#             counts_off = np.zeros(tmp['counts_off'].shape)
#             laser = True
#         else:
#             counts = np.zeros(tmp['counts'].shape)
#             laser = False

#         # print(keys)
#         for key in keys:
#             avg[key] = np.zeros(tmp[key].shape)

#     i=0
#     for run in runs:
#         i=i+1
#         file = proc_folder+f'{run:04d}.h5'

#         #load processed data
#         with h5py.File(file,'r') as f:
#             # define count
#             if laser:
#                 count_on = np.asarray(f['counts_on'])
#                 count_off = np.asarray(f['counts_off'])
#                 laser = True
#             else:
#                 count = np.asarray(f['counts'])

#             # do count-weighted averaging
#             for key in keys:
#                 #####ZY_edits - 060126 - make the sum keys work
#                 if key.endswith('sumSV') or key.endswith('sumI0'):
#                     if avg[key].shape == f[key].shape:
#                         avg[key] = avg[key] + np.asarray(f[key])
#                     else:
#                         print(f'Run {run} {key} shapes do not match')
#                     continue

#                 if laser:
#                     if 'on' in key:
#                         count = count_on
#                     elif 'off' in key:
#                         count = count_off
#                 else:
#                     count = count

#                 if avg[key].shape==f[key].shape:
#                     if len(avg[key].shape) == 1:
#                         avg[key] = avg[key] + np.asarray(f[key])*count
#                     else:
#                         avg[key] = avg[key] + np.asarray(f[key])*count[:,np.newaxis]
#                 else:
#                     #FIXME: Here implement stuff for stitching
#                     print(f'Run {run} {key} shapes do not match')
#         # Sum counts per scanvar
#         if laser:
#             counts_on = counts_on + count_on
#             counts_off = counts_off + count_off
#             print(counts_off)

#         else:
#             counts = counts + count
#     # Divide by count per scanvar
#     for key in keys:
#         #####ZY_edits - 060126 - make the sum keys work
#         if key.endswith('sumSV') or key.endswith('sumI0'):
#             continue
#         if laser:
#             if 'on' in key:
#                 if len(avg[key].shape) == 1:
#                     avg[key] = avg[key]/counts_on
#                 else:
#                     avg[key] = avg[key]/counts_on[:,np.newaxis]
#             elif 'off' in key:
#                 if len(avg[key].shape) == 1:
#                     avg[key] = avg[key]/counts_off
#                 else:
#                     avg[key] = avg[key]/counts_off[:,np.newaxis]
#         else:
#             if len(avg[key].shape) == 1:
#                 avg[key] = avg[key]/counts
#             else:
#                 avg[key] = avg[key]/counts[:,np.newaxis]
#     return avg

#FIXME

# ### **Uncertainty Formula for Weighted Average**:

# [
# \sigma_{\bar{x}*{\text{w}}} = \sqrt{\frac{1}{\sum*{i=1}^{N} w_i^2} \sum_{i=1}^{N} w_i^2 \sigma_i^2}
# ]

# Where:

# * ( \sigma_{\bar{x}_{\text{w}}} ) is the uncertainty in the weighted average.
# * ( w_i ) are the weights.
# * ( \sigma_i ) are the individual uncertainties for each measurement.

# NOTE: JW comment 10/08/26: this function is not being used right now?
def pixel2emi(pixel, dat, mono=[], calib=[], points=[], w_calib_line=10, plot=True):
    if calib == []:
        if mono == []:
            raise TypeError('Need mono energies for emission calibration')
        else:
            emi,calib = calib_emi(mono, dat, points, w_calib_line=w_calib_line, plot=plot)
    else:
        emi = pixel*calib[0]+calib[1]

    return(emi,calib)


# JW edits 10/08/26: fixed some bugs
def emi2ET(mono, emission, data, std, step):
    '''
    Rebin a (mono, emission) map onto a (mono, energy transfer) grid.

    Energy transfer for each pixel is mono - emission. Pixels are grouped into
    energy-transfer bins of width `step`; each bin holds the mean of its pixels,
    and its error is the error of that mean.

    Parameters
    ----------
    mono : (n_mono,) incident energies [eV]
    emission : (n_pix,) emission energy of each pixel [eV]
    data : (n_mono, n_pix) map
    std : (n_mono, n_pix) error of each map element
    step : energy-transfer bin width [eV]

    Returns
    -------
    mono : (n_mono,) unchanged
    E_trans : (n_bins,) energy-transfer bin CENTERS [eV]
    data_trans : (n_mono, n_bins) mean of the pixels in each bin, NaN if empty
    std_trans : (n_mono, n_bins) error of that mean, NaN if empty
    '''
    mono = np.asarray(mono, dtype=float)
    emission = np.asarray(emission, dtype=float)
    data = np.asarray(data, dtype=float)
    std = np.asarray(std, dtype=float)

    # NOTE: energy transfer of every map element, (n_mono, n_pix):
    #       row i is mono[i] minus the emission energy of each pixel
    Etrans_in = mono[:, np.newaxis] - emission[np.newaxis, :]

    # NOTE: full range covered by any mono row. Rows that do not reach the
    #       ends of this range get NaN there, so they show as blank
    Emin = np.min(Etrans_in)
    Emax = np.max(Etrans_in)

    # NOTE: explicit bin EDGES, starting at Emin and extending just past Emax
    #       so every pixel falls inside some bin
    n_bins = int(np.floor((Emax - Emin) / step)) + 1
    edges = Emin + step * np.arange(n_bins + 1)

    # NOTE: the returned axis is the bin CENTERS, so pcolor and the CIE cuts
    #       place each bin at its middle, not its lower edge
    E_trans = 0.5 * (edges[:-1] + edges[1:])

    # NOTE: empty bins stay NaN, so they are blank in plots and skipped by
    #       nansum in the CIE cuts, instead of counting as measured zeros
    data_trans = np.full((len(mono), n_bins), np.nan)
    std_trans = np.full((len(mono), n_bins), np.nan)

    # NOTE: a pixel is usable only if both its value and its error are finite,
    #       so the mean and its error are computed from the same pixels
    is_valid = np.isfinite(data) & np.isfinite(std)

    for row in range(len(mono)):
        for b in range(n_bins):

            # NOTE: half-open bin [edge_b, edge_b+1), so a pixel exactly on an
            #       edge lands in one bin instead of none
            in_bin = (Etrans_in[row] >= edges[b]) & (Etrans_in[row] < edges[b + 1])
            usable = in_bin & is_valid[row]

            n_pixels = usable.sum()
            if n_pixels == 0:
                continue

            data_trans[row, b] = np.mean(data[row, usable])

            # NOTE: error of the mean of n independent pixels,
            #       sqrt(sum(std^2)) / n
            std_trans[row, b] = np.sqrt(np.sum(std[row, usable]**2)) / n_pixels

    return mono, E_trans, data_trans, std_trans

def calib_emi(mono, dat, end_points_el, w_calib_line=5, plot=True):
    #FIXME
    ix1 = find_nearest(mono,end_points_el[0][0])
    ix2 = find_nearest(mono,end_points_el[1][0])
    iy1 = end_points_el[0][1]
    iy2 = end_points_el[1][1]

    nx, ny = dat.shape

        # --- Fit center line y ≈ m*x + c in (mono units -> pixel)
    # Use the axis values (mono[ix]) for x
    x_vals = np.array([mono[ix1], mono[ix2]], dtype=float)
    y_vals = np.array([iy1, iy2], dtype=float)
    #y = m*x+c
    m = (iy2-iy1)/(mono[ix2]-mono[ix1])
    c = iy1-m*mono[ix1]

    mono_points = []
    ypix_at_max = []

    # --- Scan each x index within the span; search along y within top/bottom bounds
    for ix in np.arange(ix1, ix2 + 1):
        x_here = mono[ix]
        y_center = m * x_here + c
        y_lo = int(np.floor(y_center - w_calib_line/2))
        y_hi = int(np.ceil (y_center + w_calib_line/2))
        
        if y_hi < y_lo:
            y_lo, y_hi = [y_hi, y_lo]

        rixs_slice = dat[ix, y_lo:y_hi + 1]
        if rixs_slice.size == 0 or np.all(np.isnan(rixs_slice)):
            continue

        rel = int(np.nanargmax(rixs_slice))
        iy_max = y_lo + rel

        mono_points.append(x_here)
        ypix_at_max.append(iy_max)

    mono_points = np.asarray(mono_points)
    ypix_at_max = np.asarray(ypix_at_max)

    a,b = np.polyfit(ypix_at_max, mono_points, 1)
    emi = np.polyval([a, b], np.arange(ny))

    if plot_on:
        plt.figure(figsize=(7, 6))
        # Display Z with axes: x = mono (horizontal), y = pixel (vertical)
        extent = [mono.min(), mono.max(), 0, ny - 1]
        # plt.imshow(Z.T, origin='lower', extent=extent, aspect='auto')
        plt.pcolormesh(mono,range(0,dat.shape[1]),dat.T,cmap='terrain_r')
        plt.colorbar(label='Intensity')

        # Draw the two points
        plt.scatter([x_vals[0], x_vals[1]], [y_vals[0], y_vals[1]],
                    s=60, c='white', edgecolor='k', label='given points')
        plt.scatter(mono_points, ypix_at_max,
                    s=60, c='white', edgecolor='k', label='given points')

        # Draw center and top/bottom edges
        x_line = np.linspace(mono[ix1], mono[ix2], 200)
        y_center = m * x_line + c
        y_top = y_center +  w_calib_line/2
        y_bot = y_center -  w_calib_line/2
        plt.plot(x_line, y_center, 'w--', lw=1.5, label='center line',color='k')
        plt.plot(x_line, y_top,    'w-',  lw=1.0, alpha=0.8,color='k')
        plt.plot(x_line, y_bot,    'w-',  lw=1.0, alpha=0.8,color='k')

        # Scatter maxima used for the fit
        plt.scatter(mono_points, ypix_at_max, s=20, c='yellow', edgecolor='k', label='max @ each x')

        # Plot fitted calibration line (x vs pixel)
        ypix = np.arange(ny)
        plt.plot(np.polyval([a, b], ypix), ypix, 'r-', lw=2,
                 label=f'fit: mono = {a:.6g} * pixel + {b:.6g}')
        plt.xlim(mono[0],mono[-1])
        plt.xlabel('Energy (mono)')
        

    return emi, [a,b]


def gaussian_moments(x, y):
    """
    Fast Gaussian fit using moments.

    Returns
    -------
    A     : amplitude
    mu    : center
    sigma : width
    C     : offset
    """

    # estimate offset from edges
    C = np.median(np.concatenate([y[:10], y[-10:]]))
    y0 = y - C

    # guard against bad curves
    if np.all(y0 <= 0):
        return np.nan, np.nan, np.nan, C

    y0 = np.clip(y0, 0, None)

    norm = np.sum(y0)
    mu = np.sum(x * y0) / norm
    sigma = np.sqrt(np.sum((x - mu)**2 * y0) / norm)
    A = np.max(y0)

    return A, mu, sigma, C


def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

def gaussian_linear(x, A, mu, sigma, C, D):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + (C + D * x)

def asymmetric_gaussian(x, A, mu, sigma1, sigma2,C,D):
    return np.where(x < x0,
                    A * np.exp(-((x - mu)**2) / (2 * sigma1**2)),
                    A * np.exp(-((x - mu)**2) / (2 * sigma2**2)))+D*x+C

def gaussian_moments_linear(x, y):
    """
    Fast Gaussian fit using moments with linear baseline.

    Returns
    -------
    A     : amplitude
    mu    : center
    sigma : width
    C     : baseline intercept
    D     : baseline slope
    """

    # estimate baseline from edges
    n = min(10, len(x) // 2)
    x_edges = np.concatenate([x[:n], x[-n:]])
    y_edges = np.concatenate([y[:n], y[-n:]])

    # linear fit to edges
    D, C = np.polyfit(x_edges, y_edges, 1)

    baseline = C + D * x
    y0 = y - baseline

    # guard against bad curves
    if np.all(y0 <= 0):
        return np.nan, np.nan, np.nan, C, D

    y0 = np.clip(y0, 0, None)

    norm = np.sum(y0)
    mu = np.sum(x * y0) / norm
    sigma = np.sqrt(np.sum((x - mu)**2 * y0) / norm)
    A = np.max(y0)

    return A, mu, sigma, C, D



def fit_gaussian_fast(x, y, lb, ub):
    p0 = gaussian_moments(x, y)
    bounds = [lb, ub]

    if np.any(np.isnan(p0[:3])):
        return p0

    try:
        p0=np.asarray(p0)
        p0[p0<lb]= lb[p0<lb]
        p0[p0>ub]= ub[p0>ub]
        popt, _ = curve_fit(
            gaussian,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=2000
        )
        return popt
    except RuntimeError:
        return p0
    

def fit_gaussian_fast_linear(x, y, lb, ub):
    tmp = gaussian_moments_linear(x, y)
    p0 = [tmp[0],tmp[1],tmp[2],tmp[2],tmp[3],tmp[4]]
    bounds = [lb, ub]

    if np.any(np.isnan(p0[:3])):
        return p0

    try:
        p0 = np.asarray(p0)
        p0[p0 < lb] = lb[p0 < lb]
        p0[p0 > ub] = ub[p0 > ub]

        popt, _ = curve_fit(
            # gaussian_linear,
            asymmetric_gaussian, #(x, A, mu, sigma1, sigma2,C,D)
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=3000
        )
        return popt
    except RuntimeError:
        return p0


def reconst_svd(u,s,v,N):
    
    data_denoised = (u[:, :N] * s[:N]) @ v[:N, :]
    return data_denoised

def reconst_singlecomp(u,s,v,x):
    component_x = s[x] * np.outer(u[:, x], v[x, :])
    return component_x


# LS Waveform SVD functions 
def decomp_(waves,nevts):    
    # randomly pick up nevts events
    if nevts<waves.shape[0]:
        idx = np.array(random.sample(range(0,waves.shape[0]),nevts))
        # print(idx)
    else:
        idx = np.arange(waves.shape[0])
    start,end = (0,waves.shape[1])  #which section of waveform to use for svd.  use to select out saturation region

    #instead manually calculate svd by using the smaller lh sv.
    s,u = np.linalg.eigh(np.dot(waves[idx,start:end],waves[idx,start:end].transpose()))
    v = np.dot(np.linalg.pinv(u),waves[idx,start:end])

 # get the real component of the eigenvector
    temp = np.array([i/np.dot(i,i)**0.5 for i in v[:]])#normalizing
    v = np.real(temp)
    #print(s.shape,v.shape, np.flipud(s).shape, np.flipud(v).shape)
    return np.flipud(v)[:1000],np.flipud(s)[:1000]


def decomp_1d(vec_dark,val_dark,wfs,ROI_svd,neigs = 2):  # this function does pixel-resolved singular value decomposition
    # ROI SVD 
    #print(val_dark[:neigs].sum()/val_dark.sum())
    target_1d = wfs
    bg_mask = (wfs[0]*0+1).astype(bool)
    bg_mask[ROI_svd[0]:ROI_svd[1]]=False 
    signal_1d = target_1d/np.dot(np.dot(target_1d[:,bg_mask],np.linalg.pinv(vec_dark[:neigs][:,bg_mask])),vec_dark[:neigs])

    return signal_1d[:,ROI_svd[0]:ROI_svd[1]]

# v, s = decomp_(dat[loff],2000) #decomposition of BG data (dark)
# sl = slice(0,-1)
# traces = decomp_1d(v,s,dat[:,:][:],ROI) #all data only ROI



def standard_error_IO(data_matrix):
    '''
    Function to determine the standard error for a distribution with on and off shots and varying intensity of on.

    An example is the SVLS detector after dropletting where each pixel has a value of either 0 or several 100s ADUs

    Parameters
    ----------
    data_matrix : float
    2D matrix containing the raw data with data_matrix.shape[0] is the number of shots

    Returns
    -------
    uncertainty_sum : float
    Array containing the standard error for the sum of all values in bin

    datsum : float
    Array containing the sum of all values in bin

    standard_error_mean : float
    Array containing the standard error for the mean of all values in bin

    datmean : float
    Array containing the mean of all values in bin

    '''
    # Number of shots (N)
    N = data_matrix.shape[0]

    # Determine p_on (proportion of non-zero values for each pixel)
    p_on = np.nansum(data_matrix > 0, axis=0) / N

    # Determine V_on_mean (mean of non-zero values for each pixel)
    V_on_mean = np.divide(
                    np.nansum(data_matrix, axis=0),
                    np.nansum(data_matrix > 0, axis=0),
                    out=np.zeros_like(np.nansum(data_matrix, axis=0), dtype=float),
                    where=np.nansum(data_matrix > 0, axis=0) != 0
                    )

    # Determine V_on_std (standard deviation of non-zero values for each pixel)
    V_on_std = np.nanstd(data_matrix * (data_matrix > 0), axis=0)

    # Calculate the variance for each pixel (row)
    variance_signal = p_on * (V_on_std**2 + V_on_mean**2)

    # # Standard error of the mean for each pixel
    standard_error_mean = np.sqrt(variance_signal) / np.sqrt(N) #if N shots is 0 we got a different problem

    # Uncertainty in the total sum for each pixel
    uncertainty_sum = np.sqrt(N) * np.sqrt(variance_signal)

    datsum = np.nansum(data_matrix,axis=0)

    datmean = np.nanmean(data_matrix,axis=0)

    return datsum, uncertainty_sum, datmean, standard_error_mean



def bin_svls(data,bin_axis,bins,scantype='fly'):
    '''
    Function binning a given data set along a given binning variable
    
    Binning can be done for different scan types (fly or step scan), and independent of what the 
    scanvariable is. Different ways of binning can be determined

    Parameters
    ----------
    data : array
        n-dimensional array with n e{1,2,3}, with one dimension being N = number of images
    bin_axis : array
        1xN array containing the values of the scanvariable for each image 
    bins : list
        bins[0] determines the type of binning ('Nbins','bin_width','bin_edges')
        bin[1] defines for 
            'Nbins': number of bins
            'bin_width': the width of each bin
            'bin_edges': [start value, end value, #steps]

    Returns
    -------
    bin_centers: array
        binned scanvariable 
    binned_dat_sum : array
        n D array containing the data summed per bin
    binned_dat_mean : array
        n D array containing the data averaged per bin
    binned_dat_std : array
        n D array containing the standard deviation of the data per bin
    '''
    #FIXME: do I call this function for each detector somewhere else, or do I loop through the detectors here
    # for now writing this for an individual detector, do on and off stuff outside this funciton too
    bin_axis = bin_axis.squeeze()
    if bin_axis.ndim > 1:
        raise ValueError('scanvar too many dimensions')
    if False:
        idx = da.argsort(bin_axis)
    else:
        idx = np.argsort(bin_axis)

    #FIXME: for single shot we will need an approximate / chunked sort 

    
    bin_axis = bin_axis[idx]
    data = data[idx]

    #Create bins depending on type of scan
    if scantype == 'fly':
        print('fly)')
        if bins[0] == 'Nbins':
            print('Nbins')
            bin_counts, bin_edges = np.histogram(bin_axis, bins=bins[1], density=False)
        elif bins[0] == 'bin_width':
            print('bin_width')
            Nbins = int((np.max(bin_axis)-np.min(bin_axis))/bins[1])
            bin_counts, bin_edges = np.histogram(bin_axis, bins=Nbins, density=False)
        #FIXME: option for bins with equal number of data points
        elif  bins[0] == 'bin_edges':
            print('bin_edges')
            bin_edges = np.linspace(float(bins[1][0]),float(bins[1][1]),int(bins[1][2]))
            bin_counts, bin_edges1 = np.histogram(bin_axis, bin_edges, density=False)


        else:
            raise ValueError('binning type unclear')
    
        # print('bin_counts', len(bin_counts))
        # print('bin_edges', len(bin_edges))
        
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
    elif scantype == 'step':
        scanvar,bin_counts = np.unique(bin_axis,return_counts=True)
        print(f'scanvar in bin_data {len(scanvar)}')
        bin_edges = scanvar
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    elif scantype == 'static':
        bin_centers = np.mean(bin_axis)
        bin_edges = [np.mean(bin_axis)]
        bin_counts = len(bin_axis)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
         
    else:
        raise ValueError('scan type for binning not defined')
    bin_edges = np.asarray(bin_edges)
    # print(bin_edges)
    #FIXME: by using digitize are we excluding data points at both ends?

    ######
    #FIXME: does not seem to be working for delay scans
    # |
    # V
    #######
    # print('bin_axis', np.min(bin_axis),bin_axis.max())
    # print('bin_edges',np.min(bin_edges),bin_edges.max())
    # print('bin_width', bin_widths)

    inds = np.digitize(bin_axis,bin_edges)
    # print(inds)

    if data.ndim == 1:
        binned_dat_sum  = np.zeros(bin_edges.shape[0])
        binned_dat_sumerr  = np.zeros(bin_edges.shape[0])
        binned_dat_mean = np.zeros(bin_edges.shape[0])
        binned_dat_std  = np.zeros(bin_edges.shape[0])
        
    elif data.ndim == 2:
        binned_dat_sum  = np.zeros([bin_edges.shape[0],data.shape[1]])
        binned_dat_sumerr  = np.zeros([bin_edges.shape[0],data.shape[1]])
        binned_dat_mean = np.zeros([bin_edges.shape[0],data.shape[1]])
        binned_dat_std  = np.zeros([bin_edges.shape[0],data.shape[1]])
    else:
        raise ValueError('Detector shape not known')
    #FIXME: normalisation by bin counts missing
    # print('bin_counts',bin_counts)
    # bin_counts =np.append(1,bin_counts[:])
    for i in np.arange(len(bin_edges)):
        if not sum((inds==i))==0:
            binned_dat_sum[i,:], binned_dat_sumerr[i,:], binned_dat_mean[i,:], binned_dat_std[i,:]  = standard_error_IO(data[inds==i])  #/bin_counts[i]
        

    return bin_centers, binned_dat_sum[1:,:], binned_dat_sumerr[1:,:], binned_dat_mean[1:,:], binned_dat_std[1:,:], bin_counts


def get_PFY(rixs,std):
    PFY     = np.sum(rixs, axis=1)
    PFY_std = np.sqrt(np.nansum(std**2, axis=1))
    return PFY, PFY_std


def get_CIE(Ecie, width, Einc, data_trans, std_trans, return_info=False):
    '''
    Constant incident energy (CIE) cut from an (inc. energy, energy transfer) map.

    Sums the mono rows within +-width bins of the row nearest Ecie, and adds
    their errors in quadrature.

    Parameters
    ----------
    Ecie : float
        Incident energy at which to take the cut [eV].
    width : int
        Number of mono BINS on each side of the center row to include, so the
        cut sums up to 2*width + 1 rows.
    Einc : (n_mono,) incident energy axis
    data_trans : (n_mono, n_ET) map
    std_trans : (n_mono, n_ET) error map
    return_info : bool
        Also return a dict describing which rows were used.

    Returns
    -------
    CIE, CIE_std : (n_ET,)  NaN where no selected row has data
    info : dict, only if return_info
        rows       (lo, hi) row slice used
        mono_range (first, last) incident energy of the rows used [eV]
        n_rows     (n_ET,) number of rows contributing to each column
        n_selected number of rows selected
        n_rows_with_data  number of selected rows that have any data
    '''
    Einc = np.asarray(Einc, dtype=float)
    data_trans = np.asarray(data_trans, dtype=float)
    std_trans = np.asarray(std_trans, dtype=float)

    # NOTE: width is a number of bins, so it must be a whole number >= 0
    if width < 0 or int(width) != width:
        raise ValueError(f'width must be a non-negative integer, got {width}')
    width = int(width)

    # NOTE: warn rather than silently cutting at the edge of the scan. Einc
    #       holds bin centers, so allow half a bin beyond the outer centers
    half_bin = 0.5 * np.nanmedian(np.abs(np.diff(Einc))) if len(Einc) > 1 else 0.0
    if not (np.nanmin(Einc) - half_bin <= Ecie <= np.nanmax(Einc) + half_bin):
        print(f'get_CIE: {Ecie} eV is outside the scan '
              f'({np.nanmin(Einc):.2f}-{np.nanmax(Einc):.2f} eV), using the nearest edge')

    center = int(np.nanargmin(np.abs(Einc - Ecie)))

    # NOTE: symmetric window, center +- width, so 2*width + 1 rows. Clipped at
    #       the ends of the scan; a negative start would otherwise wrap around
    #       to the end of the array
    lo = max(center - width, 0)
    hi = min(center + width + 1, len(Einc))

    rows_data = data_trans[lo:hi]
    rows_std = std_trans[lo:hi]

    # NOTE: a cell counts only if both value and error are finite, so the sum
    #       and its error use the same rows
    is_valid = np.isfinite(rows_data) & np.isfinite(rows_std)
    n_rows = is_valid.sum(axis=0)

    CIE = np.sum(np.where(is_valid, rows_data, 0.0), axis=0)
    CIE_std = np.sqrt(np.sum(np.where(is_valid, rows_std, 0.0)**2, axis=0))

    # NOTE: columns with no data in any selected row are NaN, not a measured 0
    CIE[n_rows == 0] = np.nan
    CIE_std[n_rows == 0] = np.nan

    if not return_info:
        return CIE, CIE_std

    # NOTE: rows with any data at all. A row can be entirely NaN (an empty mono
    #       bin), so "every row contributes" means every row that HAS data
    n_rows_with_data = int(np.any(is_valid, axis=1).sum())

    info = {
        'rows': (lo, hi),
        'mono_range': (Einc[lo], Einc[hi - 1]),
        'n_rows': n_rows,
        'n_selected': hi - lo,
        'n_rows_with_data': n_rows_with_data,
    }
    return CIE, CIE_std, info


# ====== JW added functions ==================

# JW edit 9/16/26: added correlation filtering function 
def filter_correlation(x, y, selection_mask=None,
                       trim_method='tukey', trim_k=3.0, trim_axes='both', trim_pct=0.5,
                       n_bins=20, sigma_thresh=3.0, min_pts_per_bin=20,
                       n_iter=2, verbose=True):
    """
    Filter shots that don't follow the linear I0/transmission correlation.

    Stage 1 — trim marginal extremes in value space (before any fit), so sparse
              high-leverage points at the ends can't tilt the line toward themselves.
    Stage 2 — fit a line to the survivors, then clip points whose residual is more
              than sigma_thresh standard deviations from the mean residual. The
              spread is computed once over ALL surviving points (no sub-binning),
              which is appropriate when the correlation is tight and the residual
              scatter is roughly uniform across the x-range.

    Parameters
    ----------
    x, y : full-length arrays (e.g. I0 and transmission, one entry per shot)
    selection_mask : full-length bool mask of shots to consider (mono & laser & ...).
                     None = use every shot.
    trim_method : 'tukey' (Q1-k*IQR .. Q3+k*IQR), 'percentile', or 'mad'
    trim_k : Tukey/MAD multiplier. 1.5 = standard, 3.0 = conservative.
    trim_axes : 'x', 'y', or 'both'
    sigma_thresh : clip beyond this many standard deviations of the residuals
    n_iter : max fit/clip iterations; exits early once the mask stops changing
    n_bins, min_pts_per_bin : accepted for call-signature compatibility, unused.

    Returns
    -------
    final_mask : full-length bool array, True = shot passed every stage
    fit        : scipy linregress result from the final iteration
    keep       : bool array over the *selected* subset (aligns with selected_idx)
    info       : dict of bounds and per-stage counts
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if selection_mask is None:
        selection_mask = np.ones(x.size, dtype=bool)
    selected_idx = np.flatnonzero(selection_mask)
    x_sel, y_sel = x[selection_mask], y[selection_mask]
    n_start = x_sel.size
    if n_start == 0:
        raise ValueError("selection_mask selects no shots")

    # ------------------------------------------------------------------
    # Stage 1: marginal (value-space) trimming — BEFORE the fit
    # ------------------------------------------------------------------
    axis_names = {'x': 'x', 'y': 'y', 'both': 'xy'}[trim_axes]
    V = np.asarray({'x': [x_sel], 'y': [y_sel], 'both': [x_sel, y_sel]}[trim_axes])

    if trim_method == 'tukey':
        q1, q3 = np.percentile(V, [25, 75], axis=1)
        iqr = q3 - q1
        lo, hi = q1 - trim_k * iqr, q3 + trim_k * iqr
    elif trim_method == 'percentile':
        lo, hi = np.percentile(V, [trim_pct, 100 - trim_pct], axis=1)
    elif trim_method == 'mad':
        med = np.median(V, axis=1, keepdims=True)
        spread = 1.4826 * np.median(np.abs(V - med), axis=1)
        med = med.ravel()
        lo, hi = med - trim_k * spread, med + trim_k * spread
    else:
        raise ValueError(f"unknown trim_method: {trim_method}")

    keep = np.all((V >= lo[:, None]) & (V <= hi[:, None]), axis=0)
    n_after_trim = int(keep.sum())
    if n_after_trim < 2:
        raise ValueError("trimming removed nearly everything — loosen trim_k")

    # ------------------------------------------------------------------
    # Stage 2: fit + global mean/std residual clipping
    # ------------------------------------------------------------------
    fit = None
    resid_mean = resid_std = np.nan

    for _ in range(n_iter):
        fit = linregress(x_sel[keep], y_sel[keep])
        resid = y_sel - (fit.slope * x_sel + fit.intercept)

        # spread estimated from the surviving points only
        resid_mean = np.mean(resid[keep])
        resid_std  = np.std(resid[keep])

        if resid_std == 0 or not np.isfinite(resid_std):
            break                                   # degenerate — keep what we have

        new_keep = keep & (np.abs(resid - resid_mean) < sigma_thresh * resid_std)

        if new_keep.sum() < 2:                      # too aggressive — don't apply
            break
        if np.array_equal(new_keep, keep):
            break                                   # converged
        keep = new_keep

    n_final = int(keep.sum())
    info = {
        'bounds': {ax: (float(l), float(h)) for ax, l, h in zip(axis_names, lo, hi)},
        'n_selected': n_start,
        'n_after_trim': n_after_trim,
        'n_final': n_final,
        'n_trimmed': n_start - n_after_trim,
        'n_clipped': n_after_trim - n_final,
        'pct_dropped': 100 * (n_start - n_final) / n_start,
        'resid_mean': float(resid_mean),
        'resid_std': float(resid_std),
    }

    if verbose:
        print(f"dropped {n_start - n_final} / {n_start} shots "
              f"({info['pct_dropped']:.2f}%)  "
              f"slope={fit.slope:.4f}  r²={fit.rvalue**2:.4f}  "
              f"resid_std={resid_std:.4g}")

    final_mask = np.zeros(x.size, dtype=bool)
    final_mask[selected_idx[keep]] = True
    return final_mask, fit, keep, info




def bin_scanvar(scanvar, bin_range, bin_config=None, scantype='fly',
                verbose=False):
    """
    One binning function for every scan variable.

    Example:
        bin_idx, centers = bin_scanvar(det.mono, [696.7, 709.9], ['num_bins', 33])
    
    example when used in combination with masks
        det.masks.add('mono_bin_idx', bin_idx, note='33 uniform bins, 696.7-709.9 eV')

        for i in range(len(centers)):
            sel = m.laser_on & m.i0_finite & m.i0_above_floor & (m.mono_bin_idx == i)
            datsum, unc, datmean, se = standard_error_IO(svls[sel])

    The index is a per-shot array like any other, so it composes with every mask
    by plain numpy.  Adding a cut never means touching the binner.

    Parameters
    ----------
    scanvar : (n_frames,) array
        Per-shot value of whatever is scanned: mono energy, delay, waveplate
        angle.  Nothing here is specific to mono.
    bin_range : (start, end)
        Range to bin over, in scanvar units.
    # -------------------------------------------------------------
    bin_config : ['num_bins', int] or ['bin_edges', [e0, e1, ...]]
        'num_bins'  : The number is the number of BINS; the n+1 edges are built here.
        'bin_edges' : explicit edge positions, strictly increasing.  
    # -------------------------------------------------------------
    scantype : 'fly' or 'step'

    Returns
    -------
    bin_idx : (n_frames,) int32
        Bin number per shot, -1 for out of range. 
    centers : (n_bins,) float
        Centre of each bin: the new scan axis.

    Notes
    -----
    FIXME: A step scan with real encoder jitter breaks the np.unique assumption --
    every shot is a distinct float and you get one bin per shot. maybe add it so that
    can pass in explicit setpoints through bin_edges??
    """
    v = np.asarray(scanvar, dtype=float).squeeze()
    if v.ndim != 1:
        raise ValueError(f'scanvar must be 1-D per shot, got shape {v.shape}')

    lo, hi = float(bin_range[0]), float(bin_range[1])
    if not hi > lo:
        raise ValueError(f'bin_range must be increasing, got ({lo}, {hi})')

    # ---------------- edges ----------------
    if scantype == 'step':
        # unique setpoints are the CENTRES; edges at the midpoints between
        # them, with the outer two mirroring the first and last half-step
        centers = np.unique(v[np.isfinite(v) & (v >= lo) & (v <= hi)])
        if centers.size == 0:
            raise ValueError(f'no finite scanvar values inside bin_range '
                             f'({lo}, {hi})')
        else:
            # mid are the points between the centers, and are the natural bin_edges
            mid = 0.5 * (centers[1:] + centers[:-1])
            # but mid does not account for the very first and very last edge, so we add those on
            edges = np.concatenate([[2 * centers[0] - mid[0]], mid,
                                    [2 * centers[-1] - mid[-1]]])

    elif scantype == 'fly':
        if bin_config is None or len(bin_config) < 2:
            raise ValueError("scantype='fly' needs bin_config, either "
                             "['num_bins', N] or ['bin_edges', [...]]")
        kind, value = bin_config[0], bin_config[1]

        if kind == 'num_bins':
            n = int(value)
            if n < 1:
                raise ValueError(f'num_bins must be >= 1, got {n}')
            edges = np.linspace(lo, hi, n + 1)   # n bins need n+1 edges

        elif kind == 'bin_edges':
            e = np.asarray(value, dtype=float).squeeze()
            if e.ndim != 1 or e.size < 2:
                raise ValueError('bin_edges needs a 1-D list of at least two '
                                 'edge positions')
            if np.any(np.diff(e) <= 0):
                raise ValueError('bin_edges must be strictly increasing')
            if e.size < 2:
                raise ValueError(
                    f'bin_range ({lo}, {hi}) contains fewer than two of the '
                    f'given edges, so no complete bin survives')
            edges = e

        else:
            raise ValueError(f"bin_config[0] must be 'num_bins' or "
                             f"'bin_edges', got {kind!r}")

        # create the bin centers based on the edges
        centers = 0.5 * (edges[:-1] + edges[1:])

    else:
        raise ValueError(f"scantype must be 'fly' or 'step', got {scantype!r}")

    # ---------------- assign ----------------
    n_bins = len(centers)

    # right=False -> half-open [edge_i, edge_i+1).  digitize returns 0 below
    # the first edge and len(edges) above the last (and for NaN), so shifting
    # down by one puts real bins at 0..n_bins-1 and everything else outside.
    bin_idx = np.digitize(v, edges, right=False) - 1

    # a shot exactly on the top edge would otherwise fall outside; the top
    # bin is the one interval closed at both ends
    bin_idx[v == edges[-1]] = n_bins - 1

    bin_idx[(bin_idx < 0) | (bin_idx >= n_bins)] = -1
    bin_idx = bin_idx.astype(np.int32)

    if verbose:
        counts = np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins)
        print(f'  bin_scanvar [{scantype}]: {n_bins} bins over '
              f'{edges[0]:.6g}..{edges[-1]:.6g}, '
              f'{int((bin_idx >= 0).sum())}/{v.size} shots binned')
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            print(f'    empty bins: {empty.tolist()}')

    return bin_idx, centers



##### JW edit 10/08/26: rewriting the average data function. this will handle the averaging of the images
def avg_data_count(runs: list, proc_folder: str = '', proc_path: str = ''):
    '''
    Count-weighted average of reduced SVLS runs (process_v2 schema).

    Parameters
    ----------
    runs : list of int
        Run numbers to combine.
    proc_folder : str
        Folder prefix of the reduced files.
    proc_path : str
        Placeholder, currently unused.

    Returns
    -------
    avg : dict
        Same keys as the reduced files, combined across runs, plus 'runs'.
    '''
    if len(runs) == 0:
        raise ValueError('avg_data_count: runs is empty')

    # =================================================================
    # 1. Use the first run to record which keys exist and their shapes
    # =================================================================

    with h5py.File(proc_folder + f'{runs[0]:04d}.h5', 'r') as first_file:
        reference_shapes = {key: first_file[key].shape for key in first_file.keys()}

    # =================================================================
    # 2. Load every run and stack each key along a new run axis
    # =================================================================

    # NOTE: per_run[key] collects one array per run; after the loop each
    #       entry becomes one array of shape (n_runs, *reference_shape)
    per_run = {key: [] for key in reference_shapes}

    for run in runs:
        with h5py.File(proc_folder + f'{run:04d}.h5', 'r') as run_file:

            # NOTE: a different key set means the file came from a different
            #       version of the reduction, so stop instead of mixing them
            if set(run_file.keys()) != set(reference_shapes):
                raise KeyError(f'Run {run}: keys differ from run {runs[0]}')

            for key, shape in reference_shapes.items():
                # NOTE: a shape mismatch (different binning or pixel count)
                #       would misalign bins, so it raises instead of skipping
                if run_file[key].shape != shape:
                    raise ValueError(f'Run {run} {key}: shape {run_file[key].shape} '
                                     f'!= {shape} in run {runs[0]}')
                
                per_run[key].append(np.asarray(run_file[key][()], dtype=float))

    stacked = {key: np.stack(arrays) for key, arrays in per_run.items()}

    # NOTE: every run must sit on the same mono bins, so scanvar is checked
    #       and copied, not averaged
    scanvar = stacked['scanvar'][0]
    if not np.allclose(stacked['scanvar'], scanvar, equal_nan=True):
        raise ValueError('scanvar differs between runs')

    avg = {'scanvar': scanvar, 'runs': np.asarray(runs)}

    # =================================================================
    # 3-5. For each (laser state, filter) group: weight by counts,
    #      then average the signal and combine its error
    # =================================================================

    for state in ('on', 'off'):
        for suffix in ('', '_filt'):

            counts = stacked[f'counts_{state}{suffix}']      # (n_runs, n_mono)
            i0_sum = stacked[f'i0_sum_{state}{suffix}']      # (n_runs, n_mono)

            mean_key = f'axis_svls_{state}_mean{suffix}'
            sos_key = f'axis_svls_{state}_sos{suffix}'

            # NOTE: the sos maps are raw summed ADU, so each run's map and
            #       error are divided by that run's summed I0 first, giving
            #       a per-run ratio. The summed I0 is treated as exact.
            #       The mean maps are already normalized per shot
            with np.errstate(divide='ignore', invalid='ignore'):
                sos_norm = stacked[sos_key] / i0_sum[:, :, np.newaxis]
                sos_norm_error = stacked[sos_key + '_std'] / i0_sum[:, :, np.newaxis]

            # NOTE: (signal, error) per output key, each (n_runs, n_mono, n_pix)
            signals = {
                mean_key: (stacked[mean_key], stacked[mean_key + '_std']),
                sos_key:  (sos_norm,         sos_norm_error),
            }

            for key, (values, errors) in signals.items():

                # ---- 3. weights from the counts ------------------------
                # NOTE: counts are per mono bin, (n_runs, n_mono). Copy each
                #       bin's count onto every pixel in that bin, so the
                #       counts have the same shape as the map:
                #       (n_runs, n_mono) -> (n_runs, n_mono, n_pix)
                n_pix = values.shape[2]
                counts_per_pixel = np.repeat(counts[:, :, np.newaxis], n_pix, axis=2)

                # NOTE: an element counts in a run only if it has shots and
                #       finite values. This drops empty bins and saturated
                #       (NaN) pixels in that run alone, instead of turning
                #       the whole average NaN
                is_valid = (counts_per_pixel > 0) & np.isfinite(values) & np.isfinite(errors)

                valid_counts = np.where(is_valid, counts_per_pixel, 0.0)
                valid_values = np.where(is_valid, values, 0.0)
                valid_errors = np.where(is_valid, errors, 0.0)

                # NOTE: weights = counts / sum of counts over runs. Where no
                #       run has data this is 0/0 = NaN, so the output is NaN
                with np.errstate(divide='ignore', invalid='ignore'):
                    weights = valid_counts / valid_counts.sum(axis=0)

                # ---- 4. weighted signal --------------------------------
                avg[key] = np.sum(weights * valid_values, axis=0)

                # ---- 5. error in quadrature ----------------------------
                # NOTE: sqrt(sum(weight^2 * error^2)). For n equal runs
                #       this gives error / sqrt(n)
                avg[key + '_std'] = np.sqrt(np.sum((weights * valid_errors)**2, axis=0))

            # NOTE: counts and summed I0 are totals, so they add up
            avg[f'counts_{state}{suffix}'] = counts.sum(axis=0)
            avg[f'i0_sum_{state}{suffix}'] = i0_sum.sum(axis=0)

        # ---- fzp: (n_mono,), unfiltered only, no error key -------------
        # NOTE: same count weighting, using the unfiltered counts
        fzp = stacked[f'fzp_{state}']                        # (n_runs, n_mono)
        counts = stacked[f'counts_{state}']                  # (n_runs, n_mono)

        is_valid = (counts > 0) & np.isfinite(fzp)
        valid_counts = np.where(is_valid, counts, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            weights = valid_counts / valid_counts.sum(axis=0)

        avg[f'fzp_{state}'] = np.sum(weights * np.where(is_valid, fzp, 0.0), axis=0)

    # =================================================================
    # 6. Return
    # =================================================================

    return avg

    
