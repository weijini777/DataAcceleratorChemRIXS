
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from functools import cached_property
from pathlib import Path
from chemrixs.utils import *
from chemrixs.process_v2 import Reduced

class Average():
    """
    A class to average over several runs. 

    Takes runnumber as input, if these runs have been processed before
    they will simply be averaged, otherwise the 'Reduced' class is called 
    to process these runs first.

    Parameters
    ----------
    runs : list
        containing the integers of the runs to average

    proc_path : str or Path
        The filename structure and location of the processed runs. 
        Should be complete by simply adding the '{run:04d}.h5'

    avg : str
        String determining the way of averaging: 

    output_path : str or Path
        The filename for the output of averaged data.
    
    
    raw_path : str or Path
        The filename structure for the data to analyse.
        Should be complete by simply adding the '{run:04d}.h5'

    bgpath : str or Path
        The filename for a darkscan that can be used for background subtraction.

    fyaml : str or Path
        yaml file with settings for processing of unprocessed runs.

    bgyaml : str or Path
        yaml file with settings for processing the BG data, needed for unprocessed runs.

    save : bool
        Boolean determining if the processed data should be saved where processing necessary.

    scantype : str
        Optional, determining the type of scan for processing. If not given, this will be
        determined by the data structure.

    norm : bool
        Boolean determining if data should be normalised by I0 or not when processing.
        
    """

#####ZY_edits - 051726 - add despeckle option to handle hot pixels.
    def __init__(self, runs: list | int, proc_path: str | Path, avg: str, output_path: str | Path,
                    raw_path: str | Path, bgpath: str | Path, fyaml: str | Path, bgyaml: str | Path, 
                    save: bool = True, scantype: str = '',norm: bool = True, emi_calib: bool = False,
                    despeckle: bool = True):
        
        self.runs = runs
        self.proc_path = proc_path
        self.avg = avg
        self.output_path = output_path

        try:
            with open(fyaml, 'r') as file:
                 self.yaml = yaml.safe_load(file)
        except FileNotFoundError as fe: 
            raise FileNotFoundError('Config yaml file not found - check filename') from fe


        if len(raw_path) is not None:
            self.raw_path = raw_path
            self.bgpath = bgpath
            self.fyaml = fyaml
            self.bgyaml = bgyaml
            self.scantype = scantype
            self.norm = norm
            self.despeckle = despeckle

        self.check_reduce()

        ##### JW edits 10/08/26: reordering the self.average order + reconfiguring the laser boolean
        self.average

        self.laser = 'counts_on' in self.average

        # if 'on' in list(self.average.keys())[0]:
        #     self.laser = True
        # elif 'off' in list(self.average.keys())[0]:
        #     self.laser = True
        # else:
        #     self.laser = False

        if self.despeckle:
            self.apply_despeckle()
        
        self.get_PFYs()

        if emi_calib:
            self.plot_svls2D_ET()

        if save:
            self.save_avg()


    ##### JW edits 10/13/26: re-reduce when the stored mono shift is out of date
    def check_reduce(self):
        '''
        Make sure every run has a reduced file binned with its current shift.

        A run is (re)reduced when its file is missing, when the file has no
        'mono_shift' (made before shifts existed), or when the stored shift
        differs from the yaml 'mono_shift' table.
        '''
        mono_shift_cfg = self.yaml.get('mono_shift')

        for run in self.runs:
            proc_path = self.proc_path + f'{run:04d}.h5'
            expected_shift = get_mono_shift(run, mono_shift_cfg)

            # NOTE: decide whether this run's file can be reused
            if not os.path.isfile(proc_path):
                reason = 'no reduced file'
            else:
                with h5py.File(proc_path, 'r') as f:
                    if 'mono_shift' not in f:
                        reason = 'file has no mono_shift (made before shifts)'
                    else:
                        stored_shift = float(f['mono_shift'][()])
                        # NOTE: files are saved as float32, so compare with a
                        #       tolerance instead of ==
                        if np.isclose(stored_shift, expected_shift, rtol=0, atol=1e-6):
                            reason = None
                        else:
                            reason = (f'stored shift {stored_shift:+.4f} eV, '
                                      f'yaml says {expected_shift:+.4f} eV')

            if reason is None:
                continue

            print(f'Run {run}: re-reducing ({reason})')
            fname = self.raw_path + f'{run:04d}.h5'
            tmp = Reduced(fname, self.bgpath, self.fyaml, self.bgyaml,
                          save=True, scantype=self.scantype, norm=self.norm)
            self.scantype = tmp.data.scantype

        
        return
    
    @cached_property
    def average(self):
        try:
            avg = avg_data_count(self.runs, self.proc_path)
        except:
            print('not count averaged')
            # avg = avg_data(self.runs, self.proc_path)

        #####ZY_edits - 060126 - use "sum over sum - sos"
        # if 'axis_svls_on_sumSV' in avg and 'axis_svls_on_sumI0' in avg:
        #     avg['axis_svls_on_norm']  = avg['axis_svls_on_sumSV']  / avg['axis_svls_on_sumI0'][:, None]
        #     avg['axis_svls_off_norm'] = avg['axis_svls_off_sumSV'] / avg['axis_svls_off_sumI0'][:, None]
        #     avg['PFY_on_sos']  = avg['axis_svls_on_sumSV'].sum(axis=1)  / avg['axis_svls_on_sumI0']
        #     avg['PFY_off_sos'] = avg['axis_svls_off_sumSV'].sum(axis=1) / avg['axis_svls_off_sumI0']
        # ##### ZY_edits - 061726 - add mono+TT 3d
        # if 'axis_svls_on_ET_sumSV' in avg and 'axis_svls_on_ET_sumI0' in avg:
        #     with np.errstate(invalid='ignore', divide='ignore'):
        #         avg['axis_svls_on_ET'] = (avg['axis_svls_on_ET_sumSV']
        #                                   / avg['axis_svls_on_ET_sumI0'][:, :, None])
        return avg

    ##### JW edits 10/08/26: PFY for mean and sos, filtered and unfiltered
    def get_PFYs(self):
    # NOTE: collapse each averaged map over the emission axis. Keys follow
    #       the averaged schema, e.g. 'axis_svls_on_sos_filt' and
    #       'axis_svls_on_sos_filt_std' -> 'PFY_on_sos_filt' and 'PFY_on_sos_filt_std'
        for suffix in ('', '_filt'):
            for method in ('mean', 'sos'):
                if self.laser:
                    for state in ('on', 'off'):
                        map_key = f'axis_svls_{state}_{method}{suffix}'
                        pfy_key = f'PFY_{state}_{method}{suffix}'
                        self.average[pfy_key], self.average[pfy_key + '_std'] = get_PFY(
                            self.average[map_key], self.average[map_key + '_std'])
                else:
                    map_key = f'axis_svls_{method}{suffix}'
                    pfy_key = f'PFY_{method}{suffix}'
                    self.average[pfy_key], self.average[pfy_key + '_std'] = get_PFY(
                        self.average[map_key], self.average[map_key + '_std'])
    
    ###### JW edit 10/08/26: Changing the emission so that all the runs only take in one calibration value
    def get_emi(self):
        if self.laser:
            # setup the emission axis to loop through
            px = np.arange(self.average['axis_svls_off_mean'].shape[1])
        else:
            px = np.arange(self.average['axis_svls_mean'].shape[1])

        # for i in np.arange(len(self.yaml['emi_calib'])):
        #     emi_config = np.asarray(self.yaml['emi_calib'][i])
        #     if np.logical_and(np.all(self.runs>emi_config[0]),np.all(self.runs<emi_config[1])):
          # use one calibration for all runs
        emi_calib = self.yaml['emi_calib_all']
        emi = emi_calib[0]*px+emi_calib[1]
        return emi

#####ZY_edits - 051726 - add despeckle method
##### JW edits 10/08/26: added despeckle method to the sos paths
    def apply_despeckle(self, kernel_size=3, n_sigma=10, verbose=True):
        """
        Replace anomalous pixels in the averaged 2D maps inside self.average.

        For each pixel, compute the median and MAD over a kernel_size x
        kernel_size neighborhood (excluding the center). If
        |value - local_median| exceeds n_sigma * 1.4826 * local_MAD, the pixel
        is flagged and replaced with the local median, and its error in the
        matching '_std' map is set to NaN.

        Applied to the mean and sum-over-sum maps, laser on and off,
        filtered and unfiltered (8 maps).
        """
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')

        # NOTE: positions of the neighbors relative to the center pixel,
        #       e.g. the 8 surrounding pixels for kernel_size = 3
        half = kernel_size // 2
        offsets = [(di, dj) for di in range(-half, half + 1)
                            for dj in range(-half, half + 1)
                            if not (di == 0 and dj == 0)]

        # NOTE: every averaged map: {on, off} x {mean, sos} x {unfiltered, filtered}
        map_keys = [f'axis_svls_{state}_{method}{suffix}'
                    for state in ('on', 'off')
                    for method in ('mean', 'sos')
                    for suffix in ('', '_filt')]
        
        # only looking at the maps that exist in self.average
        map_keys = [key for key in map_keys if key in self.average]

        self.despeckle_masks = {}
        total_flagged = 0

        for key in map_keys:
            Z = np.asarray(self.average[key]).astype(float)
            ny, nx = Z.shape

            # NOTE: build a (n_neighbors, ny, nx) stack: layer k holds each
            #       pixel's k-th neighbor. Edges are padded with NaN
            nbr_stack = np.full((len(offsets), ny, nx), np.nan)
            for k, (di, dj) in enumerate(offsets):
                src_i0 = max(0, -di); src_i1 = ny - max(0, di)
                src_j0 = max(0, -dj); src_j1 = nx - max(0, dj)
                dst_i0 = max(0, di);  dst_i1 = ny - max(0, -di)
                dst_j0 = max(0, dj);  dst_j1 = nx - max(0, -dj)
                nbr_stack[k, dst_i0:dst_i1, dst_j0:dst_j1] = \
                    Z[src_i0:src_i1, src_j0:src_j1]

            # NOTE: robust local spread; 1.4826 * MAD = sigma for Gaussian noise
            local_median = np.nanmedian(nbr_stack, axis=0)
            local_mad    = np.nanmedian(np.abs(nbr_stack - local_median[None]), axis=0)
            local_sigma  = 1.4826 * local_mad

            # NOTE: flat regions have MAD = 0; floor them so a tiny deviation
            #       there is not flagged as infinitely many sigmas
            floor = np.nanmedian(local_sigma[local_sigma > 0])
            if not np.isfinite(floor):
                floor = np.nanmedian(np.abs(Z)) * 1e-3
            local_sigma = np.where(local_sigma > 0, local_sigma, floor)

            deviation = np.abs(Z - local_median)
            mask = (deviation > n_sigma * local_sigma) & ~np.isnan(Z)

            # NOTE: replace flagged pixels with their local median
            Z_clean = Z.copy()
            Z_clean[mask] = local_median[mask]
            self.average[key] = Z_clean
            self.despeckle_masks[key] = mask

            # NOTE: the old error no longer matches the replaced value, so it
            #       is set to NaN in the matching '_std' map
            std_key = key + '_std'
            if std_key in self.average:
                std_map = np.asarray(self.average[std_key]).astype(float).copy()
                std_map[mask] = np.nan
                self.average[std_key] = std_map

            total_flagged += mask.sum()
            if verbose:
                print(f'  despeckle [{key}]: kernel={kernel_size}, '
                        f'n_sigma={n_sigma} -> flagged {mask.sum()} pixels '
                        f'({100*mask.sum()/mask.size:.3f}%)')

        if verbose:
            print(f'  despeckle TOTAL: {total_flagged} pixels across '
                    f'{len(map_keys)} map(s)')

    ##### JW edits 10/08/26: adding the sos paths + filtering capabilities
    def plot_svls2D(self, calibrated=True, savefig=False, transparent=True, figsize=(12,8),
                        scale=1, mode='sos', filter=False):
            """
            Plot the averaged 2D RIXS map (inc. energy vs emission).

            Parameters
            ----------
            mode : 'mean' or 'sos'
                Which map to plot: per-shot normalized mean, or sum-over-sum.
            filter : bool
                Use the correlation-filtered map ('_filt') instead of unfiltered.
            """
            if mode not in ('mean', 'sos'):
                raise ValueError(f"mode must be 'mean' or 'sos', got {mode!r}")

            # NOTE: builds the key for the chosen map, e.g. mode='sos', filter=True
            #       -> 'axis_svls_on_sos_filt'
            suffix = '_filt' if filter else ''
            mono = self.average['scanvar']
            label = f'{mode}{" filtered" if filter else ""}'

            if self.laser:
                off_map = self.average[f'axis_svls_off_{mode}{suffix}']
                on_map  = self.average[f'axis_svls_on_{mode}{suffix}']
                diff_map = on_map - off_map

                # NOTE: emission axis in eV from the calibration, or pixel index
                if calibrated:
                    emi = self.get_emi()
                    self.average['E_emi'] = emi
                else:
                    print('emission is not calibrated')
                    emi = np.arange(off_map.shape[1])

                # NOTE: symmetric color range for the difference, so zero is white
                ddatmax = np.nanmax(np.abs(diff_map))

                fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
                ax[0].pcolor(mono, emi, off_map.T, cmap='Reds',
                            vmin=0, vmax=np.nanmax(off_map)/scale)
                ax[1].pcolor(mono, emi, on_map.T, cmap='Reds',
                            vmin=0, vmax=np.nanmax(on_map)/scale)
                ax[2].pcolor(mono, emi, diff_map.T, cmap='bwr',
                            vmin=-ddatmax, vmax=ddatmax)

                for a in ax:
                    a.set_xlabel('inc. energy (eV)')
                ax[0].set_ylabel('emission (eV)' if calibrated else 'emission (pixel)')

                ax[0].set_title('laser off')
                ax[1].set_title('laser on')
                ax[2].set_title('difference')

                ax[0].set_xlim([np.nanmin(mono), np.nanmax(mono)])

                # NOTE: run range and map type go in the figure title, so the
                #       per-panel titles are not overwritten
                fig.suptitle(f'Runs {self.runs[0]} to {self.runs[-1]}   [{label}]')

            else:
                data_map = self.average[f'axis_svls_{mode}{suffix}']

                if calibrated:
                    emi = self.get_emi()
                    self.average['E_emi'] = emi
                else:
                    print('emission is not calibrated')
                    emi = np.arange(data_map.shape[1])

                fig, ax = plt.subplots(1, 1, figsize=figsize)
                ax.pcolor(mono, emi, data_map.T, cmap='Reds',
                        vmin=0, vmax=np.nanmax(data_map)/scale)

                ax.set_xlabel('inc. energy (eV)')
                ax.set_ylabel('emission (eV)' if calibrated else 'emission (pixel)')

                ax.set_xlim([np.nanmin(mono), np.nanmax(mono)])
                ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{label}]')

            if savefig:
                fig.savefig(f'figs/SVLS2D_{self.runs[0]}_{self.runs[-1]}_{mode}{suffix}.png',
                            transparent=transparent, dpi=200, bbox_inches='tight')

    # JW edits 10/08/26: adding the sos paths + filtering capabilities
    def plot_svls1D(self, savefig=False, transparent=True, figsize=(12,8), plot_err=True,
                    mode='sos', filter=False):
        """
        Plot the PFY (averaged map collapsed over emission) against the scan variable.

        Parameters
        ----------
        mode : 'mean' or 'sos'
            Which PFY to plot: per-shot normalized mean, or sum-over-sum.
        filter : bool
            Use the correlation-filtered PFY ('_filt') instead of unfiltered.
        plot_err : bool
            Draw error bands and the error panel.
        """
        if mode not in ('mean', 'sos'):
            raise ValueError(f"mode must be 'mean' or 'sos', got {mode!r}")

        # NOTE: builds the PFY key from get_PFYs, e.g. mode='sos', filter=True
        #       -> 'PFY_on_sos_filt' and 'PFY_on_sos_filt_std'
        suffix = '_filt' if filter else ''
        scanvar = self.average['scanvar']
        label = f'{mode}{" filtered" if filter else ""}'

        # NOTE: both mean and sos now carry propagated errors, so the bands
        #       are drawn for either mode
        if self.laser:
            PFY_on  = self.average[f'PFY_on_{mode}{suffix}']
            PFY_off = self.average[f'PFY_off_{mode}{suffix}']
            PFY_diff = PFY_on - PFY_off

            # NOTE: sharey is off so the difference and error panels get their
            #       own y range instead of being squashed to the spectra's scale
            fig, ax = plt.subplots(1, 3, sharex=True, figsize=figsize)
            ax[0].plot(scanvar, PFY_off, color='tab:blue',   label='laser off')
            ax[0].plot(scanvar, PFY_on,  color='tab:orange', label='laser on')
            ax[1].plot(scanvar, PFY_diff, color='tab:blue')
            ax[1].axhline(0, color='k', linestyle='--')

            if plot_err:
                PFY_on_std  = self.average[f'PFY_on_{mode}{suffix}_std']
                PFY_off_std = self.average[f'PFY_off_{mode}{suffix}_std']

                # NOTE: on and off are independent, so their errors add in
                #       quadrature for the difference
                PFY_diff_std = np.sqrt(PFY_on_std**2 + PFY_off_std**2)

                ax[0].fill_between(scanvar, PFY_off - PFY_off_std, PFY_off + PFY_off_std,
                                   alpha=0.2, color='tab:blue')
                ax[0].fill_between(scanvar, PFY_on - PFY_on_std, PFY_on + PFY_on_std,
                                   alpha=0.2, color='tab:orange')
                ax[1].fill_between(scanvar, PFY_diff - PFY_diff_std, PFY_diff + PFY_diff_std,
                                   alpha=0.2, color='tab:blue')

                ax[2].plot(scanvar, PFY_on_std,   color='tab:orange', label='on')
                ax[2].plot(scanvar, PFY_off_std,  color='tab:blue',   label='off')
                ax[2].plot(scanvar, PFY_diff_std, color='tab:green',  label='difference')
                ax[2].set_ylabel('error')
                ax[2].legend()

            ax[0].legend()
            ax[0].set_title('PFY')
            ax[1].set_title('on - off')
            ax[2].set_title('error')

            if self.scantype == 'mono_fly':
                for a in ax: a.set_xlabel('inc. energy (eV)')
            elif self.scantype == 'delay_fly':
                for a in ax: a.set_xlabel('delay (s)')

            ax[0].set_xlim([np.nanmin(scanvar), np.nanmax(scanvar)])
            fig.suptitle(f'Runs {self.runs[0]} to {self.runs[-1]}   [{label}]')

        else:
            PFY = self.average[f'PFY_{mode}{suffix}']

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(scanvar, PFY)

            if plot_err:
                PFY_std = self.average[f'PFY_{mode}{suffix}_std']
                ax.fill_between(scanvar, PFY - PFY_std, PFY + PFY_std, alpha=0.2)

            if self.scantype == 'mono_fly':
                ax.set_xlabel('inc. energy (eV)')
            elif self.scantype == 'delay_fly':
                ax.set_xlabel('delay (s)')

            ax.set_xlim([np.nanmin(scanvar), np.nanmax(scanvar)])
            ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{label}]')

        if savefig:
            fig.savefig(f'figs/SVLS1D_{self.runs[0]}_{self.runs[-1]}_{mode}{suffix}.png',
                        transparent=transparent, dpi=200, bbox_inches='tight')



    ##### JW edits 10/08/26: accomodating new keys + small bug fixes
    def plot_svls2D_ET(self, savefig=False, transparent=True, figsize=(12,8), scale=1,
                        ETstep=0.2, mode='sos', filter=False, ylim = None):
        """
        Plot the averaged RIXS map on an (inc. energy, energy transfer) grid.

        Also stores the rebinned maps in self.average for plot_CIE:
            E_trans_{mode}{_filt}                    (n_ET,) bin centers [eV]
            data_trans_{state}_{mode}{_filt}         (n_mono, n_ET)
            std_trans_{state}_{mode}{_filt}          (n_mono, n_ET)

        Parameters
        ----------
        mode : 'mean' or 'sos'
            Which map to rebin: per-shot normalized mean, or sum-over-sum.
        filter : bool
            Use the correlation-filtered map ('_filt') instead of unfiltered.
        ETstep : float
            Energy-transfer bin width [eV].
        """
        if mode not in ('mean', 'sos'):
            raise ValueError(f"mode must be 'mean' or 'sos', got {mode!r}")

        # NOTE: builds the key for the chosen map, e.g. mode='sos', filter=True
        #       -> 'axis_svls_on_sos_filt'
        suffix = '_filt' if filter else ''
        label = f'{mode}{" filtered" if filter else ""}'
        mono = self.average['scanvar']

        # NOTE: emission energy of each pixel; needed to get energy transfer
        self.average['E_emi'] = self.get_emi()
        emi = self.average['E_emi']

        if self.laser:
            # ---- rebin laser on and laser off onto the ET grid ---------------
            # NOTE: use only pixels that are valid in BOTH on and off. Each ET
            #       bin averages 2-3 pixels; if a pixel is NaN in one state only
            #       (e.g. flagged by despeckle in on but not off), the two bins
            #       average different pixels and the difference shows a fake
            #       signal (~2% of the peak for one pixel at the elastic line)
            key_on, key_off = f'axis_svls_on_{mode}{suffix}', f'axis_svls_off_{mode}{suffix}'
            in_both = (np.isfinite(self.average[key_on]) & np.isfinite(self.average[key_on + '_std']) &
                        np.isfinite(self.average[key_off]) & np.isfinite(self.average[key_off + '_std']))

            for state in ('on', 'off'):
                map_key = f'axis_svls_{state}_{mode}{suffix}'

                # NOTE: each state uses its own error map, restricted to the
                #       pixels both states share
                _, E_trans, data_trans, std_trans = emi2ET(
                    mono, emi,
                    np.where(in_both, self.average[map_key], np.nan),
                    np.where(in_both, self.average[map_key + '_std'], np.nan),
                    ETstep)

                # NOTE: stored per state, mode and filter so plotting another
                #       mode does not overwrite this one
                self.average[f'data_trans_{state}_{mode}{suffix}'] = data_trans
                self.average[f'std_trans_{state}_{mode}{suffix}'] = std_trans

            # NOTE: the ET axis is stored next to the maps it belongs to. On and
            #       off share it, but a later call with a different ETstep must
            #       not overwrite the axis of maps made with this ETstep
            self.average[f'E_trans_{mode}{suffix}'] = E_trans

            data_trans_off = self.average[f'data_trans_off_{mode}{suffix}']
            data_trans_on  = self.average[f'data_trans_on_{mode}{suffix}']
            data_trans_diff = data_trans_on - data_trans_off

            # NOTE: on and off share one color scale so their intensities can
            #       be compared directly between the two panels
            shared_vmax = np.nanmax([np.nanmax(data_trans_off), np.nanmax(data_trans_on)]) / scale

            # NOTE: symmetric color range for the difference, from the ET maps
            #       actually plotted, so zero is white
            ddatmax = np.nanmax(np.abs(data_trans_diff))

            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
            ax[0].pcolor(mono, E_trans, data_trans_off.T, cmap='Reds',
                            vmin=0, vmax=shared_vmax, shading='auto')
            ax[1].pcolor(mono, E_trans, data_trans_on.T, cmap='Reds',
                            vmin=0, vmax=shared_vmax, shading='auto')
            ax[2].pcolor(mono, E_trans, data_trans_diff.T, cmap='bwr',
                            vmin=-ddatmax, vmax=ddatmax, shading='auto')

            for a in ax:
                a.set_xlabel('inc. energy (eV)')
            ax[0].set_ylabel('energy transfer (eV)')

            ax[0].set_title('laser off')
            ax[1].set_title('laser on')
            ax[2].set_title('difference')

            ax[0].set_xlim([np.nanmin(mono), np.nanmax(mono)])
            if ylim:
                ax[0].set_ylim(ylim)
                ax[1].set_ylim(ylim)
                ax[2].set_ylim(ylim)

            # NOTE: run range and map type in the figure title, so the panel
            #       titles are not overwritten
            fig.suptitle(f'Runs {self.runs[0]} to {self.runs[-1]}   [{label}]')

        else:
            map_key = f'axis_svls_{mode}{suffix}'

            _, E_trans, data_trans, std_trans = emi2ET(
                mono, emi, self.average[map_key], self.average[map_key + '_std'], ETstep)

            self.average[f'E_trans_{mode}{suffix}'] = E_trans
            self.average[f'data_trans_{mode}{suffix}'] = data_trans
            self.average[f'std_trans_{mode}{suffix}'] = std_trans

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.pcolor(mono, E_trans, data_trans.T, cmap='Reds',
                        vmin=0, vmax=np.nanmax(data_trans)/scale, shading='auto')

            ax.set_xlabel('inc. energy (eV)')
            ax.set_ylabel('energy transfer (eV)')
            ax.set_xlim([np.nanmin(mono), np.nanmax(mono)])
            ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{label}]')

        if savefig:
            fig.savefig(f'figs/SVLS2D_ET_{self.runs[0]}_{self.runs[-1]}_{mode}{suffix}.png',
                        transparent=transparent, dpi=200, bbox_inches='tight')
            

    def elastic_calibrate_from_two_points(self,
        mono,
        rixs_map_full,
        p1, p2,
        width_pixels=20,
        plot_on=True
        ):
        """
        Calibrate using a parallelogram ROI defined by two points and a fixed width.

        Parameters
        ----------
        mono : (nx,) array
            X-axis values (energy). Assumed monotonic; used on FIRST axis of rixs_map_full.
        rixs_map_full : (nx, ny) array
            2D map with first index along mono (x), second index = pixel (y).
        p1, p2 : tuple
            Two points defining the *center line* of the ROI, as (x_value_in_mono_units, y_pixel_index).
            Example: p1 = (x1, y1_pixel), p2 = (x2, y2_pixel).
        width_pixels : int or float
            Total vertical thickness of the parallelogram in pixels (constant in y).
            Top/Bottom are ±width/2 around the center line y(x) in pixel space.
        plot_on : bool
            If True, show a diagnostic plot.

        Returns
        -------
        calibrated_axis : (ny,) array
            mono ≈ a*pixel + b evaluated over all pixels.
        fit : (a, b)
            Linear fit parameters such that mono ≈ a*pixel + b.
        details : dict
            Extra info: sampled (x_mono, y_pix) of maxima, indices, etc.
        """
        mono = np.asarray(mono)
        Z = np.asarray(rixs_map_full)
        nx, ny = Z.shape

        # --- Convert the two points to (ix, iy) using mono -> x index
        x1, y1 = p1
        x2, y2 = p2
        ind_x1 = find_nearest(mono, x1)
        ind_x2 = find_nearest(mono, x2)
        px_y1 = int(round(y1))
        px_y2 = int(round(y2))
        # --- Center line
        m =(px_y2-px_y1)/(mono[ind_x2]-mono[ind_x1])
        c = px_y1-m*mono[ind_x1]

        half_w = 0.5 * float(width_pixels)

        # --- Determine x-span (in indices) to scan
        ix_start = min(ind_x1, ind_x2)
        ix_end   = max(ind_x1, ind_x2)

        x_samples = []
        ypix_at_max = []
        used_ix = []

        # --- Scan each x index within the span; search along y within top/bottom bounds
        for i in range(ix_start, ix_end + 1):
            x_here = mono[i]
            y_center = m * x_here + c
            y_lo = int(np.floor(y_center - half_w))
            y_hi = int(np.ceil (y_center + half_w))
            if y_lo > y_hi:
                y_lo, y_hi = y_hi, y_lo
            y_lo = max(0, min(ny - 1, y_lo))
            y_hi = max(0, min(ny - 1, y_hi))
            if y_hi < y_lo:
                continue

            col_slice = Z[i, y_lo:y_hi + 1]
            if col_slice.size == 0 or np.all(np.isnan(col_slice)):
                continue

            rel = int(np.nanargmax(col_slice))
            iy_max = y_lo + rel

            x_samples.append(x_here)
            ypix_at_max.append(iy_max)
            used_ix.append(i)

        x_samples = np.asarray(x_samples)
        ypix_at_max = np.asarray(ypix_at_max)

        if x_samples.size < 2:
            raise RuntimeError("Not enough maxima found within the parallelogram to fit a line.")

        # --- Fit mono ≈ a * pixel + b  (same calibration as your original)
        a, b = np.polyfit(ypix_at_max, x_samples, 1)
        calibrated_axis = np.polyval([a, b], np.arange(ny))

        # --- Optional diagnostic plot
        if plot_on:
            plt.figure(figsize=(6, 4))
            # Display Z with axes: x = mono (horizontal), y = pixel (vertical)
            plt.pcolormesh(mono,range(0,Z.shape[1]),Z.T,cmap='Reds',vmin=0,vmax=np.max(Z))
            plt.colorbar(label='Intensity')

            # Draw the two points
            plt.scatter([mono[ind_x1], mono[ind_x2]], [px_y1, px_y2],
                        s=60, c='white', edgecolor='k')

            # Draw center and top/bottom edges
            x_line = np.linspace(mono[ix_start], mono[ix_end], 200)
            y_center = m * x_line + c
            y_top = y_center + half_w
            y_bot = y_center - half_w
            plt.plot(x_line, y_center, 'w--', lw=1.5, color='k',label='initial guess')
            plt.plot(x_line, y_top,    'w-',  lw=1.0, alpha=0.8,color='k')
            plt.plot(x_line, y_bot,    'w-',  lw=1.0, alpha=0.8,color='k')

            # Scatter maxima used for the fit
            plt.scatter(x_samples, ypix_at_max, s=20, c='yellow', edgecolor='k')

            # Plot fitted calibration line (x vs pixel)
            ypix = np.arange(ny)
            plt.plot(np.polyval([a, b], ypix), ypix, 'r-', lw=2,
                     label=f'fit: mono = {a:.6g} * pixel + {b:.6g}')
            plt.xlim(mono[0],mono[-1])
            plt.xlabel('Energy (mono)')
            plt.ylabel('Pixel')
            plt.title(f'Elastic Calibration via Parallelogram (width={width_pixels} px)')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        details = {
            "x_samples": x_samples,
            "ypix_at_max": ypix_at_max,
            "used_x_indices": np.array(used_ix, dtype=int),
            "center_line_m_c": (m, c),
            "width_pixels": width_pixels,
        }
        return calibrated_axis, (a, b), details
    
    
    ### JW_comments 10/01/26: this function is never called, so the live elastic calibration is never used.
    def emi_calibration(self, p1, p2, width_pixels = 10, plot_on = True, use = 'on'):
        if use == 'on':
            mono = self.average['scanvar_on']
            rixs = self.average['axis_svls_on_mean']
        elif use == 'off':
            mono = self.average['scanvar_off']
            rixs = self.average['axis_svls_off_mean']
        else:
            mono = self.average['scanvar']
            rixs = self.average['axis_svls_mean']

        calibrated_axis, (a, b), details = self.elastic_calibrate_from_two_points(
                                    mono,
                                    rixs,
                                    p1, p2,
                                    width_pixels=width_pixels,
                                    plot_on=plot_on
                                 )
        self.average['E_emi'] = calibrated_axis
        return calibrated_axis, (a, b), details

    def save_avg(self):
        runs = self.runs
        output = h5py.File(f'./avg/Run{runs[0]:04d}to{runs[-1]:04d}.h5','w')
        fname = f'./avg/Run{runs[0]:04d}to{runs[-1]:04d}.h5'
        print(f'saving data in {fname}')
        keys = self.average.keys()
        print(keys)

        for key in keys:
            output.create_dataset(key,dtype='f',data=self.average[key])

        output.close()



    ##### JW edits 10/08/26: adding the sos paths + filtering capabilites
    def plot_CIE(self, inc_energy, width=2, mode='sos', filter=False, savefig=False,
                    transparent=True, figsize=(12,8), xlim=None):
            '''
            Plot a constant incident energy (CIE) cut from the energy-transfer maps.

            Run plot_svls2D_ET with the same mode and filter first; it creates the
            maps this reads.

            Parameters
            ----------
            inc_energy : float
                Center incident energy of the cut [eV].
            width : int
                Number of mono BINS on each side of the center to sum
                (2*width + 1 bins in total).
            mode : 'mean' or 'sos'
            filter : bool
                Use the correlation-filtered maps ('_filt').
            savefig, transparent, figsize : figure options
            xlim : (min, max) or None
                Energy-transfer range to show. None shows the range covered by
                every summed row.
            '''
            if mode not in ('mean', 'sos'):
                raise ValueError(f"mode must be 'mean' or 'sos', got {mode!r}")

            # NOTE: keys written by plot_svls2D_ET for this mode and filter
            suffix = '_filt' if filter else ''
            label = f'{mode}{" filtered" if filter else ""}'

            E_trans_key = f'E_trans_{mode}{suffix}'
            if E_trans_key not in self.average:
                raise KeyError(f'{E_trans_key} not found; run plot_svls2D_ET(mode={mode!r}, '
                            f'filter={filter}) first')

            mono = self.average['scanvar']
            E_trans = self.average[E_trans_key]

            # NOTE: default x range = columns where every summed row that has data
            #       contributes, so the ragged ends of the ET range are not shown.
            #       Falls back to any column with data, then to the full axis, so an
            #       empty mono bin in the window cannot leave the range empty
            def default_xlim(*infos):
                full = np.ones(len(E_trans), dtype=bool)
                any_data = np.ones(len(E_trans), dtype=bool)
                for info in infos:
                    full &= (info['n_rows'] == info['n_rows_with_data']) & (info['n_rows'] > 0)
                    any_data &= info['n_rows'] > 0
                for columns in (full, any_data):
                    if columns.any():
                        return [np.min(E_trans[columns]), np.max(E_trans[columns])]
                return [np.min(E_trans), np.max(E_trans)]

            if self.laser:
                # ---- cut laser off and laser on --------------------------------
                data_off = self.average[f'data_trans_off_{mode}{suffix}']
                std_off  = self.average[f'std_trans_off_{mode}{suffix}']
                data_on  = self.average[f'data_trans_on_{mode}{suffix}']
                std_on   = self.average[f'std_trans_on_{mode}{suffix}']

                # NOTE: use only cells that have data in BOTH on and off. Otherwise
                #       a mono bin with on shots but no off shots is summed into on
                #       only, and the difference shows a whole extra bin of signal
                #       that is not a pump effect
                in_both = (np.isfinite(data_on) & np.isfinite(std_on) &
                        np.isfinite(data_off) & np.isfinite(std_off))

                CIE_off, CIE_off_std, info_off = get_CIE(
                    inc_energy, width, mono,
                    np.where(in_both, data_off, np.nan),
                    np.where(in_both, std_off, np.nan), return_info=True)
                CIE_on, CIE_on_std, info_on = get_CIE(
                    inc_energy, width, mono,
                    np.where(in_both, data_on, np.nan),
                    np.where(in_both, std_on, np.nan), return_info=True)

                # NOTE: on and off are independent, so the error of the difference
                #       is the quadrature sum (no factor of 1/2)
                CIE_diff = CIE_on - CIE_off
                CIE_diff_std = np.sqrt(CIE_on_std**2 + CIE_off_std**2)

                if xlim is None:
                    xlim = default_xlim(info_on, info_off)

                fig, ax = plt.subplots(1, 2, sharex=True, figsize=figsize)

                ax[0].plot(E_trans, CIE_off, label='laser off', color='tab:blue')
                ax[0].fill_between(E_trans, CIE_off - CIE_off_std, CIE_off + CIE_off_std,
                                color='tab:blue', alpha=0.2)
                ax[0].plot(E_trans, CIE_on, label='laser on', color='tab:orange')
                ax[0].fill_between(E_trans, CIE_on - CIE_on_std, CIE_on + CIE_on_std,
                                color='tab:orange', alpha=0.2)
                ax[0].legend()
                ax[0].set_xlabel('energy transfer (eV)')
                ax[0].set_ylabel('intensity (arb. u.)')
                ax[0].set_title('CIE')

                ax[1].plot(E_trans, CIE_diff, color='tab:blue')
                ax[1].fill_between(E_trans, CIE_diff - CIE_diff_std, CIE_diff + CIE_diff_std,
                                color='tab:blue', alpha=0.2)
                ax[1].axhline(0, color='k', linestyle='--')
                ax[1].set_xlabel('energy transfer (eV)')
                ax[1].set_ylabel('on - off')
                ax[1].set_title('difference')

                ax[0].set_xlim(xlim)

                # NOTE: on and off now use the same cells, so either info
                #       describes both
                info = info_on

            else:
                CIE, CIE_std, info = get_CIE(
                    inc_energy, width, mono,
                    self.average[f'data_trans_{mode}{suffix}'],
                    self.average[f'std_trans_{mode}{suffix}'], return_info=True)

                # NOTE: unchanged from before: this branch keeps the cut
                self.average['CIE'] = CIE

                if xlim is None:
                    xlim = default_xlim(info)

                fig, ax = plt.subplots(1, 1, figsize=figsize)
                ax.plot(E_trans, CIE)
                ax.fill_between(E_trans, CIE - CIE_std, CIE + CIE_std, alpha=0.2)
                ax.set_xlabel('energy transfer (eV)')
                ax.set_ylabel('intensity (arb. u.)')
                ax.set_xlim(xlim)

            # NOTE: the title gives the incident energies actually summed, which
            #       can differ from inc_energy at the edge of the scan
            #       It also shows how many of the summed bins have data, since a
            #       clipped or partly empty window sums fewer bins than 2*width + 1
            mono_lo, mono_hi = info['mono_range']
            fig.suptitle(f'Runs {self.runs[0]} to {self.runs[-1]}   '
                        f'CIE {mono_lo:.2f}-{mono_hi:.2f} eV '
                        f'({info["n_rows_with_data"]} of {2*int(width)+1} bins with data)   [{label}]')

            if savefig:
                fig.savefig(f'figs/SVLS_CIE_{self.runs[0]}_{self.runs[-1]}_'
                            f'{inc_energy:.2f}eV_{mode}{suffix}.png',
                            transparent=transparent, dpi=200, bbox_inches='tight')
       

# cutE = 528.5
# inds = np.argmin(np.abs(np.asarray(OK_mono['mono_off'])-cutE))
# cutE = 528.3
# ind_comp = np.argmin(np.abs(np.asarray(dat_o_es['arr_0'])-(cutE-8)))
# f = 1e4

# ax[0].plot(np.asarray(OK_mono['E_trans_off']), np.nansum(np.asarray(OK_mono['data_trans_off'])[inds-2:inds+2,:],axis=0),label='laser off, 522.8eV')
# ax[1].plot(np.asarray(OK_mono['E_trans_off']), np.nansum(np.asarray(OK_mono['data_trans_on'])[inds-2:inds+2,:],axis=0),label='laser on')

    ##### plotting method for 3D data
    def plot_mono_at_delay(self, tau, emi_lo=None, emi_hi=None, diff=False, figsize=(8,4)):
            """Mono spectrum (PFY vs energy) at the nearest absolute-time bin to `tau` (s)."""
            ET = self.average['axis_svls_on_ET']                      # (nT, nE, emission)
            tb, eb = self.yaml['time_bins'], self.yaml['bins']
            tcen = 0.5*(np.linspace(float(tb[1][0]),float(tb[1][1]),int(tb[1][2]))[:-1]
                        + np.linspace(float(tb[1][0]),float(tb[1][1]),int(tb[1][2]))[1:])
            ecen = 0.5*(np.linspace(float(eb[1][0]),float(eb[1][1]),int(eb[1][2]))[:-1]
                        + np.linspace(float(eb[1][0]),float(eb[1][1]),int(eb[1][2]))[1:])
            it = int(np.argmin(np.abs(tcen - tau)))
            sl = slice(None) if emi_lo is None else slice(emi_lo, emi_hi+1)

            pfy_on = np.nansum(ET[it][:, sl], axis=1)
            plt.figure(figsize=figsize)
            if diff:
                pfy_off = np.nansum(self.average['axis_svls_off_norm'][:, sl], axis=1)
                plt.plot(ecen, pfy_on - pfy_off, 'k'); plt.ylabel('on − off PFY (SoS)')
            else:
                plt.plot(ecen, pfy_on); plt.ylabel('PFY (SoS)')
            plt.xlabel('inc. energy (eV)')
            plt.title(f'mono spectrum at τ ≈ {tcen[it]*1e15:.0f} fs')
            plt.tight_layout(); plt.show()
            return ecen, pfy_on
    

##### JW edits 10/13/26: per-run ground state spectra, a visual check of the runs
    def plot_runs_ground_state(self, mode='sos', filter=False, savefig=False,
                               transparent=True, figsize=(10,6), xlim=None):
        """
        Overlay the laser-off PFY of every run in this average, each normalized
        to its own maximum, read from the existing reduced files.

        Parameters
        ----------
        mode : 'mean' or 'sos'
        filter : bool
            Use the correlation-filtered maps ('_filt').
        xlim : (min, max) or None
            Incident energy range to show.
        """
        if mode not in ('mean', 'sos'):
            raise ValueError(f"mode must be 'mean' or 'sos', got {mode!r}")
        suffix = '_filt' if filter else ''
        label = f'{mode}{" filtered" if filter else ""}'

        # NOTE: one color per run, spread along a colormap in run order
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(self.runs)))

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for run, color in zip(self.runs, colors):
            # NOTE: same location Reduced.save_dat writes to
            with h5py.File(f'./proc/Run{run:04d}.h5', 'r') as f:
                scanvar = np.asarray(f['scanvar'][()], dtype=float)
                off_map = np.asarray(f[f'axis_svls_off_{mode}{suffix}'][()], dtype=float)
                counts  = np.asarray(f[f'counts_off{suffix}'][()], dtype=float)
                i0_sum  = np.asarray(f[f'i0_sum_off{suffix}'][()], dtype=float)

            # NOTE: sos maps in the reduced file are raw summed ADU, so divide
            #       by that bin's summed I0 first, as avg_data_count does
            with np.errstate(divide='ignore', invalid='ignore'):
                if mode == 'sos':
                    off_map = off_map / i0_sum[:, np.newaxis]
                pfy = np.nansum(off_map, axis=1)

            # NOTE: empty mono bins are stored as 0; make them NaN so they are
            #       not drawn as real zero intensity
            pfy[counts == 0] = np.nan

            # NOTE: each run scaled to its own maximum, so differences in total
            #       intensity between runs do not hide the energy position
            ax.plot(scanvar, pfy / np.nanmax(pfy), color=color, alpha=0.7, lw=1,
                    label=f'run {run}')

        ax.set_xlabel('inc. energy (eV)')
        ax.set_ylabel('laser-off PFY (normalized)')
        ax.legend(fontsize=7, ncol=2)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}: ground state per run   [{label}]')

        if savefig:
            fig.savefig(f'figs/ground_state_runs_{self.runs[0]}_{self.runs[-1]}_'
                        f'{mode}{suffix}.png',
                        transparent=transparent, dpi=200, bbox_inches='tight')