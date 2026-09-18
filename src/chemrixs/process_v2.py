# from functools import cached_property
from pathlib import Path

# its time to upgrade to logging instead of print statements
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import h5py
import scipy.ndimage as ndimage
import scipy.stats as st
import os
import gc

# import yaml
# import contextlib
from chemrixs.utils import *
from chemrixs.smalldata import SmallData
class Reduced():
    """
    A  class for processing the incoming data.

    context manager magic

    Parameters
    ----------
    path : str or Path
        The filename for the data to analyse.

    bgpath : str or Path
        The filename for a darkscan that can be used for background subtraction.

    fyaml : str or Path
        yaml file with settings for processing.

    bgyaml : str or Path
        yaml file with settings for processing the BG data.

    save : bool
        Boolean determining if the processed data should be saved.

    scantype : str
        Optional, determining the type of scan. If not given, this will be
        determined by the data structure.

    norm : bool
    # NOTE: in our case, the data should always be normalized, whether that is through shot to shot or sum over sum
        Boolean determining if data should be normalised by I0 or not
        

    Notes
    -----
    To check if a particular attribute is available, use ``hasattr(obj, attr)``.
    Many attributes will not show up dynamically in an interpreter, because they are
    gotten dynamically from the file.

    TODO: add more error messages for potential failures
    """

    def __init__(self, path: str | Path | h5py.File, bgpath: str | Path | h5py.File, 
                 fyaml: str | Path, bgyaml: str | Path, save: bool = True, scantype: str = '',norm: bool = True):
        
        
        self.data = SmallData(path, fyaml, scantype)
        
        if scantype != '':
            self.scantype = scantype
        else:
            self.scantype = self.data.scantype
        
        if hasattr(self.data.integrating, 'axis_svls'):
            self.m    = self.data.integrating.axis_svls.m
        
        # the dictionary used to store the values to be set as attributes
        self.out = {}
        
        #load BG data or process it from smalldata and save
        if os.path.isfile(self.data.yaml['red_bg_path']): #load BG from preprocessed file
            #FIXME: how does preprocessed file look? make sure everything necessary is there
            self.bg_preproc = True
            self.bg = BG(self.data.yaml['red_bg_path'])
            
        else: #process BG
            self.bg_preproc = False
            self.bg = SmallData(bgpath,bgyaml,'static')


        self.process_int()
        self.save_out()
        if save ==True:
            self.save_dat()
            print('saved processed data')
            if self.bg_preproc == False:
                self.save_bg()
        
        # remove self.out to clear up storage
        delattr(self, 'out')
               
      
    def is_open(self) -> bool:
        """
        Function to check whether the file is open.
        """
        if not self.bg_preproc:
            return (self.data.is_open() & self.bg.is_open())
        else:
            return self.data.is_open()

    def __del__(self):
        """
        Function to close the file when the object is deleted.
        """
        if self.data.is_open:
            self.data.close()
        if self.bg.is_open:
            self.bg.close()

    def close(self):
        """
        Function to close the file.
        """
        if self.data.is_open:
            self.data.close()
        if self.bg.is_open:
            self.bg.close()

    def __exit__(self,*exc):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def open(self):  
        """
        Open the file.
        """
        if not self.data.is_open(): 
            self.data.open()
        if not self.bg_preproc and self.bg.is_open():
            self.bg.open()
        else:
            self.bg = h5py.File(self.data.yaml['red_bg_path'], "r")



    # right now we will only have the SVLS detector
    def process_int(self):
        '''
        parsing function to call the individual processing funcitons for area detectors
        '''
        if 'axis_svls' in self.data.yaml['int_detectors']:
            self.proc_svls()
            # I have this under axis_svls right now temporarily because it is part of the svls integrated detector
            self.proc_fzp()
        
        
  ###### JW edits 9/17/26: I am only making changes to the proc_svls because that is what we are working with in
                        #  this experiment. This is a combination of proc_svls and bin_int_det. also added correlation filters

    def proc_svls(self):
        
        # ========= Import + Set Up Necessary Configs ==================================
        background_type = self.data.yaml['int_detectors']['axis_svls']['backgroundtype']
        
        svls_cfg = self.data.yaml['svls']
        corr_cfg = self.data.yaml['corr_filter']
        offset_roi = svls_cfg['offset_roi']
        thresh_min = svls_cfg['threshold'][0]
        thresh_max = svls_cfg['threshold'][1]
        axis_svls = self.data.integrating.axis_svls
        svls_full = self.data.integrating.axis_svls.full_area
        
        I0  = axis_svls.I0
        fzp = axis_svls.fzp
        # ========== Processing the SVLS Raw Image =======================================
        
        # NOTE: JW 9/16/26, threshold or background subtract first?
        
        # NOTE: we want to keep the max values as nan instead of just 0 it as the I0 could be huge and blow our entire thing up
        # we will be able to handle the nan, they will just get dropped
        svls_full[svls_full < thresh_min] = 0
        svls_full[svls_full > thresh_max] = np.nan
        bg_full = self.bg.integrating.axis_svls.full_area
        #PROCESS BG
        if background_type == 'dark':
            # processing the background on the fly
            if self.bg_preproc == False:
                if bg_full.ndim == 2:
                    self.bg.svls = np.nanmean(bg_full[:,:],0)
                    svls_proc = svls_full - self.bg.svls[np.newaxis,:]
                elif bg_full.ndim == 3:
                    self.bg.svls = np.nanmean(bg_full[:,:,:],0)
                    svls_proc = svls_full - self.bg.svls[np.newaxis,:,:]
            
                delattr(self.bg.integrating.axis_svls,'full_area')
            elif self.bg_preproc == True:
                self.bg.svls = np.asarray(self.bg.file['axis_svls'])
                if svls_full.ndim == 2:
                    svls_background_subtracted = svls_full - self.bg.svls[np.newaxis,:]
                    svls_proc = np.flip(svls_background_subtracted,1)
                elif self.data.integrating.axis_svls.full_area.ndim == 3:
                    svls_proc = svls_full - self.bg.svls[np.newaxis,:,:]
               
        elif background_type == 'None':
            svls_proc = svls_full
            delattr(self.bg.integrating.axis_svls,'full_area')
            
        svls_norm = np.nansum(svls_proc, axis = 1) / I0
            
        # =========== Define the Masks ====================================
        laser_on = self.m.laser_on
        laser_off = self.m.laser_off
        good_i0 =  self.m.good_i0
        
        # apply the good_i0 to our laser on and off masks right away, so we are only working with good i0 shots
        laser_on &= good_i0
        laser_off &= good_i0
        
        if (self.scantype == 'mono' or self.scantype=='mono_fly'):
            mono_bin_idx_mask = self.m.mono_bin_idx
        if (self.scantype == 'delay' or self.scantype=='delay_fly'):
            # TODO: implement the scanvariable bin idx for the delay runs
            pass
        
        
        # ========== Apply the Binning based on kind of run ======================
        if self.scantype == 'static':
            logging.info('static run!')
            # TODO: implement something for the static runs
            pass
        
        # step scans
        if (self.scantype == 'mono' or self.scantype == 'delay'):
            logging.info('step scan!')
            # TODO: implement when using 
            pass
            
        if (self.scantype == 'mono_fly' or self.scantype == 'delay_fly'):
            logging.info('fly scan')
            
            if self.scantype == 'delay_fly':
                scanvar = self.data.integrating.axis_svls.delay
                # TODO: implment when using
                if self.data.yaml['TT_corr']['bool']:
                    tt_corr=np.loadtxt(f'proc/leading_edge_{self.data.run}.txt')
                    scanvar_on = (scanvar[laser_on]).squeeze() - (tt_corr-self.data.yaml['TT_corr']['offset'])*5*1e-15 #add pixel converted to s
                    scanvar_off = (scanvar[laser_off]).squeeze() - (np.nanmean(tt_corr)-self.data.yaml['TT_corr']['offset'])*5*1e-15 #no TT correction for laser off shots but we wanna make sure the scan axis still matches
                    logging.info('corrected delay')
                else:
                    scanvar_on = (scanvar[laser_on]).squeeze()
                    scanvar_off = (scanvar[laser_off]).squeeze()
                    logging.warning('uncorrected delay')
            
            if self.scantype == 'mono_fly':
                mono = self.data.integrating.axis_svls.mono
                
                # =========== Set Up the Output Arrays ===========================
                
                # helper function that will give us the correct return key from all the combinations
                def _key(*parts, prefix='axis_svls'):
                    return '_'.join([prefix] + [str(p) for p in parts if p])
                
                n_frames, n_pix = svls_proc.shape
                num_bins = len(self.data.integrating.axis_svls.mono_bin_centers)
                Z = lambda: np.zeros((num_bins, n_pix))
                z = lambda: np.zeros(num_bins)
                
                for state in ('on', 'off'):
                    for filt in ('', 'filt'):
                        for method in ('mean', 'sos'):
                            self.out[_key(state, method, filt)] = Z()
                            self.out[_key(state, method, filt, 'std')] = Z()
                        # keep track for both filtered and unfiltered counts for weighted averaging
                        self.out[_key(state, filt, prefix='i0_sum')] = z()
                        self.out[_key(state, filt, prefix='counts')] = z()

                # same mono bin centers for both laser on and laser off
                self.out['scanvar'] = axis_svls.mono_bin_centers
                        
                # ============ Apply Correlation Filtering ========================
                for i in range(num_bins):
                    in_bin = (mono_bin_idx_mask == i)
                    # stores the on and off masks with filtering combos
                    sel = {}

                    for state, laser in (('on', laser_on), ('off', laser_off)):
                        base = in_bin & laser
                        sel[(state, '')] = base
                        kept_mask, *_, info = filter_correlation(I0, fzp, base, verbose=False, sigma_thresh=corr_cfg['sigma_tresh'], trim_k=corr_cfg['trim_tresh'])
                        sel[(state, 'filt')] = kept_mask

                    for (state, filt), mask in sel.items():
                        n = int(mask.sum())
                        self.out[_key(state, filt, prefix='counts')][i] = n
                        if n == 0:
                            continue
                        
                        # ========== Apply Error Propogation with both filtered and unfiltered =================

                        _, _, datmean, se_mean = standard_error_IO(svls_norm[mask])
                        self.out[_key(state, 'mean', filt)][i] = datmean
                        self.out[_key(state, 'mean', filt, 'std')][i] = se_mean

                        datsum, unc_sum, _, _ = standard_error_IO(svls_proc[mask])
                        self.out[_key(state, 'sos', filt)][i] = datsum
                        self.out[_key(state, 'sos', filt, 'std')][i] = unc_sum
                        self.out[_key(state, filt, prefix='i0_sum')][i] = np.nansum(I0[mask])
                    
        # clear up some local memory
        del svls_full, svls_proc
        gc.collect()
    
    ##### JW edit: store the fzp as an attribute
    def proc_fzp(self):
        
        ###### ========== Import necessary information ====================
        fzp = self.data.integrating.axis_svls.fzp
        I0 = self.data.integrating.axis_svls.I0
        
        if (self.scantype == 'mono' or self.scantype=='mono_fly'):
            mono_bin_idx_mask = self.m.mono_bin_idx
            num_bins = len(self.data.integrating.axis_svls.mono_bin_centers)
        if (self.scantype == 'delay' or self.scantype=='delay_fly'):
            # TODO: implement the scanvariable bin idx for the delay runs
            pass
        
        # since our fzp is correlated with the i0, lets normalize it first
        fzp_norm = fzp / I0
        
        laser_on = self.m.laser_on
        laser_off = self.m.laser_off
        good_i0 =  self.m.good_i0
        
        # apply the good_i0 to our laser on and off masks right away, so we are only working with good i0 shots
        laser_on &= good_i0
        laser_off &= good_i0
        
        ###### ========= Bin the fzp with our scanvariable bins =============
        if (self.scantype == 'mono_fly' or self.scantype == 'delay_fly'):
            logging.info('fly scan')
            
        z = lambda: np.zeros(num_bins)    
        fzp_on, fzp_off = z(), z()
        
        for i in range(num_bins):
            on_mask, off_mask = (mono_bin_idx_mask == i & laser_on), (mono_bin_idx_mask == i & laser_off)
            fzp_on[i] = np.nanmean(fzp_norm[on_mask])
            fzp_off[i] = np.nanmean(fzp_norm[off_mask])
        
        self.out['fzp_on'] = fzp_on
        self.out['fzp_off'] = fzp_off
        

    ##### JW edits 9/18/26: changed the way we are saving the h5 files to match our self.out shape
    def save_dat(self):
        
        print('saving data')
        run = self.data.run
        output = h5py.File(f'./proc/Run{run:04d}.h5','w')
        for dat in self.out.keys():
            output.create_dataset(dat, dtype = 'f', data = self.out[dat])
            
        output.close()
        
    
    ##### JW edits 9/18/26: set the attributes for all fields in self.out
    def save_out(self):
        """ process all of the keys in self.out and save the corresponding data
            also clear up some RAM by deleting temporary stored arrays
        """
        for dat in self.out.keys():
            setattr(self, dat, self.out[dat])
        
        



    def save_bg(self):
        """
        Save processed BG
        """
        if len(self.data.yaml['red_bg_path'])==0:
            bgfname = f'./proc/BG_run{self.bg.run}.h5'
        else:
            bgfname = self.data.yaml['red_bg_path']
        output = h5py.File(bgfname,'w')
        if 'andor_dir' in self.data.yaml['int_detectors']:
            output.create_dataset('andor_dir',dtype='f',data=self.bg.dir)
        if 'andor_vls' in self.data.yaml['int_detectors']:
            output.create_dataset('andor_vls',dtype='f',data=self.bg.vls)
        if 'axis_svls' in self.data.yaml['int_detectors']:
            output.create_dataset('axis_svls',dtype='f',data=self.bg.svls)


class BG():
    """
    A  class for passing in previously processed BG data.

    Parameters
    ----------

    bgpath : str or Path
        The filename for a darkscan that can be used for background subtraction.
    """
    def __init__(self,bgpath: str | Path | h5py.File):
        self.file = h5py.File(bgpath,'r')
