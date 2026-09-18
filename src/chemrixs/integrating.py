
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import yaml
from chemrixs.detector import Detector
from chemrixs.utils import *
import scipy.stats as st
from chemrixs.masks import Masks


class Integrating():

    
    """
    Class that is called by the small data class to load data from the integrating detectors.

    This class loads data through the Detector class as cached property -
    That way the data is only loaded once it is actually called. This makes it much faster to perform
    small tasks where simple metadata is required, rather than reading in the whole
    header.

    In this class the detectors to be loaded and the respective keys are being defind.

    Anything that is read in is stored in memory so the second access is much faster.
    However, the memory can be released simply by deleting the attribute (it can be
    accessed again, and the data will be re-read).


    Parameters:
    -----------
    intgrp: h5py.Group
        group containing the integrating data defined in the smalldata class

    Attributes
    ----------
    __getattr__:

    summing_channels:
        reducing fim and apd from multichannel waveforms to integrated value for each shot
        
        ##### JW edit 9/17/26: although fzp is not multichannel, we are integrating over it for
                               each shot, so i will just add it here. 
                               the FIMS, APDs, FZP all seem like "supporting detectors" to our main
                               SVLS signal, so grouping them in this context seems ok

    countmask: 
        returns integrated data where frames with faulty number of counts have been filtered out



    Notes:
    ------
    TODO: detectors and keys should be moved to yaml file, then loaded here
    Detector and key names may need to be updated if small data structure changes

    """

    def __init__(self, intgrp: h5py.Group, fyaml: dict, epics: dict, scantype: str = '', run: int = 1):
        self.scantype = scantype
        self.yaml = fyaml
        self.epics = epics
        self.run = run

        print(run)
        
        for detector, det_spec_dict in self.yaml['int_detectors'].items():
            if detector in intgrp.keys():
                #Creating a different class for each detector to avoid printing of attributes on all of them
                detobj = type(det_spec_dict["clsname"], (Detector,), {})
                #Create an attribute for each detector which will in turn will have attributes for the keys specified in the dictionary
                setattr(
                    self,
                    detector,
                    detobj(
                        intgrp[detector],
                        self.yaml[det_spec_dict['attrdict']],
                        useDask=det_spec_dict['useDask'],
                        chunks=det_spec_dict['chunks']
                    )
                )
        # print(sum(self.andor_vls.count_mask))
        ##### ZY_edits - add timestamp
        self.top_timestamp = (np.asarray(intgrp['timestamp'])
                              if self.yaml.get('timestamp_sort', False) and 'timestamp' in intgrp
                              else None)

        self.get_scanvar(intgrp)
        self.countmask()
        self.summing_channels()
        
        ##### JW edits 9/16/26: add the mask as an attribute to be passed down to the next levels for processing
        # this has to happen after countmask and summing_channels
        # we are only doing this processing right now for SVLS and non dark runs
        for detector in self.yaml['int_detectors']: 
            if (detector == 'axis_svls' and self.scantype != 'static'):
                det = getattr(self,detector)
                setattr(det, 'm', self.process_masks())
        #############################################################

    def __getattr__(self, name):
        #This will create an error if detector is not in the small data file
        if name in self.yaml['int_detectors']:
            raise KeyError(f'{name} is not in this file')       
        return super().__getattribute__(name)
    
    def summing_channels(self):

        #FIXME: case for prereduced fims and APDs
        '''
        Calling function to integrate the waveform for all waveform detectors
        
        These detectors include APDs and fims.
        '''

        for detector in self.yaml['int_detectors']: 
            det = getattr(self,detector)
            print(det)
            sum_channels(det, self.yaml)
            
            #'clearing cache'
            for channel in self.yaml['channels_to_integrate']:
                try:
                    delattr(getattr(self, detector),channel)
                except:
                    'did not delete non existing channel'
            #combine fim1 and fim0 to I0
            # NOTE: although 'fim0' is deleted, there is still a 'fim_0' 
            # under the integrating object that stores the raw waveform,
            # it is taking up a lot of data, need to find way to delete it properly
            if (hasattr(det,'fim_0') and hasattr(det,'fim_1')):
                I0 = det.fim0.copy()+det.fim1.copy()
                setattr(det,'I0',I0)
                delattr(det,'fim0')
                delattr(det,'fim1')
            elif hasattr(det,'fim_0'):
                I0 = det.fim0.copy()
                setattr(det,'I0',I0)
                delattr(det,'fim0')
            elif hasattr(det,'fim1'):
                # JW edit 9/16/26: small typo of fim0 when should be fim1
                I0 = det.fim1.copy()
                setattr(det,'I0',I0)
                delattr(det,'fim1')
                
            # JW edits 9/17/26: add fzp processing
            if hasattr(det, 'fzp'): 
                fzp = getattr(det, 'fzp')
                fzp_cfg = self.yaml['fzp']
                bg_0 = fzp[:, fzp_cfg['offset_roi'][0]:fzp_cfg['offset_roi'][1]].mean(axis=1)
                fzp_bgsub = fzp - bg_0[:, np.newaxis]
                fzp_summed = fzp_bgsub[:, fzp_cfg['roi'][0]:fzp_cfg['roi'][1]].sum(axis=1).flatten()
                
                # overwrite the old one and replace it with the new one
                setattr(det, 'fzp', fzp_summed)
    
                
    


   
    def countmask(self):
        '''
        Function to filter on the counts per integrated frame

        every integratind fram should include the same number of shots
        for some frames this will not be the case due to various issues,
        these frames should be filtered out for all detectors
        '''
            
        for detector, det_spec_dict in self.yaml['int_detectors'].items(): 
            useDask=det_spec_dict['useDask']
            det=getattr(self,detector)
            #####ZY_edits - 062126 - add timestamp sorting and rolling as options before countmask, to fix potential issues with timestamp misalignment and Andor roll (if not already corrected in raw data)
            # ----- SORT + ROLL on the FULL sequence, BEFORE the countmask -----
            if self.yaml.get('timestamp_sort', False) and not useDask:
                idx = np.argsort(self.top_timestamp)          # full timestamp, NOT [countmask]
                for at in self.yaml[det_spec_dict['attrdict']]:
                    setattr(det, at, getattr(det, at)[idx])
                if (self.scantype == 'delay' or self.scantype == 'delay_fly'):
                    setattr(det, 'delay', getattr(det, 'delay')[idx])
                if (self.scantype == 'mono'  or self.scantype == 'mono_fly'):
                    setattr(det, 'mono',  getattr(det, 'mono')[idx])

                if self.yaml.get('andor_roll', False):
                    setattr(det, 'full_area', np.roll(getattr(det, 'full_area'), 1, axis=0))
                    for at in self.yaml[det_spec_dict['attrdict']]:
                        setattr(det, at, getattr(det, at)[1:])
                    if (self.scantype == 'delay' or self.scantype == 'delay_fly'):
                        setattr(det, 'delay', getattr(det, 'delay')[1:])
                    if (self.scantype == 'mono'  or self.scantype == 'mono_fly'):
                        setattr(det, 'mono',  getattr(det, 'mono')[1:])

            # ----- COUNTMASK now runs on the sorted+rolled data -----
            if len(self.yaml['expected_count']) == 0:
                expected_count = st.mode(det.count, keepdims=False)[0]
            else:
                expected_count = self.yaml['expected_count'][0]
            countmask = (det.count<expected_count+2)&(det.count>expected_count-2)
            for at in self.yaml[det_spec_dict['attrdict']]:
                a = getattr(det,at)
                if useDask:
                    mask_nd = countmask.reshape((countmask.shape[0],) + (1,) * (a.ndim - 1))
                    masked = a * mask_nd
                else:
                    masked = a[countmask]
                setattr(det, at, masked)

            if (self.scantype=='delay' or self.scantype=='delay_fly'):
                a = getattr(det,'delay')
                if useDask:
                    mask_nd = countmask.reshape((countmask.shape[0],) + (1,) * (a.ndim - 1))
                    masked = a * mask_nd
                else:
                    masked = a[countmask]
                setattr(det, 'delay', masked)

            if (self.scantype=='mono' or self.scantype=='mono_fly'):
                a = getattr(det,'mono')
                if useDask:
                    mask_nd = countmask.reshape((countmask.shape[0],) + (1,) * (a.ndim - 1))
                    masked = a * mask_nd
                else:
                    masked = a[countmask]
                setattr(det, 'mono', masked)

    def get_scanvar(self,intgrp):
        if (self.scantype=='mono' or self.scantype=='mono_fly'):
            #FIXME: fix mono scantype
            if len(self.yaml['mono_calib'])==0:
                #FIXME: just place incoming values here
                print('mono is not calibrated')
                for detector in self.yaml['int_detectors']: 
                    det = getattr(self,detector)
                    hrencoder = getattr(det,'mono_encoder')/getattr(det,'count')
                    setattr(det, 'mono', hrencoder)
            else:
                for detector in self.yaml['int_detectors']: 
                    det = getattr(self,detector)
                    mono_calib=[]
                    for i in np.arange(len(self.yaml['mono_calib'])):
                        mono_config = np.asarray(self.yaml['mono_calib'][i])
                        if np.logical_and((self.run>mono_config[0]),(self.run<mono_config[1])):
                            mono_calib = mono_config[2:4]
                    if len(mono_calib)==0:
                        print('mono is not calibrated for this run')
                        for detector in self.yaml['int_detectors']: 
                            det = getattr(self,detector)
                            hrencoder = getattr(det,'mono_encoder')/getattr(det,'count')
                            setattr(det, 'mono', hrencoder)
                    else:
                        #for integrating detectors, the mono encoder value is the sum over all shots
                        if self.scantype=='mono_fly':
                            # hrencoder = getattr(det,'mono_encoder')/getattr(det,'count')
                            ##### ZY_edits - 061926 - change count to accept+-1 counts
                            expected_count = st.mode(det.count, keepdims=False)[0]
                            hrencoder = getattr(det,'mono_encoder')/expected_count
                            tmp = np.polyval(mono_calib,hrencoder)
                            premirror = get_premirror_pitch(self.epics['MONO_premirror_pitch'])
                            mono = mono_energy(tmp,premirror)
                            # mono = np.polyval(self.yaml['mono_calib'],mono)
                            setattr(det, 'mono', mono)
                            ##### ZY_edits - 061726 - read the run's nominal delay
                            if self.yaml.get('TT_corr', {}).get('mono', False):
                                dkey = self.yaml['TT_corr'].get('delay_key',
                                                                self.yaml['scanvar']['delay'])
                                try:
                                    d = intgrp[detector][dkey][()]
                                    setattr(det, 'delay_nominal',
                                            float(np.nanmean(d.squeeze()/getattr(det,'count').squeeze())))
                                except Exception:
                                    print(f'WARN: nominal-delay key "{dkey}" missing -> mono-TT nominal=0')
                                    setattr(det, 'delay_nominal', 0.0)
                        elif self.scantype=='mono':
                            #FIXME: how do I here pull the scanvar 
                            hrencoder = getattr(det,'mono_encoder')/getattr(det,'count')
                            tmp = np.polyval(mono_calib,hrencoder)
                            premirror = get_premirror_pitch(self.epics['MONO_premirror_pitch'])
                            mono = mono_energy(tmp,premirror)
                            # mono = np.polyval(self.yaml['mono_calib'],mono)
                            setattr(det, 'mono', mono)
                            # mono = np.polyval(self.yaml['mono_calib'],hrencoder)
                            # setattr(det, 'mono', mono)
                    
        elif (self.scantype=='delay' or self.scantype==('delay_fly')):
            if self.scantype=='delay':
                delay_attr = self.yaml['scanvar']['delay']
            elif self.scantype=='delay_fly':
                delay_attr = self.yaml['scanvar']['delay']
            #FIXME: where is delay defined
            #np.logical_or(scan_var_name =='lxt',scan_var_name == 'lxt_ttc')
            for detector in self.yaml['int_detectors']: 
                det = getattr(self,detector)
                count = getattr(det,'count')
                # countmask = getattr(det, 'countmask')
                ##### ZY_edits - 062126 - change count to accept+-1 counts, fix for delay scan
                # expected_count = st.mode(count.squeeze(), keepdims=False)[0]
                tmp = intgrp[detector][delay_attr][()]
                delay = tmp.squeeze()/count.squeeze()
                print(f'delay shape {delay.shape}')
                setattr(det, 'delay', delay)
        else:
            print('scanvariable is unkown, binning not possible')
            
    ##### JW edits 9/16/26: I am doing the binning here, so then the masks are ready to be used in process.py
    def get_scanvar_bins(self):
        """ conducts binning on the the scanvariable, binning will be saved and used in the processing and reduction

        Returns:
           scanvar+bin_idx_mask (nframes,): a mask that contains the bin index at each value.
           centers (nbins, )              : the newly binned axis, contains the center values.
        """
        # if mono run
        if (self.scantype == 'mono' or self.scantype =='mono_fly'):
            for detector in self.yaml['int_detectors']: 
                det = getattr(self,detector)
                mono = getattr(det, 'mono')
                bin_ranges = self.yaml['mono_bin_ranges']
                bin_cfg = self.yaml['mono_bin_cfg']
                scantype = None
            # if mono-fly:
            if self.scantype =='mono_fly':
                scantype = 'fly'
            # if mono-step 
            if self.scantype == 'mono':
                scantype = 'step'
            
            # do the binning
            scanvar_bin_idx_mask, centers = bin_scanvar(mono, bin_ranges,bin_cfg, scantype)
        
        return scanvar_bin_idx_mask, centers
            
            
            
        

    ##### JW edits 9/16/26: creating a masks object and storing it as an attribute in integrating, 
    # this function should be processed at the really end as we need the number of frames after countmasking

    # right now, this is mainly only designed for SVLS, mono fly runs. 
    def process_masks(self):
        
        # will at least return something if not mono scans
        masks = None
        
        if (self.scantype == 'mono' or self.scantype=='mono_fly'):
            # ensure we are working with the desired integrated detector
            for detector in self.yaml['int_detectors']: 
                det = getattr(self,detector)
                
                evc = getattr(det, 'eventcodes')
                counts = getattr(det, 'count')
                # only works for mono runs right now
                mono = getattr(det, 'mono')
                
                expected_count = st.mode(counts, keepdims=False)[0]
                
                evc_on = np.asarray([evc[:,self.yaml['evc'][True]]/expected_count>0.5]).squeeze()#self.data.integrating.axis_svls.eventcodes[:,self.data.yaml['evc'][True]]#weird ymal magic turns 'on' automatically into True
                evc_off = np.asarray([evc[:,self.yaml['evc'][False]]/expected_count>0.5]).squeeze()#self.data.integrating.axis_svls.eventcodes[:,self.data.yaml['evc'][False]]#weird ymal magic turns 'off' automatically into False
                
                mono_bin_idx_mask, mono_bin_centers = self.get_scanvar_bins()
                
                # store the bin_centers as an attribute
                setattr(det, 'mono_bin_centers', mono_bin_centers)
                
                # examine I0 values, create mask that drops the values that are below 10% or NAN
                I0 = getattr(det, 'I0')
                finite_i0 = np.isfinite(I0) & (I0 > 0)
                med_i0 = np.median(I0[finite_i0]) if finite_i0.any() else 0.0
                good_i0 = finite_i0 & (I0 > 0.10 * med_i0)
            
                
                # initalize a Mask object
                masks = Masks(len(mono))
                
                # add the masks
                masks.add('laser_on', evc_on)
                masks.add('laser_off', evc_off)
                masks.add('mono_bin_idx', mono_bin_idx_mask)
                masks.add('good_i0', good_i0)
                
        # return the mask
        return masks
                
                
            