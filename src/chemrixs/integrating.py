
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import yaml
from chemrixs.detector import Detector
from chemrixs.utils import *
import scipy.stats as st


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
        # # print(sum(self.andor_vls.count_mask))
        # ##### ZY_edits - add timestamp
        self.top_timestamp = (np.asarray(intgrp['timestamp'])
                              if self.yaml.get('timestamp_sort', False) and 'timestamp' in intgrp
                              else None)

        self.get_scanvar(intgrp)
        self.countmask()
        self.summing_channels()

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
            if (hasattr(det,'fim_0') and hasattr(det,'fim_1')):
                I0 = det.fim0.copy()+det.fim1.copy()
                setattr(det,'I0',I0)
                delattr(det,'fim0')
                delattr(det,'fim1')
            elif hasattr(det,'fim0'):
                I0 = det.fim0.copy()
                setattr(det,'I0',I0)
                delattr(det,'fim0')
            elif hasattr(det,'fim1'):
                I0 = det.fim0.copy()
                setattr(det,'I0',I0)
                delattr(det,'fim1')


   
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

            #####ZY_edits - 062126 - add timestamp sorting and rolling as options before countmask, to fix potential issues with timestamp misalignment and Andor roll (if not already corrected in raw data)
            # ----- SORT + ROLL on the FULL sequence, BEFORE the countmask -----
                    setattr(det, at, getattr(det, at)[idx])
                if (self.scantype == 'delay' or self.scantype == 'delay_fly'):
                    setattr(det, 'delay', getattr(det, 'delay')[idx])
                if (self.scantype == 'mono'  or self.scantype == 'mono_fly'):
                    setattr(det, 'mono',  getattr(det, 'mono')[idx])

                if self.yaml.get('andor_roll', False):
                    print('hey yes doing the rolling')
                    setattr(det, 'full_area', np.roll(getattr(det, 'full_area'), 1, axis=0))
                    for at in self.yaml[det_spec_dict['attrdict']]:
                        setattr(det, at, getattr(det, at)[1:])
                    if (self.scantype == 'delay' or self.scantype == 'delay_fly'):
                        if (self.scantype == 'mono'  or self.scantype == 'mono_fly'):
                            setattr(det, 'mono',  getattr(det, 'mono')[1:])
#########end changes

            if len(self.yaml['expected_count']) == 0:
                expected_count = st.mode(det.count, keepdims=False)[0]
            else:
                expected_count = self.yaml['expected_count']
            countmask = (det.count<=expected_count+1)&(det.count>=expected_count-1)
            # breakpoint()
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
        # 1. get detector
        # 2. find mono_encoder per shot
        # 3. calibrate mono_encoder with mono_calib coefficients
        # 4. store incident energy as the 'mono' attribute
        for detector in self.yaml['int_detectors']:
            det = getattr(self, detector)
            
            # NOTE: mono scans
            if self.scantype == 'mono_fly' or self.scantype=='mono':
                # if our mono_calib is empty in the yaml file, not calibrated yet
                # NOTE: This is assuming that we will never have the situation where yaml.mono_calib has some values
                # in it but are missing the coefficients
                mono_calib = self.yaml['mono_calib']
                if len(mono_calib) != 0:
                    mono_calib_coeff=[]
                    # NOTE: loop through all mono calibration configurations, for example:
                    # run 1 - 81, 82 - 150, 151 - 253. then iterator would be range(3)
                    for mono_config in mono_calib:
                        if (self.run>=mono_config[0]) and (self.run<=mono_config[1]):
                            mono_calib_coeff = mono_config[2:4]
                            
                    # with the coefficients, calculate the shot to shot incident energy
                    # with expected count
                    expected_count = st.mode(det.count, keepdims=False)[0]
                    hrencoder = getattr(det,'mono_encoder')/expected_count
                    
                    # # with normal count
                    # hrencoder = getattr(det,'mono_encoder')/getattr(det,'count')
                    
                    tmp = np.polyval(mono_calib_coeff,hrencoder)
                    premirror = get_premirror_pitch(self.epics['MONO_premirror_pitch'])
                    mono = mono_energy(tmp,premirror)
                    setattr(det, 'mono', mono)
                    
                # the mono_calib in the yaml is empty, so not calibrated
                else:
                    hrencoder = getattr(det,'mono_encoder')/getattr(det,'count')
                    setattr(det, 'mono', hrencoder)
                    print('mono is not calibrated')
            # NOTE: delay scans
            elif self.scantype == 'delay' or self.scantype == 'delay_fly':
                if self.scantype=='delay':
                    delay_attr = self.yaml['scanvar']['delay']
                elif self.scantype=='delay_fly':
                    # FIXME: Is the delay fly scanvar not delay_fly but is also delay
                    delay_attr = self.yaml['scanvar']['delay']
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