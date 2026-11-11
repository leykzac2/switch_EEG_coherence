#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 09:49:57 2025

@author: leykza

ICA
update: 11/7/26

Script that takes the annotated data, applies a 60Hz notch and average 
re-referencing to average of all electrodes before performing ICA.
'IC solutions' are saved to then be evaluated by MNEs IC label and a human
to decide which components to reject in 5_ComponentRejection.py.
Running this on all subjects takes a handful of hours. It is best to run it 
completely and leave it alone, ideally overnight for example.
        - loads annotated data saved from 3_post_run_qualitycheck.py
        - (1) To address line noise, a 60Hz notch is applied to downsampled data and checked to see what was removed around events (addressing line noise: lecture 6).
        - (2) Filtered data is re-referenced to the common average of all electrodes (assignment 5).
        - Data decomposition is applied on the resulting data using ICA
        - saves ICs for use in component rejection in 5_ComponentRejection.py

"""
from glob import glob
from mne_bids import BIDSPath

import os
import mne

from mne.preprocessing import ICA

#%% Load data 

# open fif file
subPath     = r'D:\switch_coherence_data'
subFolders     = glob(os.path.join(subPath, 'sub-*'))

for sub in subFolders:    

    subName    = sub.split('/')[-1]
    subName    = subName.split('-')[-1]
    
    eventFiles = glob(os.path.join(subPath, 'sub-' + subName, '*'))
    session    = eventFiles[0].split('-')[-1]  
       
    # run number (1,2)
    for ri in range(1, 3):
        
        # EEG
        bids_path   = BIDSPath(root    = subPath,
                               subject = subName, 
                               session = session,
                               task    = 'Main',
                               run     = ri,
                               extension = 'fif', 
                               datatype  ='eeg')


        raw          = mne.io.read_raw_fif(str(bids_path))
        
        # uncomment to look at the raw trace with annotated events before notch
        # raw.plot()
        
        # (1) To address line noise, a 60Hz notch is applied to downsampled data 
        # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter
        # Applies a zero-phase notch filter to the channels selected by “picks”. By default the data of the Raw object is modified inplace.
        raw.load_data().notch_filter([60])
        
        # uncomment to check to see what was removed around events (addressing line noise: lecture 6)
        # raw.plot()
        
        raw = raw.set_montage("easycap-M1") # set the montage of the cap used
        
        # (2) re-referenced to the common average of all electrodes
        # https://mne.tools/stable/generated/mne.set_eeg_reference.html
        raw = raw.load_data().set_eeg_reference("average")

        raw.annotations.rename({'bad blink': 'blink'}) # renaming to start with 'bad' so they are removed in the ICA decomposition
        
        # Data decomposition using Independent Component Analysis (ICA). https://mne.tools/stable/generated/mne.preprocessing.ICA.html
        ica = ICA(random_state=123, 
                  max_iter='auto', 
                  method = 'infomax', 
                  fit_params=dict(extended=True))
        ica.fit(raw, # Run the ICA decomposition on raw data
                reject_by_annotation=True) # Whether to omit bad segments from the data before fitting. If True (default), annotated segments whose description begins with 'bad' are omitted# Run the ICA decomposition on raw data
        
        bids_ica_path   = BIDSPath(root    = subPath,
                               subject = subName, 
                               session = session,
                               task    = 'Main',
                               run     = ri,
                               description = 'ICSolutions',
                               extension = 'fif')
        
        # Store ICA solution into a fiff file. 
        ica.save(str(bids_ica_path) + '.fif', overwrite=True)