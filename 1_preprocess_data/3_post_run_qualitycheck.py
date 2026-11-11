#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The purpose of this step is to obtain some basic quality metrics of the EEG data and start some minimal
cleaning for easier use in later scripts.

It performs the following steps in order:
    - loads reannotated BIDS format data from 2_event_checks.py
    - (1) zero-phase, Butterworth bandpass (1-100Hz) filter preserving timing and phase (filtering: lecture 4, sosfiltfilt: lecture 6, using MNE analogous IIR butterworth filter). 
    - (2) downsample data to 250Hz and decimated (20:1) using (resample_poly: assignment 5, MNE verson of resample with polyphase method). Done early for computational purposes. 
    - (3) annotate data to remove periods of non-interest (breaks, time before and after task, blinks)
    - MNE filtering steps tend to use the scipy.signal base code, but documentation to the functions used are provided
 
MNE documentation of class methods
raw.resample: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
    The intended purpose of this function is primarily to speed up computations (e.g., projection calculation) 
    when precise timing of events is not required, as downsampling raw data effectively jitters trigger timings. 
    It is generally recommended not to epoch downsampled data, but instead epoch and then downsample, 
    as epoching downsampled data jitters triggers.

raw.filter (IIR): 
    4th order, zero-phase, non-causal Butterworth bandpass (1-100Hz) filter 
    (https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter)

mne.preprocessing.find_eog_events
    https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html
    # uses an FIR filter to reject data around voltage change in selected electrodes
    past X-amount of Hz (Designs a two-pass forward and reverse, zero-phase, non-causal bandpass filter)
'''
from glob import glob
from mne_bids import (BIDSPath,write_raw_bids,mark_channels)
from utilities_eeg import relabel_events, detect_bad_channels, test_numpy, test_mne
from scipy.signal import butter, sosfiltfilt, resample_poly
import os
import numpy as np
import mne 
import pandas as pd 
import matplotlib.pyplot as plt 

#%% Load data 

l_freq       = 1 
h_freq       = 100
minGap       = [3, 0.45, 0.45] # used to remove flashes from task

# Go through each vhdr BIDS formatted file in each subject folder, and open it
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
                               extension = 'vhdr', 
                               datatype  ='eeg')


        raw          = mne.io.read_raw_brainvision(str(bids_path))
        sfreq        = raw.info["sfreq"] # raw fs
        
        # (1) 4th order, zero-phase, non-causal Butterworth bandpass (1-100Hz) filter (https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter)
        iir_params = dict(order=4, ftype="butter")
        raw = raw.load_data().filter(l_freq, 
                                     h_freq, 
                                     method     = 'iir', 
                                     iir_params = iir_params, # When method='iir', phase='zero' (default) or equivalently 'zero-double' constructs and applies IIR filter twice, once forward, and once backward (making it non-causal) using filtfilt(); phase='forward' will apply the filter once in the forward (causal) direction using lfilter().
                                     phase      = 'zero', 
                                     verbose=None)

        # (2) resample to 250Hz with resample poly - will move this to after epoching later
        raw = raw.copy().resample(sfreq   = 250,
                                  method  = "polyphase",
                                  verbose = True,)
       
        # annotate 
        raw, task_annot = relabel_events(raw, minGap) 
        #task_annot = raw.annotations.copy()
        
        # take out breaks
        onset_tasks = task_annot.onset
        iti         = np.diff(onset_tasks)
        max_iti     = np.max(iti)
        max_iti_idx = np.argmax(iti)

        br_onsets = [onset_tasks[max_iti_idx-1]+2] # check time in s of start of task to end
        br_durations = [max_iti] # 600s = 10mins per block
        descriptions = ["bad break"]

        all_annot = task_annot.copy()
        all_annot.append(br_onsets, br_durations, descriptions)
        
        # ---- time before and after task

        # before
        bf_onsets = raw.first_time
        bf_durations = [onset_tasks[0] - bf_onsets]
        descriptions = ["bad before"]


        all_annot.append(bf_onsets, bf_durations, descriptions)
        
        # after
        taskend_idx   = np.argmax(onset_tasks)
        task_end      = onset_tasks[taskend_idx]

        af_onsets    = task_end + 10 # 10s after last stim
        af_durations = [60] 
        descriptions = ["bad after"]

        all_annot.append(af_onsets, af_durations, descriptions)

        raw.set_annotations(all_annot)  # add to existing
        
        # take out blinks
        # ---- blinks
        # https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html
        # uses an FIR filter to reject data around voltage change in 'Fp1', 'Fp2' past 10Hz (Designs a two-pass forward and reverse, zero-phase, non-causal bandpass filter:)
        eog_events   = mne.preprocessing.find_eog_events(raw, 
                                                         ch_name = ['Fp1', 'Fp2'],
                                                         l_freq = 1, 
                                                         h_freq = 10, 
                                                         reject_by_annotation=True) 

        bl_onsets    = eog_events[:, 0] / raw.info["sfreq"]-0.25
        bl_durations = [0.5] * len(eog_events)
        descriptions = ["bad blink"] * len(eog_events)

        blink_annot  = mne.Annotations(
            bl_onsets, 
            bl_durations, 
            descriptions, 
            orig_time = raw.info["meas_date"]
        )

        all_annot.append(bl_onsets, bl_durations, descriptions)

        raw.set_annotations(all_annot)  # add to existing

        bads = detect_bad_channels(raw)
        raw.info['bads'] = bads
        
        mark_channels(bids_path=bids_path, ch_names=bads,
                      status="bad", verbose=False)
        
        
        save_fname = str(bids_path).split('.')[0] + '.fif'
        
        raw.save(save_fname, overwrite=True)



