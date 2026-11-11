#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:48:34 2025

@author: leykza
"""

import mne 
import numpy as np 

import os
import pickle
import pandas as pd
import re

from glob import glob
from mne_bids import (BIDSPath,write_raw_bids,mark_channels)
from mne_icalabel import label_components
import matplotlib.pyplot as plt 
from scipy.signal import butter, sosfiltfilt, resample_poly, sosfreqz

def relabel_events(raw, minGap):
    """
    

    Parameters
    ----------
    raw : TYPE
        DESCRIPTION.

    Returns
    -------
    raw : TYPE
        DESCRIPTION.

    """
    
    sfreq = raw.info['sfreq']
    
    events, events_dict = mne.events_from_annotations(raw)
    
    # events_clean        = mne.pick_events(events, 
    #                                       exclude = [1127, 255, 10001]) 
    
    # convert events to annotations
    stim_ann            = mne.annotations_from_events(events, 
                                                      sfreq,
                                                      event_desc = [10001])
    
    resp_ann            = mne.annotations_from_events(events, 
                                                      sfreq, 
                                                      event_desc = [10002])
       
    # get t onset of stim 
    stim_ann_onset      = stim_ann.onset
    
    # remove accidental markers from photodiose flashes
    valid_idx = [0]
    j = 1
    jj = 0
    while True:
       
        if j == stim_ann_onset.shape[0]:
            break
        elif (stim_ann_onset[j] - stim_ann_onset[jj]) < minGap[2]: # this 2 is hard coded for the main run rn
            j += 1 # same as j = j+1
        else:
            valid_idx.append(j)
            jj = j
            j += 1
    
    stim_ann       = stim_ann[valid_idx]
    stim_ann_onset = stim_ann.onset
    # now you have valid stim annotations, resp annotations 
    # so you can reannotate the data with just these two
    
    new_annot = mne.Annotations(onset       = list(stim_ann.onset) + list(resp_ann.onset),
                                duration    = list(stim_ann.duration) + list(resp_ann.duration),
                                description = ['stim_onset']*len(stim_ann.description) + ['button_press']*len(resp_ann.description))

    raw.set_annotations(new_annot)
    
    return raw, new_annot


def detect_bad_channels(raw):

    eeg_data        = raw.get_data() # we retrieve an array that contains all the data from the recording
    nchans,nsamples = eeg_data.shape

    perc_art_chans = np.zeros((nchans, 1)) # array that contain the perc of recording that was an artifact
    flat_thr           = 1e-6
    artifact_thr       = 100e-6

    for ch in range(nchans):
        
        ch_data = np.abs(eeg_data[ch, :])
        
        ch_type = mne.channel_type(raw.info, ch)
        if ch_type == 'misc':
            ch_data *= 1e2
            
        perc_art_chans[ch,:] = np.sum(np.logical_or(ch_data>=artifact_thr, ch_data<=flat_thr))/nsamples

    bad_channels = np.where(perc_art_chans>=0.2)[0].tolist()
    
    ch_names     = raw.ch_names
    bads         = [raw.ch_names[c] for c in bad_channels]
    return bads
    

def remove_components(bids_path, bids_ica_path, components):
   
    raw          = mne.io.read_raw_fif(str(bids_path))
    raw = raw.set_montage("easycap-M1")
    raw = raw.load_data().set_eeg_reference("average")

    raw.annotations.rename({'bad blink': 'blink'})
    ica = mne.preprocessing.read_ica(str(bids_ica_path)+'.fif')
    
    if components is None:
        ic_labels = label_components(raw, ica, 'iclabel')
        exclude_idx = [
            idx for idx, label in enumerate(ic_labels['labels']) if label not in ["brain", "other"]
        ]
        
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        ica.exclude     = exclude_idx + eog_indices # components chosen based on various plots above       
    else:
        ica.exclude=components
    
    # ! ica.apply() changes the Raw object in-place, so let's make a copy first:
    clean_data      = raw.copy()
    ica.apply(clean_data)
    
    return clean_data, raw

def read_components(excel_path):

    # Load your Excel file
    df = pd.read_excel(excel_path)

    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Fill down filenames
    df["filename"] = df["filename"].ffill()
    
    # Keep only ICA rows
    df = df[df["components_marked_bad"].astype(str).str.match(r"ICA\d{3}")]
    
    # Clean verified_bad column
    df["verified_bad"] = df["verified_bad"].astype(str).str.strip().str.upper()
    
    df["component"] = (
    df["components_marked_bad"]
    .str.extract(r"ICA(\d{3})")
    .astype(int))

    # Keep ONLY rows where verified_bad starts with "YES"
    df = df[df["verified_bad"].str.startswith("YES")]
    
    # Extract subject and run numbers
    df["subject"] = df["filename"].str.extract(r"sub-(\d+)").astype(int)
    df["run"] = df["filename"].str.extract(r"run-(\d+)").astype(int)
    
    # Group by subject + run
    result = df.groupby(["subject", "run"])["component"].apply(list).reset_index()

    return result


def load_pickle_dict(file_path):
    """
    Load a dictionary from a pickle (.pkl) file.
    
    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        dict: The loaded dictionary.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded object is not a dictionary.
        pickle.UnpicklingError: If the file is not a valid pickle.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error unpickling file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading pickle file: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary, but got {type(data).__name__}")

    return data

def check_events_with_behavior(eeg, beh, session, sub, ri, tolerance=0.01):
    
    events, events_dict = mne.events_from_annotations(eeg)

    with mne.use_log_level("WARNING"):  # hide some irrelevant info messages
        
        events_stim     = events[events[:,2]==events_dict['stim_onset'],:]
        event_dict = {'stim_onset' : events_dict['stim_onset']}
        #events = np.insert(events, 1, events_dom, axis=1)
        
        events_bp     = events[events[:,2]==events_dict['button_press'],:]
        event_dict_bp = {'button_press' : events_dict['button_press']}
        #events = np.insert(events, 1, events_dom, axis=1)
        
        events_beh     = []        
        events_beh_dur = []
        
    if np.logical_and(sub == 5, ri == 1): # special case dealing with extra flips
        extra_flips = [1,423]
        events_stim = np.delete(events_stim, extra_flips, axis=0)
        
    if np.logical_and(sub == 8, ri == 1): # special case dealing with extra flips
        extra_flips = [415]
        events_stim = np.delete(events_stim, extra_flips, axis=0)
    
    if np.logical_and(sub == 8, ri == 2): # special case dealing with extra flips
        extra_flips = [421]
        events_stim = np.delete(events_stim, extra_flips, axis=0)
    
    if np.logical_and(sub == 9, ri == 1): # special case dealing with extra flips
        extra_flips = [420,841]
        events_stim = np.delete(events_stim, extra_flips, axis=0)
          
                
       
    for block in range(1,3):
         
        subName_tmp = session + 'S'+ str(sub) # for the sake of finding the key in beh data
        block_idx = block
        dict_key = f'MainRun_Run-{ri}_Block-{block_idx}_{subName_tmp}'
        
        #load behavioral data for block
        beh_data        = beh[dict_key][2]
     
        events_beh.append(beh_data['onset']-beh_data['onset'][0])
        events_beh_dur.append(beh_data['duration'])       
     
    
    # collect EEG events   
    first_ts_eeg = events_stim[0,0]/eeg.info['sfreq']
    events_eeg        = events_stim[:,0]/eeg.info['sfreq']-events_stim[0,0]/eeg.info['sfreq']     

    if np.any(np.logical_and((np.diff(events_eeg)<=120),(np.diff(events_eeg)>=20))):
       
        screen_flips      = np.where((np.logical_and((np.diff(events_eeg)<=120),(np.diff(events_eeg)>=20))))
        last_before_break = np.where(np.diff(events_eeg)>120)
            
        if np.size(screen_flips) == 2: 
            events_beh[1]  = events_beh[1] +  events_eeg[last_before_break[0]+2]
            events_beh[0]  = events_beh[0] +  events_eeg[1]
            
        elif np.size(screen_flips) > 2:
            events_beh[1]  = events_beh[1] +  events_eeg[last_before_break[0]+3]
            events_beh[0]  = events_beh[0] +  events_eeg[2]
            
    else:
        last_before_break = np.where(np.diff(events_eeg)>120)
        events_beh[1]  = events_beh[1] +  events_eeg[last_before_break[0]+1]
        
    # collect beh events
    events_beh     = pd.concat(events_beh)
    events_beh_dur = pd.concat(events_beh_dur)
    
    new_index = range(events_beh.size) # to make beh go from 0-840
    events_beh.index     = new_index
    events_beh_dur.index = new_index
    
    # compare events before cleaning      
    # plt.figure()
    # plt.eventplot(events_eeg, lineoffsets=1, color='r')
    # plt.eventplot(events_beh, lineoffsets=0, color='b')
    
    # clean and align for checking w beh
    new_eeg_events = align_with_tolerance(events_beh, events_eeg, tolerance)
    
    # plt.figure()
    # plt.eventplot(new_eeg_events, lineoffsets=1, color='r')
    # plt.eventplot(events_beh, lineoffsets=0, color='b')
    
    # references events back to original space before making annotations
    new_eeg_events = new_eeg_events + first_ts_eeg
    all_eeg_events = np.concatenate([new_eeg_events,events_bp[:,0]/eeg.info['sfreq']])
    
    # make annotations from beh events 
    my_annot  = mne.Annotations(
        onset    = all_eeg_events, 
        duration = np.zeros(all_eeg_events.shape[0],), # in seconds
        description = ['stim_onset']*new_eeg_events.shape[0] + ['button_press']*events_bp.shape[0])

    eeg.set_annotations(my_annot)
    
    return eeg    

    
def align_with_tolerance(reference, target, tolerance):
    """
    Match timestamps within a tolerance.

    Parameters
    ----------
    reference : list of float
        Reference timestamps

    target : list of float
        Target timestamps

    tolerance : float
        Maximum allowed difference

    Returns
    -------
    aligned : list
        Matched target timestamps or None if missing
    """

    aligned = []

    for r in reference:

        # Find closest target timestamp
        closest = None
        min_diff = float("inf")

        for t in target:
            diff = abs(r - t)

            if diff < min_diff:
                min_diff = diff
                closest = t

        # Accept only if within tolerance
        if min_diff <= tolerance:
            aligned.append(closest)
        else:
            aligned.append(float(r))

    return np.array(aligned)
