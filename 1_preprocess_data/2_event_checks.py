#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:21:17 2025

@author: leykza

updated 11/2026

Takes BIDS format data, annotates events, and plots stim and behavior onsets across task time
for cross checking data and ensuring EEG markers are coming in correctly.

In this data, stim onsets on the EEG data are recorded through a photodiode (light) sensor on the subject
screen, but it would sometimes mark a 'stim onset' when the stimulus offset (change in light).

Later scripts will check all onsets in the EEG and compare them with the stim onsets and 
behavior recorded within PsychoPy, as well as match which stimulus was what type, since
the photodiode doesn't mark which kind, just that there was a stim onset.

Purpose is for removing erroneous markers (any marker shorter than the interstimulus intervals),
reannotating them back into the raw data and visualizing that the EEG markers are roughly as 
they should be (n stim events, blocks are clearly noticable and specific to run types). 

There are 840 trials every main run, jittered randomly between 0.4-1.6s.

! Be mindful that this script is written for when the data comes right from the BIDS renaming. 
After it is reannotated here,the marker IDs change and gets saved and overwritten. 
If you need to redo running a sub through this, go back to 1_rename_to_BIDS.py 
and run the sub that was renamed. That way it will start the process over.

"""

from glob import glob
from mne_bids import (BIDSPath,write_raw_bids)

import os

import mne
import matplotlib.pyplot as plt 

#%% Go through each vhdr BIDS formatted file in each subject folder, and open it
subPath       = r'D:\switch_coherence_data'
subFolders     = glob(os.path.join(subPath, 'sub-*'))


figFolder      = r'D:\switch_coherence_data\subFigs'

RunTypes       = ['Main']  # we are just using Main runs
RunNumbers     = [2]  # there are two Main runs each sub
minGap         = [0.45] # used to remove flashes from task, minimum amount of ITI time a marker is allowed
maxGap         = [40] # to remove screen flips. When the blocks starts there is a marker (light change) but no real trial
#%% sub

missing_markers= []

for sub in subFolders:
    
    subName    = sub.split('/')[-1]
    subName    = subName.split('-')[-1]
    
    eventFiles = glob(os.path.join(subPath, 'sub-' + subName, '*'))
    session    = eventFiles[0].split('-')[-1]  
    
    # run type (tt, mock, main)
    for rti in range(len(RunTypes)):
        
        # run number (1,2,2)
        for ri in range(1, RunNumbers[rti]+1):
            # EEG
            bids_path   = BIDSPath(root    = subPath,
                                   subject = subName, 
                                   session = session,
                                   task    = RunTypes[rti],
                                   run     = ri,
                                   extension = 'vhdr', 
                                   datatype  ='eeg')
        
            raw          = mne.io.read_raw_brainvision(str(bids_path))
            sfreq        = raw.info["sfreq"]
            events, events_dict = mne.events_from_annotations(raw)
           
            # look at its events before cleaning. You will see marker on and offsets, 
            # which here are almost the same time. so pick markers that onset first
            # For some reason what are clearly the trial onsets in the plot are labeled response, and vice versa 
            mne.viz.plot_events(
                    events, sfreq = raw.info["sfreq"], 
                    first_samp          = raw.first_samp, 
                    event_id            = events_dict)
            # check which events happen first in time and choose the onsets
            stim_ann1            = mne.annotations_from_events(events, sfreq, event_desc=[1255]) 
            resp_ann1            = mne.annotations_from_events(events, sfreq, event_desc=[251]) 
            stim_ann2            = mne.annotations_from_events(events, sfreq, event_desc=[1127]) 
            resp_ann2            = mne.annotations_from_events(events, sfreq, event_desc=[255]) 
            stim_ann_onset1      = stim_ann1.onset
            stim_ann_onset2      = stim_ann2.onset
            resp_ann_onset1      = resp_ann1.onset
            resp_ann_onset2      = resp_ann2.onset
            
            # set channel type and montage
            raw.set_channel_types(dict(zip(raw.ch_names, ['eeg']*raw.info['nchan'])))
            raw = raw.set_montage("easycap-M1")
                
            # remove the other events based on the onsets
            events_clean = mne.pick_events(events, exclude = [1127, 255, 10001]) 
            
            
            stim_ann            = mne.annotations_from_events(events_clean, sfreq, event_desc=[1255]) 
            resp_ann            = mne.annotations_from_events(events_clean, sfreq, event_desc=[251]) 
           
            # get onset of stim_ann
            stim_ann_onset      = stim_ann.onset
            
            
            if stim_ann_onset.size < 840: # note which runs have missing markers
                missing = subName,ri
                missing_markers.append(missing) 
            # remove accidental photodiode markers (flashes)  
            valid_idx = [0]
            j = 1
            jj = 0
            while True:
               
                if j == stim_ann_onset.shape[0]:
                    break
                elif (stim_ann_onset[j] - stim_ann_onset[jj]) < minGap[rti]:
                    j += 1 # same as j = j+1
                else:
                    valid_idx.append(j)
                    jj = j
                    j += 1
            
            stim_ann = stim_ann[valid_idx] # this should be ~840 in size
            
            # now you have valid stim annotations, resp annotations 
            # so you can reannotate the data with just these two
            
            new_annot = mne.Annotations(onset       = list(stim_ann.onset) + list(resp_ann.onset),
                                        duration    = list(stim_ann.duration) + list(resp_ann.duration),
                                        description = list(stim_ann.description) + list(resp_ann.description))
            
            
            # reannotate raw
            raw.set_annotations(new_annot)
            events_clean, events_dict = mne.events_from_annotations(raw)
        

            if '1255' not in list(events_dict.keys()): # if photodiode died
                nResponses = len(mne.pick_events(events_clean, include = events_dict['251']))
                events_dict = {'button_press' : events_dict['251']}

            else:
                
                # count events and cross check with behav files
                nResponses = len(mne.pick_events(events_clean, include = events_dict['251']))
                nTrials    = len(mne.pick_events(events_clean, include = events_dict['1255']))
                 
                events_dict  = {'stim onset' : events_dict['1255'], 'button_press' : events_dict['251']}

            # look at its events  
            fig = mne.viz.plot_events(
                    events_clean, sfreq = raw.info["sfreq"], 
                    first_samp          = raw.first_samp, 
                    event_id            = events_dict
                    )
            
            plt.title(RunTypes[rti] + ' ' + str(ri) + ' ' + str(subName))
            
            figname = figFolder + '/' +  subName + '_' + RunTypes[rti] + ' ' + str(ri) + '.png'
            
            # save the image with the events if you want
            #plt.savefig(figname)
            plt.close()
            
            # write the new fif file
            save_path   = BIDSPath(root    = subPath,
                                   subject = subName, 
                                   session = session,
                                   task    = RunTypes[rti],
                                   run     = ri,
                                   extension = 'fif',
                                   datatype  ='eeg')
            
            raw.annotations.rename({'1255' : 'stim onset', '251' : 'button_press'})
            write_raw_bids(raw, bids_path=save_path, events=events_clean, format='BrainVision',
                           event_id=events_dict, overwrite=True, verbose=False)

            
            
            