# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:26:57 2026

@author: Leykza

updated 11/8/2026

    - Imports IC solutions data made from 4_ICA.py
    - IC labeled data are automatically labeled for good/brain and bad/non-brain components with MNE iclabel
    - This step should be done going one subject run at a time, stopping at ica.plot_properties to see the components labeled 'bad'  
    - The later analyses scripts will pull info from an excel sheet/table made at this point to remove bad components for analysis 
    - That way the decisions on what components to remove can be modified based on analyses, instead of saving a certain type of 'clean data' itself somewhere
    - Components plotted here should be manually checked and saved in an excel sheet for later analysis scripts
    - So as the user stops at ica.plot_properties, they should document somewhere:
        - the subject
        - the run number
        - the ICA component
        - and note whether it should be rejected (Y/N)
      
iclabel documentation:
    https://mne.tools/mne-icalabel/stable/generated/api/mne_icalabel.iclabel.iclabel_label_components.html#mne_icalabel.iclabel.iclabel_label_components    
    https://mne.tools/mne-icalabel/stable/generated/api/mne_icalabel.label_components.html

"""
from glob import glob
from mne_bids import (BIDSPath)
#from utilities_eeg import relabel_events, detect_bad_channels

import os
import numpy as np
import mne
#import pandas as pd 
#import matplotlib.pyplot as plt 
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne_icalabel import label_components

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
        
        #
        bids_path   = BIDSPath(root    = subPath,
                               subject = subName, 
                               session = session,
                               task    = 'Main',
                               run     = ri,
                               extension = 'fif', 
                               datatype  ='eeg')

        # fif data is loaded, set and re-referenced again (it itself was not modified in place earlier) 
        # to use in label_components() against the fitted ica instance 
        raw          = mne.io.read_raw_fif(str(bids_path))
        raw = raw.set_montage("easycap-M1")
        raw = raw.load_data().set_eeg_reference("average")

        raw.annotations.rename({'bad blink': 'blink'})
        
        # load IC solutions
        bids_ica_path   = BIDSPath(root    = subPath,
                               subject = subName, 
                               session = session,
                               task    = 'Main',
                               run     = ri,
                               description = 'ICSolutions',
                               extension = 'fif')
        
        ica = mne.preprocessing.read_ica(str(bids_ica_path)+'.fif') # the fitted ICA instance, read from the BIDS ica path
        
        # Automatically label the ICA components with 'iclabel' method
        ic_labels = label_components(raw, # data
                                     ica, # the fitted ICA instance
                                     'iclabel') # method for labeling components
       # which ones are not in brain or other, so more confidently classified as noise
        exclude_idx = [
            idx for idx, label in enumerate(ic_labels['labels']) if label not in ["brain", "other"]
        ]
        
        # can look at the classified good ones if you want
        # good_idx = [
        #     idx for idx, label in enumerate(ic_labels['labels']) if label in ["brain", "other"]
        # ]
        
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        
        # breakpoint to stop here and see run components, document which ones should be rejected
        #ica.plot_properties(raw, np.unique(exclude_idx + eog_indices)) # bads
        
        # barplot of ICA component "EOG match" scores
        #ica.plot_scores(eog_scores)
        
        
#raw.plot()

#%% Optional plotting to visualize componenets and component traces

# # visualize artifact component data
# # pick some channels that clearly show blinks
# regexp = r"Fp1"
# artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
# raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# #%%  summary of how the ocular artifact manifests across each channel type
# eog_evoked = create_eog_epochs(raw, ch_name=regexp).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()

# #%% plot sources
# ica.plot_sources(raw, show_scrollbars=False)

# # plot components
# ica.plot_components()


