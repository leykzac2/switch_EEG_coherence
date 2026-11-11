# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:28:38 2026

@author: Leykza
"""
'''
7/14/26 
- update to 3 cycles
- 6-70Hz frequency estimation
- power calculated from -50-450ms


updated 11/2026

Cleans data through removal of unwanted ICA components, inserts missing 
EEG markers if any, and epochs around stim onsets.

Epochs are organized according to trial contrast of interest

PSD estimations are taken from the epoch averages

MNE documentation
interpolate bad channels
https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.interpolate_bads

Epochs
https://mne.tools/stable/generated/mne.Epochs.html

ICA methods
https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA

ica.apply
https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.apply
'''
from glob import glob
from mne_bids import (BIDSPath,write_raw_bids,mark_channels)
from utilities_eeg import *

import os
import numpy as np
import mne
import pandas as pd 
import pickle
import matplotlib.pyplot as plt 
import scipy.stats

from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.minimum_norm import read_inverse_operator, source_induced_power
from mne.channels import find_ch_adjacency
from mne.datasets import sample
from mne.stats import combine_adjacency, spatio_temporal_cluster_test, f_mway_rm, f_threshold_mway_rm, fdr_correction
from mne.viz import plot_compare_evokeds
#%% Load data 

# open fif file
subPath     = r'D:\switch_coherence_data'
subFolders     = glob(os.path.join(subPath, 'sub-*'))

excel_path  = r'D:\switch_coherence_data\ICA_check_after_notch_filter.xlsx'

# open behavioral file here
beh_path    = r'D:\switch_coherence_data\beh_pkl' # where the data saved in psychopy is

components             = read_components(excel_path)
nSubjects              = len(subFolders)

all_subjects_switch    = []
all_subjects_nonswitch = []

all_subjects_switch_EEG    = []
all_subjects_nonswitch_EEG = []

all_ns_epochs = [] # run condition, not trial types
all_sw_epochs = [] # run condition, not trial types

subFolders.remove(subFolders[7])

for sub in subFolders:   
    subName    = sub.split('/')[-1]
    subName    = subName.split('-')[-1]
    subn       = int(subName)
    eventFiles = glob(os.path.join(subPath, 'sub-' + subName, '*'))
    session    = eventFiles[0].split('-')[-1]  
   
    behfile_path =  glob(os.path.join(beh_path,f'{subName}.pkl'))[0]
     
    # behav results
    beh_results_dict = load_pickle_dict(behfile_path)
    
    # run number (1,2)
    for ri in range(1, 3):
        
        # EEG
        bids_path       = BIDSPath(root = subPath,
                               subject = subName, 
                               session = session,
                               task    = 'Main',
                               run     = ri,
                               extension = 'fif', 
                               datatype  ='eeg')

            
        bids_ica_path   = BIDSPath(root    = subPath,
                           subject = subName, 
                           session = session,
                           task    = 'Main',
                           run     = ri,
                           description = 'ICSolutions',
                           extension = 'fif')
    
        rej_components  = components[(components['subject'] == subn) & (components['run']==ri)].reset_index()
        # function will average rereference the fif data, remove components with ica.exclude and ica.apply
        clean_data, _   = remove_components(bids_path, bids_ica_path, rej_components['component'][0]) 
        
        # interpolate bad channels
        clean_data.interpolate_bads(reset_bads=True)

        #clean_data.plot() # have this plot for later
        events, events_dict = mne.events_from_annotations(clean_data) # get events from your new clean data
        
        # epoch data 
        with mne.use_log_level("WARNING"):  # hide some irrelevant info messages
            
            events     = events[events[:,2]==events_dict['stim_onset'],:]
            event_dict = {'stim_onset' : events_dict['stim_onset']}
            
            # check EEG events
            plt.figure()
            plt.eventplot(events[:,0], lineoffsets=0, color='r')
        
            # make sure all eeg events align with real trials (via beh data onsets) and missing markers are added
            clean_data = check_events_with_behavior(clean_data, beh_results_dict, session, subn, ri)
            
            #clean_data.plot() # uncomment to check that it matches prior clean_data plot
            events, events_dict = mne.events_from_annotations(clean_data)
            plt.eventplot(events[:,0], lineoffsets=0, color='r')

            
            
            events     = events[events[:,2]==events_dict['stim_onset'],:]
            event_dict = {'stim_onset' : events_dict['stim_onset']}
            
            epochs = mne.Epochs(
                clean_data, 
                events, 
                tmin     =-0.05, # update to crop here
                tmax     =0.45, 
                event_id = event_dict, 
                preload  =True,
                reject_by_annotation=False,
                event_repeated='error'
            )
        
        # Downsample with polyphase in future
        # mne.Epochs.resample(sfreq   = 250,
        #                           method  = "polyphase",
        #                           verbose = True,))
        # Decimate
        # mne.Epochs.decimate()
  
        for block in range(1,3):
            
            subName_tmp = session + 'S'+ str(subn) # for the sake of finding the key in beh data
            block_idx = block
            dict_key = f'MainRun_Run-{ri}_Block-{block_idx}_{subName_tmp}'
            
            #load behavioral data for block
            beh_data            = beh_results_dict[dict_key][2]
            ntrials_block   = beh_data.shape[0] # number of trials in block (it may vary a bit)
            
            if len(epochs)>840: # can remove later
                
                if block == 1:
                    start_idx = 1
                    ntrials_first_block = ntrials_block + start_idx
                elif block == 2: 
                    start_idx = ntrials_first_block + 1  
            else:
                if block == 1: # this is what is needed
                    start_idx = 0
                    ntrials_first_block = ntrials_block + start_idx
                elif block == 2: 
                    start_idx = ntrials_first_block 

            if beh_results_dict[dict_key][1] == 'non-switch':
                #beh data
                all_subjects_nonswitch.append(beh_data)
                # eeg data
                all_subjects_nonswitch_EEG.append(epochs['stim_onset'][start_idx:start_idx+ntrials_block])
                          
            elif beh_results_dict[dict_key][1] == 'switch':
                #beh
                all_subjects_switch.append(beh_data)
                # eeg
                all_subjects_switch_EEG.append(epochs['stim_onset'][start_idx:start_idx+ntrials_block]) # this will save it twice, since both blocks in EEG are together


#%% same task repeats (within switch)

nSubjects = len(subFolders)
subjectIndices = np.arange(start=0, stop=2*nSubjects, step=2)
nblocks        = 2

all_switch    = []
all_repeat    = []

for sub in subjectIndices:    
    
    sw_subject_data = all_subjects_switch_EEG[sub:sub+2]
    sw_beh_data     = all_subjects_switch[sub:sub+2]
    
    tmp_sw_faces      = []
    tmp_sw_social     = []
    tmp_sw_gabor      = []
    tmp_sw_mov_dot    = []
    tmp_sw_words      = []
    tmp_sw_mov_sound  = []
    tmp_sw_pitches    = []
      
    tmp_v_repeat   = []
    tmp_a_repeat   = []
    tmp_vmod_switch    = []
    tmp_amod_switch    = []
        
    vision_tasks      = (1,2,3,4)
    aud_tasks         = (5,6,7)
    
    for bidx in range(nblocks):
        
        assert(sw_beh_data[bidx].shape[0]==len(sw_subject_data[bidx]))
    
        trialType           = sw_beh_data[bidx]['trialType'].astype(int)
        
        
        tmp_sw_faces.append(sw_subject_data[bidx][trialType==1])
        tmp_sw_social.append(sw_subject_data[bidx][trialType==2])
        tmp_sw_gabor.append(sw_subject_data[bidx][trialType==3])
        tmp_sw_mov_dot.append(sw_subject_data[bidx][trialType==4])
        tmp_sw_words.append(sw_subject_data[bidx][trialType==5])
        tmp_sw_mov_sound.append(sw_subject_data[bidx][trialType==6])
        tmp_sw_pitches.append(sw_subject_data[bidx][trialType==7])
        
        # take repeat tasks
        for index, trial in enumerate(trialType):
             
        
            next_trial = index + 1
            
            if next_trial <= 419:
               
                # vision * repeats
                if np.any(trialType[next_trial] == vision_tasks) and np.any(trialType[index] == vision_tasks):
                    
                    tmp_v_repeat.append(sw_subject_data[bidx][trialType[next_trial]])
                
                # aud  * repeats
                elif np.any(trialType[next_trial] == aud_tasks) and np.any(trialType[index] == aud_tasks):
                    
                    tmp_a_repeat.append(sw_subject_data[bidx][trialType[next_trial]])
                
                # switch aud --> vision * domain switch 
                elif np.any(trialType[next_trial] == vision_tasks) and np.any(trialType[index] == aud_tasks):
                     
                    tmp_vmod_switch.append(sw_subject_data[bidx][trialType[next_trial]])
                
                # switch vision --> aud * domain switch
                elif np.any(trialType[next_trial] == aud_tasks) and np.any(trialType[index] == vision_tasks):
                     
                    tmp_amod_switch.append(sw_subject_data[bidx][trialType[next_trial]])
                    
    all_switch.append(mne.concatenate_epochs(tmp_amod_switch + tmp_vmod_switch))
    all_repeat.append(mne.concatenate_epochs(tmp_v_repeat + tmp_a_repeat))  
    

#%% PSDs and spectrograms 
# sw_db_power_vrepeat_switch      = [] for later
# sw_db_power_arepeat_switch      = []

# sw_db_power_vtask_switch     = []
# sw_db_power_atask_switch     = []

# sw_db_power_vmod_switch      = []
# sw_db_power_amod_switch      = [] 

pad_sec               = 0.25
sfreq                 = clean_data.info['sfreq']

pad_samples           = int(round(pad_sec*sfreq))
# ---- repeats
     
freqs = np.linspace(6, 70, 50) # - 6-70Hz
p_all_repeat = []    
for sub in range(len(all_repeat)): 
    
    ep_sw_rp_tmp = all_repeat[sub]

    #n_cycles      = 3  
    
    dat           = ep_sw_rp_tmp.get_data()
    padded_data   = np.pad(dat, pad_width=((0,0),(0,0),(pad_samples, pad_samples)),
                                           mode='reflect')
    
    # Welch PSD using padded data
    from scipy.signal import welch
    f, p = welch(padded_data, 
                 fs     = sfreq, # downsampled
                 nperseg=int(0.5 * sfreq), # 500 ms pieces 
                 scaling='density') # give me uV^2/Hz
    p_all_repeat.append(p.mean(0)) # average over trials
    
p_all_repeat = np.stack(p_all_repeat)
   
   
    # ignore below, for later
#     dat_tf      = mne.time_frequency.tfr_array_morlet(data=padded_data,
#                                                           sfreq=sfreq,
#                                                           freqs=freqs, 
#                                                           n_cycles=n_cycles, 
#                                                           n_jobs=-1,
#                                                           output='power')
    
#     dat_tf      = dat_tf[:,:,:, pad_samples:-pad_samples]
    
#     mean_epochs = np.mean(dat_tf, axis=3)
#     std_epochs  = np.std(dat_tf, axis=3)
    
#     z_score_tf  = (dat_tf-mean_epochs[:,:,:,np.newaxis])/std_epochs[:,:,:,np.newaxis]
#     # ! normalize instead to whole recording
#     db_pow      = 10*np.log10(dat_tf/mean_epochs[:,:,:,np.newaxis])
            
# # store time, freq, power. put them in list of task containing a list of subjects 
#     if taskidx == 0:
#         sw_db_power_vrepeat_switch.append(db_pow.mean(0))
#     elif taskidx == 1:
#         sw_db_power_arepeat_switch.append(db_pow.mean(0))

# ---- switch
p_all_switch = []
for sub in range(len(all_switch)): 
    
    ep_sw_tmp = all_switch[sub]

    #n_cycles      = 3  
    
    dat           = ep_sw_tmp.get_data()
    padded_data   = np.pad(dat, pad_width=((0,0),(0,0),(pad_samples, pad_samples)),
                                           mode='reflect')
    # Welch PSD using padded data
    f, p = welch(padded_data, 
                 fs     = sfreq, # downsampled
                 nperseg=int(0.5 * sfreq), # 500 ms pieces 
                 scaling='density') # give me uV^2/Hz
    p_all_switch.append(p.mean(0)) # average over trials
    
p_all_switch = np.stack(p_all_switch)

# all subject, channel average
plt.semilogy(f, p_all_repeat.mean(axis=(0,1)), label = 'repeat') # average over subjects and channels
plt.semilogy(f, p_all_switch.mean(axis=(0,1)), label = 'switch') # average over subjects and channels
plt.xlabel("Frequency (Hz)")
plt.ylabel("uV$^2$/Hz")
plt.legend()
plt.title('All subject and channels average')

# individual channels 
for chidx in range(64):
    plt.figure()
    plt.semilogy(f, p_all_repeat[:,chidx,:].mean(axis=(0)), label='repeat') # average over subs, in one channel
    plt.semilogy(f, p_all_switch[:,chidx,:].mean(axis=(0)), label='switch') # average over subs, in one channel
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("uV$^2$/Hz")       
    plt.legend()
    plt.title(clean_data.ch_names[chidx])

# plot raw trace to see if there is oscillatory activity
clean_data.plot()          
       # dat_tf      = mne.time_frequency.tfr_array_morlet(data=padded_data,
       #                                                       sfreq=sfreq,
       #                                                       freqs=freqs, 
       #                                                       n_cycles=n_cycles, 
       #                                                       n_jobs=-1,
       #                                                       output='power')
       
       # dat_tf      = dat_tf[:,:,:, pad_samples:-pad_samples]
       
       # mean_epochs = np.mean(dat_tf, axis=3)
       # std_epochs  = np.std(dat_tf, axis=3)
       
       # z_score_tf  = (dat_tf-mean_epochs[:,:,:,np.newaxis])/std_epochs[:,:,:,np.newaxis]
       # db_pow      = 10*np.log10(dat_tf/mean_epochs[:,:,:,np.newaxis])
               
       # # store time, freq, power. put them in list of task containing a list of subjects 
       # if taskidx == 0:
       #     sw_db_power_vtask_switch.append(db_pow.mean(0))
       # elif taskidx == 1:
       #     sw_db_power_atask_switch.append(db_pow.mean(0))
           

# things below are tbd for later
#%% all stim epochs average channels

# all_sw_repeats = np.stack([sw_db_power_vrepeat_switch,
#                            sw_db_power_arepeat_switch])

# all_task_switch = np.stack([sw_db_power_vtask_switch, 
#                             sw_db_power_atask_switch])

# all_mod_switch = np.stack([sw_db_power_vmod_switch,
#                            sw_db_power_amod_switch])

# all_sw_repeats = all_sw_repeats.mean(axis=2)
# all_task_switch = all_task_switch.mean(axis=2)
# all_mod_switch = all_mod_switch.mean(axis=2)

# epochs_power    = np.concatenate([all_task_switch, all_sw_repeats], axis=0) 
# epochs_power    = np.swapaxes(np.asarray(epochs_power), 1, 0)

# # so we have subjects × conditions × observations
# # where the time-frequency observations are freqs × times:
# #%% ANOVA
# n_switch_type   = 2
# n_modtype = 2

# factor_levels   = [n_switch_type, n_modtype]  # number of levels in each factor (conditions x stim type)
# effects         = "A*B"  # this is abbreviated form to look for factor A, B and interaction A*B

# n_freqs         = len(freqs)
# times           = 1e3 * ep_sw_rp_tmp.times
# n_times         = len(times)

# fvals, pvals = f_mway_rm(epochs_power, factor_levels, effects=effects)

# effect_labels = ['task switch', 'modality type', 'task switch x modality type']
# fig, axes = plt.subplots(3, 1, figsize=(6, 6), layout="constrained")

# # let's visualize our effects by computing f-images
# for effect, sig, effect_label, ax in zip(fvals, pvals, effect_labels, axes):
#     # show naive F-values in gray
#     ax.imshow(
#         effect,
#         cmap="gray",
#         aspect="auto",
#         origin="lower",
#         extent=[times[0], times[-1], freqs[0], freqs[-1]], alpha=0.2
#     )
#     # create mask for significant time-frequency locations
#     effect[sig >= 0.05] = np.nan
#     c = ax.imshow(
#         effect,
#         cmap="jet",
#         aspect="auto",
#         origin="lower",
#         extent=[times[0], times[-1], freqs[0], freqs[-1]],vmin=0, vmax=20,
#     )
#     fig.colorbar(c, ax=ax)
#     ax.set_xlabel("Time (ms)")
#     ax.set_ylabel("Frequency (Hz)")
#     ax.set_title(f'Time-locked response for "{effect_label}"')
    
# #%%
# all_channels       = clean_data.info["ch_names"]
# all_channels       = mne.pick_channels(all_channels, include=all_channels)
   
# frontal_midline    = ['Fp1','Fz','Fp2','FC1','FC2']
# frontal_midline    = mne.pick_channels(clean_data.info["ch_names"], include=frontal_midline)

# anterior_frontal   = ['AF3','AF4','AF7','AF8','F3','F4']#'F1','F2']
# anterior_frontal   = mne.pick_channels(clean_data.info["ch_names"], include=anterior_frontal)

# left_temporal      = ['F7','FT7','FC5','T7','C5']
# left_temporal      = mne.pick_channels(clean_data.info["ch_names"], include=left_temporal)

# right_temporal     = ['F8','FT8','FC6','C6','T8']
# right_temporal     = mne.pick_channels(clean_data.info["ch_names"], include=right_temporal)

# posterior_central  = ['C1','Cz','C2']
# posterior_central  = mne.pick_channels(clean_data.info["ch_names"], include=posterior_central)

# parietal_occipital = ['CP3','CP1','CPz','CP2','CP4','P3','P1','Pz','P2','P4','PO3','POz','PO4']
# parietal_occipital = mne.pick_channels(clean_data.info["ch_names"], include=parietal_occipital)

# #%% rois based on topo plots
# early_increase_frontcent = ['Cz','C2','FC2','FC4','F4','CPz','Fz']
# early_increase_frontcent = mne.pick_channels(clean_data.info["ch_names"], include=early_increase_frontcent)

# early_increase_occip  = ['O1','PO3','PO7','PO9','P7','P5','O2','PO4','PO8','PO10','P8','P6']
# early_increase_occip = mne.pick_channels(clean_data.info["ch_names"], include=early_increase_occip)

# late_decrease_frontcent  = ['Fp1','Fp2','AF3','AF4','Cz','CP2','CPz']
# late_decrease_frontcent  = mne.pick_channels(clean_data.info["ch_names"], include=late_decrease_frontcent)

# late_decrease_occip      = ['O2', 'PO7','PO4','PO8','P8','P6']
# late_decrease_occip      = mne.pick_channels(clean_data.info["ch_names"], include=late_decrease_occip)

# #%% cb permutation testing 
# plt.rcParams.update({'font.size': 20})

# ch_label    = 'all_channels'
# all_sw_repeats = np.stack([sw_db_power_vrepeat_switch,
#                            sw_db_power_arepeat_switch])

# all_task_switch = np.stack([sw_db_power_vtask_switch, 
#                             sw_db_power_atask_switch])

# all_mod_switch = np.stack([sw_db_power_vmod_switch,
#                            sw_db_power_amod_switch])

# all_sw_repeats = all_sw_repeats[:,:, all_channels, :,:].mean(2)
# all_mod_switch = all_mod_switch[:,:, all_channels, :,:].mean(2)
# all_task_switch = all_task_switch[:,:, all_channels, :,:].mean(2)

# #%% look at channel/ROI specific PSDs in welch
# #%%
# epochs_power    = np.concatenate([all_task_switch, all_sw_repeats], axis=0) 
# full_model     = "A*B" # this means A + B + A:B
# n_subjects = 12

# effect_of_interest = "A"

# def stat_fun(*args):
#     return f_mway_rm(
#         np.swapaxes(args, 1, 0),
#         factor_levels=factor_levels,
#         effects=effect_of_interest,
#         return_pvals=False,
#     )[0]


# # The ANOVA returns a tuple f-values and p-values, we will pick the former.
# pthresh  = 0.01  # set threshold rather high to save some time
# f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, full_model, pthresh)

# tail     = 1  # f-test, so tail > 0
# n_permutations = 2000  # Save some time (the test won't be too sensitive ...)

# F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
#     epochs_power,
#     stat_fun=stat_fun,
#     threshold=f_thresh[0],
#     tail=tail,
#     n_jobs=None,
#     n_permutations=n_permutations,
#     buffer_size=None,
#     out_type="mask",
#     seed=0,
# )

# good_clusters = np.where(cluster_p_values < 0.05)[0]
# F_obs_plot = np.full_like(F_obs, np.nan)
# for ii in good_clusters:
#     F_obs_plot[clusters[ii]] = F_obs[clusters[ii]]

# fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
# for f_image, cmap in zip([F_obs, F_obs_plot], ["gray", "jet"]):
#     c = ax.imshow(
#         f_image,
#         cmap=cmap,
#         aspect="auto",
#         origin="lower",
#         extent=[times[0], times[-1], freqs[0], freqs[-1]],vmax=20
#     )

# # outline sig clusters on diff spectrogram
# for ii in good_clusters:
#     plt.contour(times, freqs, clusters[ii].astype(float), levels=[0,5], linewidth=30, colors='w')
    
# #%% plot cluster over spectrogram
# plt.imshow(
#     np.mean(all_mod_switch-all_sw_repeats,(0,1)),
#     cmap='jet',
#     aspect="auto",
#     origin="lower",
#     extent=[times[0], times[-1], freqs[0], freqs[-1]], vmin = -0.3, vmax = 0.3)
# plt.colorbar(label= 'power (dB)')

# # outline sig clusters on diff spectrogram
# for ii in good_clusters:
#     plt.contour(times, freqs, clusters[ii].astype(float), levels=[0,5], linewidths=5, colors='w')
    
# plt.xlabel("Time (ms)")
# plt.ylabel("Frequency (Hz)")

# plt.title(
#     'Time-locked response for late_decrease_occip\n'
#     " effect of condition cluster-level corrected (p <= 0.05)"
# )

# #%% average power and difference
# plt.rcParams.update({'font.size': 20})

# ch_label    = 'parietal_occipital'
# all_sw = np.stack([sw_db_power_face, sw_db_power_social, sw_db_power_gabor, 
#                    sw_db_power_mov_dot, sw_db_power_words, sw_db_power_mov_sound, 
#                    sw_db_power_pitches])

# all_ns = np.stack([ns_db_power_face, ns_db_power_social, ns_db_power_gabor, 
#                    ns_db_power_mov_dot, ns_db_power_words, ns_db_power_mov_sound, 
#                    ns_db_power_pitches])

# epochs_power_diff    = (all_sw[:,:, anterior_frontal, :,:] - all_ns[:,:, anterior_frontal, :,:]).mean(2)
# all_sw = all_sw[:,:, parietal_occipital, :,:].mean(2)
# all_ns = all_ns[:,:, parietal_occipital, :,:].mean(2)
