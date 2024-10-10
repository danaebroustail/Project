import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings('ignore') 

# the following utility functions assume that the input is a processed DLC file.
def compute_speed(df_DLC):

    # compute speed
    distance = np.sqrt(np.diff(df_DLC['msTop_x'])**2 + np.diff(df_DLC['msTop_y'])**2)
    time = np.diff(df_DLC['time_seconds'])
    speed = distance/time
    # add speed to the dataframe
    df_DLC['speed_cm/s'] = np.append(speed, 0)
    
    return df_DLC

def compute_distance_to_pup(df_DLC):
    # compute distance to pup

    distance = np.sqrt((df_DLC['msTop_x'] - df_DLC['pup_x'])**2 + (df_DLC['msTop_y'] - df_DLC['pup_y'])**2)
    df_DLC['distance_to_pup'] = distance 
    return df_DLC

def compute_head_angle_to_pup(df_DLC, add_vector_columns = False):

    # define mouse head direction with respect to average of ears and nose
    earRight_x, earRight_y, earLeft_x, earLeft_y,  = df_DLC['earRight_x'], df_DLC['earRight_y'], df_DLC['earLeft_x'], df_DLC['earLeft_y']
    nose_x, nose_y, pup_x, pup_y = df_DLC['nose_x'], df_DLC['nose_y'], df_DLC['pup_x'], df_DLC['pup_y']

    # compute between ears coordinate
    between_ears_x, between_ears_y = (earRight_x + earLeft_x)/2, (earRight_y + earLeft_y)/2
    # compute vector from nose to between ears
    mouse_dir_vector_x, mouse_dir_vector_y = nose_x - between_ears_x, nose_y - between_ears_y
    pup_dir_vector_x, pup_dir_vector_y = pup_x - between_ears_x, pup_y - between_ears_y

    if add_vector_columns:
        df_DLC['between_ears_x'] = between_ears_x
        df_DLC['between_ears_y'] = between_ears_y
        df_DLC['mouse_dir_vector_x'] = mouse_dir_vector_x
        df_DLC['mouse_dir_vector_y'] = mouse_dir_vector_y
        df_DLC['pup_dir_vector_x'] = pup_dir_vector_x
        df_DLC['pup_dir_vector_y'] = pup_dir_vector_y

    # compute angle between mouse direction and pup direction

    dot_product = [np.dot([mouse_dir_x, mouse_dir_y],
                          [pup_dir_x, pup_dir_y]) for mouse_dir_x, mouse_dir_y, pup_dir_x, pup_dir_y
                   in zip(mouse_dir_vector_x, mouse_dir_vector_y, pup_dir_vector_x, pup_dir_vector_y)]
    
    norm_mouse_dir = [np.linalg.norm([mouse_dir_x, mouse_dir_y]) for mouse_dir_x, mouse_dir_y
                      in zip(mouse_dir_vector_x, mouse_dir_vector_y)]
    
    norm_pup_dir = [np.linalg.norm([pup_dir_x, pup_dir_y]) for pup_dir_x, pup_dir_y
                    in zip(pup_dir_vector_x, pup_dir_vector_y)]
    
    cos_theta = [dot/(norm_mouse_dir*norm_pup_dir) for dot, norm_mouse_dir, norm_pup_dir
                 in zip(dot_product, norm_mouse_dir, norm_pup_dir)]
    
    angle = np.arccos(cos_theta)

    # add angle to the dataframe (as degrees)
    df_DLC['head_angle_to_pup_degrees'] = angle * 180/np.pi
    
    return df_DLC

def extract_base_parameters(df_DLC, df_summary):
    """
    Extracts base parameters such as speed, distance to pup, and head angle to pup for each trial 
    from the given DataFrame.
    Parameters:
    df_DLC (pd.DataFrame): DataFrame containing DeepLabCut (DLC) tracking data with a 'frame_index' column.
    df_summary (pd.DataFrame): DataFrame containing summary information for each trial, including 
                               'BehavRecdTrialEndSecs' and 'PupDispDropSecs' columns.
    Returns:
    pd.DataFrame: Updated DataFrame with additional columns for speed ('speed_cm/s'), 
                  distance to pup ('distance_to_pup'), and head angle to pup ('head_angle_to_pup_degrees').
                  These columns have default NaN values for frames that don't belong to any trial.
    """

    # for each trial, get the start and end frames
    end_times, start_times, trial_nums = df_summary['BehavRecdTrialEndSecs'], df_summary['PupDispDropSecs'], df_summary['TrialNum']

    # create NaN columns for speed, distance to pup and head angle to pup
    df_DLC['speed_cm/s'] = np.nan
    df_DLC['distance_to_pup'] = np.nan
    df_DLC['head_angle_to_pup_degrees'] = np.nan

    # iterate over each trial
    for end, start, trial_num in zip(end_times, start_times, trial_nums):
            # compute frame indices
            end_frame, start_frame = round(end*30), round(start*30)

            print(f"Processing trial {trial_num} Start frame: {start_frame} End frame: {end_frame}")

            # get the data for the trial
            mask = (df_DLC['frame_index'] >= start_frame) & (df_DLC['frame_index'] <= end_frame)
            trial_DLC = df_DLC.loc[mask, :] 

            # compute speed
            trial_DLC = compute_speed(trial_DLC)
            # compute distance to pup
            trial_DLC = compute_distance_to_pup(trial_DLC)
            # compute head angle to pup
            trial_DLC = compute_head_angle_to_pup(trial_DLC)
            
            # update the dataframe
            df_DLC.loc[mask, :] = trial_DLC

    return df_DLC



#### Plotting utility functions #####

def plot_mouse_angle_to_pup(trial_df_DLC,
                            ylim = None,
                            xlim = None,
                            frame_index = None):

    # select a random frame in the trial
    if frame_index is None:
        frame_index_min, frame_index_max = trial_df_DLC['frame_index'].min(), trial_df_DLC['frame_index'].max()
        frame_index = random.randint(frame_index_min, frame_index_max)
    trial_1_DLC_frame = trial_df_DLC.loc[trial_df_DLC['frame_index'] == frame_index, :]

    # plot the frame
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if xlim is None or ylim is None:
        xlim, ylim =  max(trial_df_DLC['msTop_x'].max(), trial_df_DLC['pup_x'].max()), max(trial_df_DLC['msTop_y'].max(), trial_df_DLC['pup_y'].max())

    trial_1_DLC_frame.plot(x='earRight_x', y='earRight_y', style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'black')
    trial_1_DLC_frame.plot(x='earLeft_x', y='earLeft_y', style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'black')
    trial_1_DLC_frame.plot(x='nose_x', y='nose_y', style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'red')
    trial_1_DLC_frame.plot(x='pup_x', y='pup_y', style='o', ax=ax,xlim=(0, xlim), ylim=(0, ylim), color = 'purple')
    trial_1_DLC_frame.plot(x='between_ears_x', y='between_ears_y', style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'blue')

    # draw lines
    # draw line from nose to between ears in blue
    ax.plot([trial_1_DLC_frame['nose_x'].values[0], trial_1_DLC_frame['between_ears_x'].values[0]],
            [trial_1_DLC_frame['nose_y'].values[0], trial_1_DLC_frame['between_ears_y'].values[0]], 'k-', color = 'blue')

    # draw line from between ears to pup in green
    ax.plot([trial_1_DLC_frame['between_ears_x'].values[0], trial_1_DLC_frame['pup_x'].values[0]],
            [trial_1_DLC_frame['between_ears_y'].values[0], trial_1_DLC_frame['pup_y'].values[0]], 'k-', color = 'green')

    # add angle value to the plot in the title
    angle = trial_1_DLC_frame['head_angle_to_pup_degrees'].values[0]

    ax.set_title("Angle between mouse head direction and pup direction: " + str(angle) + " degrees")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")