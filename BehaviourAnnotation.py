import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings

import BehaviourFeatureExtractor as BF
from BehaviourFeatureExtractor import convert_seconds_to_frame


def find_pickup_point(experiment_data, mouse_id, day, trial_num, config_BF):

    df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"].copy()
    trial_pickup_time = df_summary.loc[df_summary[config_BF["DLC_summary_columns"]["trial_num"]] == trial_num,
                                                 config_BF["DLC_summary_columns"]["mouse_first_pick_up"]].values[0]
    trial_start_time = df_summary.loc[df_summary[config_BF["DLC_summary_columns"]["trial_num"]] == trial_num,
                                                 config_BF["DLC_summary_columns"]["pup_displacement"]].values[0]
    trial_end_time = df_summary.loc[df_summary[config_BF["DLC_summary_columns"]["trial_num"]] == trial_num,
                                                 config_BF["DLC_summary_columns"]["trial_end"]].values[0]

    return trial_pickup_time, trial_start_time, trial_end_time

def label_pup_interaction_behaviors(experiment_data, mouse_id, day, trial_num,
                                  event_time_point, config_BF, BF_instance, kernel_size=20,
                                  pre_event_window_size_time=10, frame_rate=30,):
    """Labels mouse behaviors (approach, crouching, active interaction) in the time window before pup pickup"""
    
    # Get required data from experiment_data
    df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"].copy()    
    trial_DLC = experiment_data[mouse_id][day]["trials"][trial_num]["dlc_data"].copy()
    
    # conversion to frames
    event_time_point_frame = convert_seconds_to_frame(event_time_point, frame_rate)
    pre_event_window_size_frames = convert_seconds_to_frame(pre_event_window_size_time, frame_rate)

    trial_start_frame_index = trial_DLC["frame_index"].iloc[0]
    start_frame_index = event_time_point_frame - pre_event_window_size_frames

    if start_frame_index < trial_start_frame_index:
        warnings.warn(f"Window is too big, spills over trial start, resetting window size to {trial_start_frame_index - start_frame_index}")
        pre_event_window_size_frames = trial_start_frame_index - start_frame_index

    # Get window of interest
    window = trial_DLC.loc[start_frame_index:event_time_point_frame].copy()

    # Calculate distance to pup center
    pup_corner_bounds = BF_instance.extract_pup_starting_position_bounds(df_summary, trial_num)
    pup_center_x = pup_corner_bounds["xmin"] + (pup_corner_bounds["xmax"] - pup_corner_bounds["xmin"]) / 2
    pup_center_y = pup_corner_bounds["ymin"] + (pup_corner_bounds["ymax"] - pup_corner_bounds["ymin"]) / 2
    window["distance_mouse_pup_center"] = np.sqrt((window["mouse_x"] - pup_center_x)**2 + (window["mouse_y"] - pup_center_y)**2)
    window["distance_head_pup_center"] = np.sqrt((window["head_x"] - pup_center_x)**2 + (window["head_y"] - pup_center_y)**2)

    # Convert distances to cm and calculate derivatives
    columns = [config_BF["DLC_behaviour_columns"]["mouse_speed"], "distance_mouse_pup_center", "distance_head_pup_center"]
    px_cm_ratio = config_BF["pixels_to_cm_ratio"]
    for col in columns:
        window[col+"_cm"] = window[col] / px_cm_ratio
        window[col+"_cm_deriv"] = np.gradient(window[col+"_cm"])

    # Smooth derivatives
    kernel = np.ones(kernel_size) / kernel_size
    window["mouse_speed_px/s_cm_convolved"] = np.convolve(window["mouse_speed_px/s_cm"], kernel, mode='same')
    window["mouse_speed_px/s_cm_deriv_convolved"] = np.gradient(window["mouse_speed_px/s_cm_convolved"])
    window["distance_mouse_pup_center_cm_deriv_convolved"] = np.convolve(window["distance_mouse_pup_center_cm_deriv"], kernel, mode='same')

    # Label behaviors
    mask_approach = (window["distance_mouse_pup_center_cm_deriv_convolved"] < -0.1)
    mask_crouching = (window["distance_mouse_pup_center_cm"] < 2) & (window["mouse_speed_px/s_cm_convolved"] < 5)
    mask_active_interaction = (window["distance_head_pup_center_cm"] < 2) & (window["mouse_speed_px/s_cm_convolved"] < 5)
        
    # mask approach, crouching and active interaction
    mask_crouching[mask_active_interaction] = False
    mask_approach[mask_crouching | mask_active_interaction] = False
    window["in_nest"][mask_approach | mask_crouching | mask_active_interaction] = False
    
    window["approach"] = mask_approach
    window["crouching"] = mask_crouching
    window["active_interaction"] = mask_active_interaction

    return window

def plot_pup_usv_to_pickup_point(experiment_data, mouse_id, day, trial_num, window=None, config_BF=None, BF_instance=None):
    """Plots the labeled behaviors and USV data leading up to pup pickup"""
    
    #df_summary = experiment_data[ms_id][d]["Behavior"]["df_summary"].copy()
    #trials = experiment_data[ms_id][d]["trials"].copy()
    #trial_DLC = trials[trial_num]["dlc_data"].copy()
    
    # clearing the plot
    plt.close('all')

    trial_pickup_time, trial_start_time, trial_end_time = find_pickup_point(experiment_data, mouse_id, day, trial_num, config_BF)
    trial_pickup_time_minutes = f"{str(int(trial_pickup_time//60))}:{int(trial_pickup_time%60)}"
    trial_start_time_minutes = f"{str(int(trial_start_time//60))}:{int(trial_start_time%60)}"

    if window is None:
        window = label_pup_interaction_behaviors(experiment_data, mouse_id, day, trial_num,
                                                trial_pickup_time, config_BF=config_BF, BF_instance=BF_instance)  
    else:
        window = window.copy()                                         
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    window.plot(x="time_seconds", y=["distance_mouse_pup_center_cm"], color="blue", ax=ax)

    # Add behavior shading
    for behavior, color in [("crouching", "red"), ("approach", "green"), 
                          ("in_nest", "cyan"), ("active_interaction", "blue")]:
        for i in range(len(window[behavior])):
            if window[behavior].iloc[i]:
                ax.axvspan(window["time_seconds"].iloc[i],
                          window["time_seconds"].iloc[i]+1/30,
                          facecolor=color, alpha=0.5)

    # Add pickup point indicator
    # trial_pickup_time = window["time_seconds"].iloc[-1]
    ax.axvspan(trial_pickup_time, trial_pickup_time + 1/30,
               facecolor='magenta', alpha=0.5)

    # Add USV plot
    ax2 = ax.twinx()
    window.plot(x="time_seconds", y=["average_frequency"],
                ax=ax2, kind="scatter", color="purple")
    ax2.set_ylabel("Average frequency of USVs (kHz)", color="purple")

    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='approach', alpha=0.5),
        mpatches.Patch(color='red', label='crouch', alpha=0.5),
        mpatches.Patch(color='cyan', label='In nest', alpha=0.5),
        mpatches.Patch(color='blue', label='Active interaction', alpha=0.5),
        mpatches.Patch(color='magenta', label='Pick up point', alpha=0.5),
        mlines.Line2D([], [], color='blue', label='Distance to pup')
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_title(f"{mouse_id} - {day} - Trial {trial_num}: Distance to pup and USV to pick up point at {trial_pickup_time_minutes}, trial start was at {trial_start_time_minutes}")
    #plt.show()

    return {"active_interaction_time": window["active_interaction"].sum()/len(window),
            "crouching_time": window["crouching"].sum()/len(window),
            "approach_time": window["approach"].sum()/len(window)}