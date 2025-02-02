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
    print(f"Event time point frame: {event_time_point_frame}, Pre-event window size frames: {pre_event_window_size_frames}")
    print(f"Trial start frame index: {trial_start_frame_index}")

    trial_start_frame_index = trial_DLC["frame_index"].iloc[0]
    start_frame_index = event_time_point_frame - pre_event_window_size_frames

    if start_frame_index < trial_start_frame_index:
        warnings.warn(f"Window is too big, spills over trial start, resetting window size to {trial_start_frame_index - start_frame_index}")
        pre_event_window_size_frames = trial_start_frame_index - start_frame_index

    # Get window of interest
    window_frames = (trial_DLC["frame_index"] >= start_frame_index) & (trial_DLC["frame_index"] <= event_time_point_frame)
    window = trial_DLC.loc[window_frames].copy()

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

if __name__ == "__main__":
    """
    Script to run behavior annotation pipeline on all trials in experiment data.
    
    Basic usage:
        python BehaviourAnnotation.py
        
        # Specify custom data directory and config file
        python BehaviourAnnotation.py --data_dir /path/to/data --config /path/to/config.json
        
    The script will:
    1. Load experiment data for all mice/days
    2. Process each trial:
        - Find pickup point (if exists)
        - Label behaviors
        - Create and save behavior plots
    3. Show summary statistics
    """
    import argparse
    from DataLoader import DataLoader
    from BehaviourFeatureExtractor import BehaviourFeatureExtractor
    import os
    import numpy as np
    import pandas as pd
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run behavior annotation pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data (default: data)')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file (default: config.json)')
    parser.add_argument('--output_dir', type=str, default='behavior_plots',
                        help='Directory to save plots (default: behavior_plots)')
    
    args = parser.parse_args()
    
    # Initialize DataLoader and BehaviourFeatureExtractor
    print(f"Initializing with data directory: {args.data_dir}")
    DL = DataLoader(args.data_dir, path_to_config_file=args.config)
    BF = BehaviourFeatureExtractor(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each mouse and day
    behavior_stats = {}
    failed_trials = []  # Track failed trials
    no_pickup_trials = []  # Track trials without pickup
    all_successful_stats = []  # Track statistics for successful trials
    
    for mouse_id in DL.df_dict:
        print(f"\nProcessing mouse {mouse_id}")
        
        # Get experiment data for this mouse
        experiment_data = DL.get_data_for_experiment(mouse_id)
        
        if not experiment_data:
            print(f"No data found for mouse {mouse_id}")
            continue
            
        behavior_stats[mouse_id] = {}
        
        # Process each day
        for day in experiment_data:
            print(f"\nProcessing day {day}")
            behavior_stats[mouse_id][day] = {}
            
            # Get trial numbers from summary data
            df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"]
            trial_nums = df_summary[BF.config["DLC_summary_columns"]["trial_num"]].unique()
            
            # Process each trial
            for trial_num in trial_nums:
                print(f"\nProcessing trial {trial_num}")
                try:
                    # Find pickup point
                    pickup_time, start_time, end_time = find_pickup_point(
                        experiment_data, mouse_id, day, trial_num, BF.config
                    )
                    
                    # Check if pickup time exists
                    if pd.isna(pickup_time):
                        print(f"No pickup time found for trial {trial_num}")
                        no_pickup_trials.append((mouse_id, day, trial_num))
                        behavior_stats[mouse_id][day][trial_num] = {
                            "status": "no_pickup",
                            "stats": None
                        }
                        continue
                    
                    # Label behaviors
                    window = label_pup_interaction_behaviors(
                        experiment_data, mouse_id, day, trial_num,
                        event_time_point=pickup_time,
                        config_BF=BF.config,
                        BF_instance=BF,
                        kernel_size=20,
                        pre_event_window_size_time=10,
                        frame_rate=30
                    )
                    
                    # Create and save plot
                    stats = plot_pup_usv_to_pickup_point(
                        experiment_data, mouse_id, day, trial_num,
                        window=window,
                        config_BF=BF.config
                    )
                    
                    # Save plot
                    plot_path = os.path.join(args.output_dir, 
                                           f"{mouse_id}_{day}_trial{trial_num}_behaviors.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    # Store statistics
                    stats_entry = {
                        "mouse_id": mouse_id,
                        "day": day,
                        "trial_num": trial_num,
                        "pickup_time": float(pickup_time),
                        **stats  # unpack the behavior statistics
                    }
                    all_successful_stats.append(stats_entry)
                    
                    behavior_stats[mouse_id][day][trial_num] = {
                        "status": "success",
                        "stats": stats,
                        "pickup_time": float(pickup_time)
                    }
                    print(f"Trial statistics: {stats}")
                    
                except Exception as e:
                    print(f"Error processing trial {trial_num}: {str(e)}")
                    failed_trials.append((mouse_id, day, trial_num))
                    behavior_stats[mouse_id][day][trial_num] = {
                        "status": "error",
                        "error": str(e),
                        "stats": None
                    }
    
    # Convert all successful stats to DataFrame for easy analysis
    df_stats = pd.DataFrame(all_successful_stats)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    total_trials = 0
    successful_trials = len(all_successful_stats)
    
    for mouse_id in behavior_stats:
        for day in behavior_stats[mouse_id]:
            total_trials += len(behavior_stats[mouse_id][day])
    
    print(f"\nProcessed {successful_trials}/{total_trials} trials successfully")
    print(f"Failed trials: {len(failed_trials)}")
    print(f"Trials without pickup: {len(no_pickup_trials)}")
    
    if no_pickup_trials:
        print("\nTrials without pickup times:")
        for mouse_id, day, trial_num in no_pickup_trials:
            print(f"- Mouse {mouse_id}, Day {day}, Trial {trial_num}")
    
    # Print aggregate statistics for successful trials
    print("\n=== Aggregate Behavior Statistics ===")
    print("\nMean proportions across all successful trials:")
    for col in ['active_interaction_time', 'crouching_time', 'approach_time']:
        mean_val = df_stats[col].mean()
        std_val = df_stats[col].std()
        print(f"{col}: {mean_val:.3f} ± {std_val:.3f}")
    
    print("\nStatistics by mouse:")
    mouse_stats = df_stats.groupby('mouse_id')[['active_interaction_time', 'crouching_time', 'approach_time']].agg(['mean', 'std'])
    print(mouse_stats)
    
    print("\nStatistics by day:")
    day_stats = df_stats.groupby('day')[['active_interaction_time', 'crouching_time', 'approach_time']].agg(['mean', 'std'])
    print(day_stats)
    
    # Save behavior statistics to file
    import json
    stats_path = os.path.join(args.output_dir, 'behavior_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(behavior_stats, f, indent=4)
    print(f"\nBehavior statistics saved to {stats_path}")
    
    # Save aggregate statistics to CSV
    df_stats.to_csv(os.path.join(args.output_dir, 'aggregate_statistics.csv'), index=False)
    print(f"Aggregate statistics saved to {os.path.join(args.output_dir, 'aggregate_statistics.csv')}")