import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
import os
from BehaviourFeatureExtractor import *
from BehaviourAnnotation import *
import pprint


### Video loading and extraction functions ###
def load_video_segment(video_path, start_time=None, end_time=None):
    """
    Load a segment of a video file between start_time and end_time.
    
    Parameters:
        video_path (str): Path to the video file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        
    Returns:
        tuple: (video_array, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # If times are not specified, use None values
    if start_time is not None:
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0
        
    if end_time is not None:
        end_frame = int(end_time * fps)
    else:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate number of frames to read
    n_frames = end_frame - start_frame
    
    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read video file")
    
    height, width, channels = frame.shape
    
    # Initialize array only for the segment we want
    video_array = np.empty((n_frames, height, width, channels), dtype=np.uint8)
    video_array[0] = frame
    
    # Read only the frames we need
    frame_idx = 1
    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        video_array[frame_idx] = frame
        frame_idx += 1
    
    cap.release()
    
    return video_array

def extract_trial_video_efficient(video_path, experiment_data, mouse_id, day,
                                trial_num=None, start_time=None, end_time=None, config_BF=None):
                                  
    # If start and end times not specified, get them from experiment_data
    if start_time is None or end_time is None:
        if trial_num is None:
            raise ValueError("Must provide either start_time/end_time or trial_num")
        df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"]
        start_time = df_summary.loc[df_summary[config_BF["DLC_summary_columns"]["trial_num"]] == trial_num, config_BF["DLC_summary_columns"]["pup_displacement"]].values[0]
        end_time = df_summary.loc[df_summary[config_BF["DLC_summary_columns"]["trial_num"]] == trial_num, config_BF["DLC_summary_columns"]["trial_end"]].values[0]

    # Load only the video segment we need
    video_array = load_video_segment(video_path, 
                                        start_time=start_time,
                                        end_time=end_time)
    
    return video_array

def save_trial_video(trial_video, output_path, fps = 30):
    height, width = trial_video.shape[1:3]
    
    # Use H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in trial_video:
        out.write(frame.astype(np.uint8))
    
    # Release the video writer
    out.release()


#### Applying the whole pipeline on full experiment data
def run_video_creation(video_dict, 
                      experiment_data,
                      mouse_id, day, trial_num, config_BF, BF):  
    """Run the full video creation pipeline for a specific trial.
    
    This function processes a trial video by:
    1. Loading the video and finding the pup pickup point
    2. Labeling mouse behaviors (approach, crouching, active interaction) in the pre-pickup window
    3. Extracting the relevant video segment
    4. Annotating the video with behavior shading
    5. Saving the annotated video
    
    Args:
        video_dict (dict): Dictionary mapping mouse_id and day to video file paths
        experiment_data (dict): Nested dictionary containing all experimental data
        mouse_id (str): ID of the mouse
        day (str): Day of the experiment
        trial_num (int): Trial number to process
        config_BF (dict): Configuration dictionary containing column names and parameters
        BF: BehaviorFeatureExtractor instance for processing behavioral data
        
    Returns:
        None. Saves the annotated video to disk.
    """
    ## 0. Load the video path
    video_path = video_dict[mouse_id][day]
    print("0. Video loaded from:", video_path)

    ## 1. Labeling the behaviors
    pickup_time, start_time, end_time = find_pickup_point(experiment_data,
                                                         mouse_id, day, trial_num, 
                                                         config_BF)
    
    window = label_pup_interaction_behaviors(experiment_data,
                                           mouse_id, day, trial_num,
                                           event_time_point=pickup_time,
                                           config_BF=config_BF,
                                           BF_instance=BF,
                                           kernel_size=20,
                                           pre_event_window_size_time=10,
                                           frame_rate=30)
    print("1. Behaviors labeled")
    print(f"Start time: {start_time} End time: {end_time}")   
    print(f"Pickup time: {pickup_time} in mins: {str(int(pickup_time//60))}:{int(pickup_time%60)}")

    # Create plot for labeled window
    behavior_stats = plot_pup_usv_to_pickup_point(experiment_data,
                                                 mouse_id, day, trial_num, 
                                                 window=window, 
                                                 config_BF=config_BF)
    print(behavior_stats)

    ## 2. Extract a specific trial
    trial_video = extract_trial_video_efficient(video_path, experiment_data,
                                              mouse_id, day, trial_num,
                                              start_time, end_time, config_BF)
    print("2. Trial extracted")

    ## 3. Annotating the video
    # Get the frame index for pickup
    pickup_frame_index = window[config_BF["DLC_columns"]["frame_index"]].iloc[-1]

    # Apply shading
    shaded_video = apply_behavior_shading(
        trial_video,
        window,
        pickup_frame_index,
        alpha=0.3,
        pickup_window_size=5
    )
    print("3. Video annotated")

    ## 4. Save the video
    original_path = video_dict[mouse_id][day]
    movie_name = f"{mouse_id}_{day}_{trial_num}_pickup_shaded.mp4"
    path_to_save = original_path.split("/")[:-1] + [movie_name]

    image_name = f"{mouse_id}_{day}_{trial_num}_pickup_annotated.png"
    plot_path = original_path.split("/")[:-1] + [image_name]

    path_to_save = "/".join(path_to_save)
    plot_path = "/".join(plot_path)

    print("4. Video saved to:", path_to_save)
    print("4. Plot saved to:", plot_path)

    save_trial_video(shaded_video, output_path=path_to_save, fps=30)
    plt.savefig(plot_path)

def truncate_video(video_path, output_dir=None, frames_to_skip=150):
    """
    Removes the first N frames from a video and saves it with 'truncated150_' prefix
    
    Args:
        video_path: path to the original video file
        output_dir: directory to save the truncated video (defaults to same as input)
        frames_to_skip: number of frames to remove from start (default 150)
    
    Returns:
        str: path to the truncated video file
    """
    # Get video filename and directory
    video_dir = os.path.dirname(video_path) if output_dir is None else output_dir
    video_name = os.path.basename(video_path)
    output_path = os.path.join(video_dir, f"truncated150_{video_name}")
    
    # Load video data efficiently using load_video_segment
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    video_array = load_video_segment(video_path, 
                                   start_time=frames_to_skip/30,  # Assuming 30 fps
                                   end_time=total_frames/30)
    
    # Save the truncated video
    save_trial_video(video_array, output_path, fps=30)
    
    return output_path

def batch_truncate_videos(video_dict):
    """
    Truncates all videos in the video dictionary by removing the first 150 frames
    
    Args:
        video_dict: dictionary of videos organized by mouse_id and day
        
    Returns:
        dict: dictionary with paths to truncated videos
    """
    truncated_videos = {}
    
    for mouse_id, days in video_dict.items():
        truncated_videos[mouse_id] = {}
        for day, video_path in days.items():
            print(f"Truncating video for {mouse_id} {day}...")
            try:
                truncated_path = truncate_video(video_path)
                truncated_videos[mouse_id][day] = truncated_path
                print(f"Saved truncated video to: {truncated_path}")
            except Exception as e:
                print(f"Error processing video for {mouse_id} {day}: {str(e)}")
                truncated_videos[mouse_id][day] = None
    
    return truncated_videos

if __name__ == "__main__":
    """
    Script to truncate videos by removing the first 150 frames.
    
    Basic usage:
        # Use default data directory ('data') and config file ('config.json')
        python video_creation.py
        
        # Specify custom data directory
        python video_creation.py --data_dir /path/to/data
        
        # Specify both custom data directory and config file
        python video_creation.py --data_dir /path/to/data --config /path/to/config.json
    
    The script will:
    1. Find all videos in the data directory
    2. Show summary of videos found
    3. Ask for confirmation
    4. Create new videos with 'truncated150_' prefix
    5. Show processing results
    """
    import argparse
    from DataLoader import DataLoader
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Truncate videos by removing first 150 frames')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data (default: data)')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    # Initialize DataLoader to get video dictionary
    print(f"Initializing DataLoader with data directory: {args.data_dir}")
    DL = DataLoader(args.data_dir, path_to_config_file=args.config)

    # filter video_dict to only include videos with correct tags
    video_dict = DL.video_dict.copy()
    
    print("### *** Video_dict *** ###")
    pprint.pprint(video_dict)

    # Print summary of videos found
    total_videos = sum(len(days) for days in video_dict.values())
    print(f"\nFound {total_videos} videos across {len(video_dict)} mice:")
    for mouse_id, days in video_dict.items():
        print(f"- Mouse {mouse_id}: {len(days)} videos")
    
    # Confirm with user
    response = input("\nProceed with truncating all videos? [y/N]: ")
    if response.lower() != 'y':
        print("Aborting...")
        exit()
    
    # Process videos
    print("\nStarting video truncation...")
    truncated_videos = batch_truncate_videos(video_dict)
    
    # Print summary of results
    print("\nTruncation complete!")
    successful = 0
    failed = 0
    for mouse_id, days in truncated_videos.items():
        for day, path in days.items():
            if path is not None:
                successful += 1
            else:
                failed += 1
                print(f"Failed to process video for mouse {mouse_id} day {day}")
    
    print(f"\nSuccessfully truncated {successful} videos")
    if failed > 0:
        print(f"Failed to truncate {failed} videos")
