import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
import json
import BehaviourFeatureExtractor as BF
from BehaviourFeatureExtractor import convert_seconds_to_frame
import pprint
import copy
import os
import networkx as nx


def convert_seconds_to_frame(seconds, frame_rate = 30):
    return round(seconds*frame_rate)

def compute_distance(coords_1, coords_2):
            coords_1_x, coords_1_y = coords_1
            coords_2_x, coords_2_y = coords_2

            return np.sqrt((coords_1_x - coords_2_x)**2 + (coords_1_y - coords_2_y)**2).mean()

def get_pick_up_time(df_summary, trial_num, trial_num_col = "TrialNum", pick_up_time_col = "MouseFirstPickUpPupSecs"):
        """
        Get the time when mouse first picked up pup for a given trial.
        
        Parameters:
        -----------
        df_summary : pandas DataFrame
            DataFrame containing summary data
        trial_num : int
            Trial number
        BF : object
            BehaviorFinder object containing configuration parameters
            
        Returns:
        --------
        float
            Time in seconds when mouse first picked up pup
        """
        pick_up_value = df_summary.loc[df_summary[trial_num_col] == trial_num, pick_up_time_col].values[0]
        
        if pd.isna(pick_up_value):
            return None
        else:
            return pick_up_value

def get_success_retrieval_time(df_summary, trial_num, trial_num_col = "TrialNum",
                                success_col = "TrialDesignAchieved",
                                end_time_col = "BehavRecdTrialEndSecs"):
    """
    Get the time when mouse successfully retrieved pup for a given trial.
    
    Parameters: 
    -----------
    df_summary : pandas DataFrame
        DataFrame containing summary data
    trial_num : int
        Trial number    
    BF : object
        BehaviorFinder object containing configuration parameters

    Returns:
    --------
    float
        Time in seconds when mouse successfully retrieved pup
    """

    success_value = df_summary.loc[df_summary[trial_num_col] == trial_num, success_col].values[0]
    
    if success_value == 1:
        return df_summary.loc[df_summary[trial_num_col] == trial_num, end_time_col].values[0]
    else:
        return None



class BehaviourAnnotator:

    def __init__(self, path_to_config_file):

        # read .json config file for 
        with open(path_to_config_file) as f:
            self.config = json.load(f)

        self.DLC_cols = self.config['DLC_columns']
        self.DLC_summary_cols = self.config['DLC_summary_columns']
        self.DLC_behaviour_cols = self.config['DLC_behaviour_columns']
        self.BF_instance = BF.BehaviourFeatureExtractor("config.json")

        self.time_col = self.DLC_cols['time']
        self.frame_index_col = self.DLC_cols['frame']

        self.frame_rate = self.config['frame_rate_dlc']
        self.minimum_distance_to_nest = self.config['minimum_distance_to_nest']
        self.likelihood_threshold = self.config['likelihood_threshold']
        self.states = self.config['Behavioral_states']
        self.single_event_states = self.config['single_event_states']

        self.pickup_col = self.states["pickup"]
        self.retrieval_col = self.states["retrieval"]

        self.distance_between_clusters_cm = self.config['distance_between_clusters_cm']
        self.pixels_to_cm_ratio = self.config['pixels_to_cm_ratio']
        self.threshold_approach_speed_cms = self.config['threshold_approach_speed_cms']
        self.threshold_approach_angle_degrees = self.config['threshold_approach_angle_degrees']


    def create_default_columns(self, trial_df):
        for state in self.states:
            trial_df[state] = False

        return trial_df

    def mark_existing_times(self, trial_df, df_summary, trial_num):

        pickup_col = self.pickup_col
        retrieval_col = self.retrieval_col

        # get pick up time
        pick_up_time = get_pick_up_time(df_summary, trial_num,
                                trial_num_col = self.DLC_summary_cols["trial_num"],
                                pick_up_time_col = self.DLC_summary_cols["mouse_first_pick_up"])
    
        print(f"* Pick up time: {pick_up_time}")
        # get success retrieval time
        success_retrieval_time = get_success_retrieval_time(df_summary, trial_num,
                                                        trial_num_col = self.DLC_summary_cols["trial_num"],
                                                        success_col = self.DLC_summary_cols["trial_success"],
                                                        end_time_col = self.DLC_summary_cols["trial_end"])
        print(f"* Success time: {success_retrieval_time}")

        if pick_up_time is not None:
            pick_up_frame = convert_seconds_to_frame(pick_up_time)
            print(f"---> Pick up frame: {pick_up_frame}, Start frame: {trial_df['frame_index'].min()}, End frame: {trial_df['frame_index'].max()}")
            trial_df.loc[trial_df["frame_index"] == pick_up_frame, pickup_col] = True

        if success_retrieval_time is not None:
            retrieval_frame = convert_seconds_to_frame(success_retrieval_time)
            print(f"---> Retrieval frame: {retrieval_frame}; Start frame: {trial_df['frame_index'].min()}; End frame: {trial_df['frame_index'].max()}")
            end_frame = trial_df["frame_index"].max()
            final_frame = min(end_frame, retrieval_frame)
            print(f"---> Final frame: {final_frame}")
            trial_df.loc[(trial_df["frame_index"] == final_frame), retrieval_col] = True

        return trial_df

    def assign_pick_up_and_success_types(self, pup_locations, df_summary, trial_num):
        
        pickup_col = self.pickup_col
        retrieval_col = self.retrieval_col

        pick_up_time = get_pick_up_time(df_summary, trial_num,
                                        trial_num_col = self.DLC_summary_cols["trial_num"],
                                        pick_up_time_col = self.DLC_summary_cols["mouse_first_pick_up"] )
        success_time = get_success_retrieval_time(df_summary, trial_num,
                                        trial_num_col = self.DLC_summary_cols["trial_num"],
                                        success_col = self.DLC_summary_cols["trial_success"],
                                        end_time_col = self.DLC_summary_cols["trial_end"])

        # sort pup_locations by start_time
        pup_locations_keys = sorted(pup_locations.keys(), key=lambda x: pup_locations[x]["start_time"])
        last_pup_location = pup_locations_keys[-1]
        print(f"Pup locations keys: {pup_locations_keys}")
        print(f"Last location: {last_pup_location}")

        ### pick up positioning ###
        marked_locations = []
        if pick_up_time is not None:
            list_before = [pup_loc for pup_loc in pup_locations_keys if pick_up_time < pup_locations[pup_loc]["start_time"]]
            list_after = [pup_loc for pup_loc in pup_locations_keys if pick_up_time > pup_locations[pup_loc]["end_time"]]
            list_included = [pup_loc for pup_loc in pup_locations_keys if pick_up_time >= pup_locations[pup_loc]["start_time"] and pick_up_time <= pup_locations[pup_loc]["end_time"]]

            if list_included:
                print("Found pick up time included in pup location:")
                location_included = list_included[0] # only one element in list_included
                # check if the pickup is located at the edges of the cluster (within a certain tolerance)

                time_to_tolerance_secs = 0.03
                difference_start_time = np.abs(pick_up_time - pup_locations[location_included]["start_time"])
                difference_end_time = np.abs(pick_up_time - pup_locations[location_included]["end_time"])

                if difference_start_time < time_to_tolerance_secs:
                    pup_locations[location_included]["pickup"] = "before"
                    print(f"Pick up time at {pick_up_time} set to before, close to start_time of cluster {location_included}: {pup_locations[location_included]['start_time']}")
                    marked_locations += [location_included]
                elif difference_end_time < time_to_tolerance_secs:
                    pup_locations[location_included]["pickup"] = "after"
                    print(f"Pick up time at {pick_up_time} set to after, close to end_time of cluster {location_included}: {pup_locations[location_included]['end_time']}")
                    marked_locations += [location_included]
                else:
                    pup_locations[location_included]["pickup"] = "included"
                    print(f"Pick up time at {pick_up_time} set to included, within the cluster {location_included}")
                    marked_locations += [location_included]
                    

                # check if the pickup is included in the cluster
                # pup_locations[location_included]["pickup"] = "included"
                # marked_locations += [location_included]

            else:
                if list_before:
                    first_cluster = list_before[0] # first element of list_before
                    pup_locations[first_cluster]["pickup"] = "before"
                    marked_locations += [first_cluster]

                if list_after:
                    last_cluster = list_after[-1] # last element of list_after
                    pup_locations[last_cluster]["pickup"] = "after"
                    marked_locations += [last_cluster]    
                
        pup_locations_no_pickup = [pup_loc for pup_loc in pup_locations if pup_loc not in marked_locations]

        ### success retrieval positioning ###
        pup_locations[last_pup_location]["success"] = "retrieval" if success_time is not None else "failed"
        
        pup_locations_no_success = [pup_loc for pup_loc in pup_locations if pup_loc!=last_pup_location]
        print(f"Pup locations no success: {pup_locations_no_success}")

        ## setting success and pickup to None ##
        for pup_loc in pup_locations_no_pickup:
            pup_locations[pup_loc]["pickup"] = "none"

        for pup_loc in pup_locations_no_success:
            pup_locations[pup_loc]["success"] = "none"

        return pup_locations
        
    def compute_distances_clusters(self, pup_locations, trial_df):

        pup_x_col = self.DLC_cols["pup"]["x"]
        pup_y_col = self.DLC_cols["pup"]["y"]
        pixel_to_cm = self.pixels_to_cm_ratio
        time_col = self.DLC_cols["time"]

        distances = []
        pup_locations_keys = sorted(pup_locations.keys(), key=lambda x: pup_locations[x]["start_time"])
        
        if len(pup_locations_keys) > 1:
            pup_locations_prev, pup_locations_next = pup_locations_keys[:-1], pup_locations_keys[1:]

            for prev_loc, next_loc in zip(pup_locations_prev, pup_locations_next):

                start_time_prev, end_time_prev = pup_locations[prev_loc]["start_time"], pup_locations[prev_loc]["end_time"]
                start_time_next, end_time_next = pup_locations[next_loc]["start_time"], pup_locations[next_loc]["end_time"]
                
                trial_prev = trial_df[(trial_df[time_col] >= start_time_prev) & (trial_df[time_col] <= end_time_prev)]
                trial_next = trial_df[(trial_df[time_col] >= start_time_next) & (trial_df[time_col] <= end_time_next)]

                (x_prev, y_prev) = (trial_prev[pup_x_col].mean(), trial_prev[pup_y_col].mean())
                (x_next, y_next) = (trial_next[pup_x_col].mean(), trial_next[pup_y_col].mean())
                pup_locations[prev_loc]["distance_to_next_cluster_cm"] = compute_distance((x_prev, y_prev), (x_next, y_next)) / pixel_to_cm

            pup_locations[pup_locations_keys[-1]]["distance_to_next_cluster_cm"] = "none"

        else:
            pup_locations[pup_locations_keys[0]]["distance_to_next_cluster_cm"] = "none"

        return pup_locations
    
    def label_pup_interaction_behaviors_trial(self, trial_df, trial_num, start_time, end_time, df_summary, kernel_size=20,
                                  pre_event_window_size_time=10, frame_rate=30,):
        """Labels mouse behaviors (approach, crouching, active interaction) in the time window before pup pickup"""
        
        time_seconds_col = self.DLC_cols["time"]
        mouse_x_col = self.DLC_cols["mouse_position"]["x"]
        mouse_y_col = self.DLC_cols["mouse_position"]["y"]
        head_x_col = self.DLC_cols["head_position"]["x"]
        head_y_col = self.DLC_cols["head_position"]["y"]
        pup_x_col = self.DLC_cols["pup"]["x"]
        pup_y_col = self.DLC_cols["pup"]["y"]
        distance_to_pup_col = self.DLC_behaviour_cols["distance_mouse_pup"]
        distance_to_head_col = self.DLC_behaviour_cols["distance_head_pup"]
        head_angle_to_pup_col = self.DLC_behaviour_cols["head_angle_to_pup"]
        mouse_speed_col = self.DLC_behaviour_cols["mouse_speed"]
        approach_col = self.states["approach"]
        crouching_col = self.states["crouching"]
        active_interaction_col = self.states["active_interaction"]
        
        # Get window of interest
        window_frames = (trial_df[time_seconds_col] >= start_time) & (trial_df[time_seconds_col] <= end_time)
        window = trial_df.loc[window_frames].copy()

        # Check that derivatives can be computed (length >2)
        print(f"Window length: {len(window)}")
        if len(window) < kernel_size:
            print(f"Not enough data to compute derivatives for trial {trial_num}, window length: {len(window)}")
            return window
        
        # Calculate distance to pup center
        pup_corner_bounds = self.BF_instance.extract_pup_starting_position_bounds(df_summary, trial_num)
        # pup_center_x = pup_corner_bounds["xmin"] + (pup_corner_bounds["xmax"] - pup_corner_bounds["xmin"]) / 2
        # pup_center_y = pup_corner_bounds["ymin"] + (pup_corner_bounds["ymax"] - pup_corner_bounds["ymin"]) / 2

        # Convert distances to cm and calculate derivatives
        columns = [mouse_speed_col, distance_to_pup_col, distance_to_head_col]
        px_cm_ratio = self.pixels_to_cm_ratio
        for col in columns:
            window[col+"_cm"] = window[col] / px_cm_ratio
            window[col+"_cm_deriv"] = np.gradient(window[col+"_cm"])

        # Smooth derivatives
        kernel = np.ones(kernel_size) / kernel_size

        window[head_angle_to_pup_col + "_convolved"] = np.convolve(window[head_angle_to_pup_col], kernel, mode='same')
        window[mouse_speed_col+"_cm_convolved"] = np.convolve(window[mouse_speed_col+"_cm"], kernel, mode='same')
        window[mouse_speed_col+"_cm_deriv_convolved"] = np.gradient(window[mouse_speed_col+"_cm_convolved"])
        window[distance_to_pup_col + "_cm" + "_deriv" + "_convolved"] = np.convolve(window[distance_to_pup_col + "_cm" + "_deriv"], kernel, mode='same')
        window[distance_to_head_col + "_cm" + "_deriv" + "_convolved"] = np.convolve(window[distance_to_head_col + "_cm" + "_deriv"], kernel, mode='same')
        
        # Label behaviors
        mask_approach = (window[distance_to_head_col + "_cm" + "_deriv" + "_convolved"] < self.threshold_approach_speed_cms) & (window[head_angle_to_pup_col + "_convolved"] <=  self.threshold_approach_angle_degrees)
        mask_crouching = (window[distance_to_pup_col + "_cm"] < 2) & (window["mouse_speed_px/s_cm_convolved"] < 5)
        mask_active_interaction = (window[distance_to_head_col + "_cm"] < 2) & (window["mouse_speed_px/s_cm_convolved"] < 5)
            
        # mask approach, crouching and active interaction
        window[approach_col] = mask_approach
        window[crouching_col] = mask_crouching
        window[active_interaction_col] = mask_active_interaction

        return window

    def annotate_full_trial(self, trial_df, trial_num, df_summary, pup_locations): 
        """Labels mouse behaviors (approach, crouching, active interaction) by iterating over all pup locations and labeling the behaviors in the time window before each pup location"""

        def check_pickup_validity(trial_df, pup_locations, pup_loc_id):
            """Check if the pickup is valid by looking at the distance to the pup and the head angle to the pup"""
            
            pup_loc = pup_locations[pup_loc_id]
            start_time = pup_loc["start_time"]
            end_time = pup_loc["end_time"]
            window = trial_df[(trial_df[time_seconds_col] >= start_time) & (trial_df[time_seconds_col] <= end_time)]
            
            num_frames = min(20, len(window))

            frames_before_pickup_distance_mean_cm = window[distance_to_pup_col].iloc[-num_frames:].mean() / self.pixels_to_cm_ratio

            threshold_distance_to_pup_cm = 5

            if (frames_before_pickup_distance_mean_cm > threshold_distance_to_pup_cm):
                return False
            else:
                return True
            
        def propagate_pup_coords_and_recompute(trial_df, mask, pup_cols, pup_related_cols):
            """Helper function to propagate pup coordinates and recompute related columns over a given mask
            
            Parameters:
            -----------
            trial_df : pandas DataFrame
                DataFrame containing trial data
            mask : boolean array
                Mask indicating which rows to update
            pup_cols : list
                List of pup coordinate columns to forward fill
            pup_related_cols : list
                List of columns that need to be recomputed after forward fill
                
            Returns:
            --------
            pandas DataFrame
                Updated trial DataFrame
            """
            print("NaN counts before forward fill: ", trial_df.loc[mask, pup_related_cols].isna().sum())
            print("---Overall NaN counts: ", trial_df[pup_related_cols].isna().sum())

            # forward fill the pup_cols
            trial_df.loc[mask, pup_cols] = trial_df.loc[mask, pup_cols].ffill().values

            pup_x_col, pup_y_col = pup_cols[0], pup_cols[1]
            head_x_col, head_y_col = self.DLC_cols["head_position"]["x"], self.DLC_cols["head_position"]["y"]
            mouse_x_col, mouse_y_col = self.DLC_cols["mouse_position"]["x"], self.DLC_cols["mouse_position"]["y"]
            distance_to_pup_col, distance_to_head_col, head_angle_to_pup_col = pup_related_cols[0], pup_related_cols[1], pup_related_cols[2]

            # recompute the pup_related_cols
            print("----> Recomputing distance to pup")
            trial_df.loc[mask, distance_to_pup_col] = self.BF_instance.compute_distance_to_pup(trial_df.loc[mask],
                                                x_col = mouse_x_col,
                                                y_col = mouse_y_col,
                                                pup_x_col = pup_x_col,
                                                pup_y_col = pup_y_col,
                                                distance_col = distance_to_pup_col)[distance_to_pup_col].values
        
            trial_df.loc[mask, distance_to_head_col] = self.BF_instance.compute_distance_to_pup(trial_df.loc[mask],
                                                x_col = head_x_col,
                                                y_col = head_y_col,
                                                pup_x_col = pup_x_col,
                                                pup_y_col = pup_y_col,
                                                distance_col = distance_to_head_col)[distance_to_head_col].values

            print("----> Recomputing head angle to pup")
            trial_df.loc[mask, head_angle_to_pup_col] = self.BF_instance.compute_head_angle_to_pup(trial_df.loc[mask], add_vector_columns = False,
                                                head_angle_to_pup_col = head_angle_to_pup_col)[head_angle_to_pup_col].values

            print("NaN counts after forward fill: ", trial_df.loc[mask, pup_related_cols].isna().sum())
            print("---Overall NaN counts: ", trial_df[pup_related_cols].isna().sum())
            
            return trial_df
        
        def get_pickup_frame(trial_df, pickup_time, frame_index_col = "frame_index"):
            if pickup_time is not None:
                pickup_mask = (trial_df[pickup_col] == True)
                pick_up_frame = trial_df[pickup_mask][frame_index_col].values[0]
                return pick_up_frame
            else:
                return None

        def get_start_end_frames_cluster(trial_df, pup_locations, pup_loc_id):

            start_time_window = pup_locations[pup_loc_id]["start_time"]
            end_time_window = pup_locations[pup_loc_id]["end_time"]

            mask_pup_location = (trial_df[time_seconds_col] >= start_time_window) & (trial_df[time_seconds_col] <= end_time_window)
            window = trial_df.loc[mask_pup_location].copy()

            end_time_frame = window[frame_index_col].max()
            start_time_frame = window[frame_index_col].min()

            return start_time_frame, end_time_frame

        time_seconds_col = self.DLC_cols["time"]
        frame_index_col = self.DLC_cols["frame"]

        mouse_id = df_summary[self.DLC_summary_cols["animal_id"]].values[0]
        day = df_summary[self.DLC_summary_cols["day"]].values[0]

        pup_x_col = self.DLC_cols["pup"]["x"]
        pup_y_col = self.DLC_cols["pup"]["y"]
        distance_to_pup_col = self.DLC_behaviour_cols["distance_mouse_pup"]
        distance_to_head_col = self.DLC_behaviour_cols["distance_head_pup"]
        head_angle_to_pup_col = self.DLC_behaviour_cols["head_angle_to_pup"]

        pickup_col = self.states["pickup"]
        drop_col = self.states["drop"]
        retrieval_col = self.states["retrieval"]
        carrying_col = self.states["carrying"]
        active_interaction_col = self.states["active_interaction"]
        approach_col = self.states["approach"]
        crouching_col = self.states["crouching"]

        frame_rate = self.frame_rate
        
        pickup_time = get_pick_up_time(df_summary, trial_num,
                                        trial_num_col = self.DLC_summary_cols["trial_num"],
                                        pick_up_time_col = self.DLC_summary_cols["mouse_first_pick_up"])
        pick_up_frame = get_pickup_frame(trial_df, pickup_time,
                                        frame_index_col = self.DLC_cols["frame"])
                                        
        print("* Pick up frame: ", pick_up_frame, "* Pickup time: ", pickup_time)

        pup_cols = [pup_x_col, pup_y_col]
        pup_related_cols = [distance_to_pup_col, distance_to_head_col, head_angle_to_pup_col]
        print(pup_related_cols)

        pup_locations_keys = sorted(pup_locations.keys(), key=lambda x: pup_locations[x]["start_time"])
        last_pup_location = pup_locations_keys[-1]

        #### annotation of carrying, pickup and drop ####
        for i, pup_loc in enumerate(pup_locations_keys):

            print(f"====== {pup_loc}: Annotation of carrying, pickup and drop  ======")

            print("Before annotation and analysis:")
            print(trial_df[[approach_col, crouching_col, active_interaction_col]].value_counts())

            start_time_window, end_time_window = pup_locations[pup_loc]["start_time"], pup_locations[pup_loc]["end_time"]

            start_time_frame, end_time_frame = get_start_end_frames_cluster(trial_df, pup_locations, pup_loc)

            pickup_type = pup_locations[pup_loc]["pickup"]
            success_type = pup_locations[pup_loc]["success"]

            print(f"====== {pup_loc}: pickup_type: '{pickup_type}' success_type: '{success_type}' ")

            if pickup_type != "none": # if there is a pickup surrounding or within the cluster

                if pickup_type == "after":
                    end_time_window = pickup_time
                    mask_start_to_pickup = (trial_df[frame_index_col] < pick_up_frame) & (trial_df[time_seconds_col] >= start_time_window)

                    trial_df = propagate_pup_coords_and_recompute(trial_df, mask_start_to_pickup, pup_cols, pup_related_cols)

                    pup_locations[pup_loc]["end_time"] = pickup_time
                    print(f"Pup location end time updated to: {pickup_time}")
                    print("Propagating pup coordinates to the pickup time")

                elif pickup_type == "before":
                    mask_pickup_to_start = (trial_df[frame_index_col] > pick_up_frame) & (trial_df[time_seconds_col] < start_time_window)
                    frame_start_drop = convert_seconds_to_frame(start_time_window, frame_rate)

                    trial_df.loc[mask_pickup_to_start, carrying_col] = True
                    trial_df.loc[trial_df[frame_index_col] == frame_start_drop, drop_col] = True

                    print(f"Carry from {pick_up_frame} to drop at {frame_start_drop}")

                elif pickup_type == "included": # drop instantly after pickup
                    frame_drop = pick_up_frame + 1
                    trial_df.loc[trial_df[frame_index_col] == frame_drop, drop_col] = True # set drop to true at the next frame
                    print(f"Pick up at {pick_up_frame}, instantly drop at {frame_drop}")

                # additional processing for last pup location
                if pup_loc == last_pup_location:
                    print(f"---> Processing last pup location: {pup_loc}")
                    if pickup_type == "after":
                        mask_pickup_to_end = (trial_df[frame_index_col] > pick_up_frame)
                        mask_end_of_trial = (trial_df[frame_index_col] == trial_df[frame_index_col].max())
                        trial_df.loc[mask_pickup_to_end, carrying_col] = True
                        end_col = (retrieval_col if success_type == "retrieval" else drop_col)
                        print(f"Setting {end_col} to True at the end of the trial")
                        trial_df.loc[mask_end_of_trial, end_col] = True

                        print(f"Carry from pick up point {pick_up_frame} to end of trial, {end_col} at {trial_df[frame_index_col].max()}")

                    elif pickup_type == "before":

                        if success_type == "retrieval":
                            mask_pickup_to_end = (trial_df[frame_index_col] > end_time_frame)
                            mask_exact_pickup = (trial_df[frame_index_col] == end_time_frame)

                            ##### create a pickup at the end of the cluster
                            if check_pickup_validity(trial_df, pup_locations, pup_loc):
                                trial_df.loc[mask_exact_pickup, pickup_col] = True
                                trial_df.loc[mask_pickup_to_end, carrying_col] = True
                                trial_df.iloc[-1][retrieval_col] = True

                                print(f"Pickup at {pick_up_frame}, Carry -> Retrieval at {trial_df[frame_index_col].max()}")
                            else:
                                #print(f"Pickup at {pick_up_frame} is invalid, skipping")
                                raise ValueError(f"Pickup at {pick_up_frame} for {mouse_id} - {day} - {trial_num} is invalid, Mouse does not move closer to the pup than 5 cm, skipping")
                        else:
                            # propagate pup coordinates to the end of the trial and adjust end time of cluster
                            mask_end_cluster_to_final = (trial_df[frame_index_col] > end_time_frame)
                            # trial_df.loc[mask_end_cluster_to_final, pup_related_cols] = trial_df.iloc[-1][pup_related_cols].values
                            trial_df = propagate_pup_coords_and_recompute(trial_df, mask_end_cluster_to_final, pup_cols, pup_related_cols)
                            
                            pup_locations[pup_loc]["end_time"] = trial_df[time_seconds_col].max()
                            print(f"Propagating pup coordinates to the end of the trial at {end_time_frame}")
                            print("Pup location end time updated to: ", trial_df[time_seconds_col].max())

                    elif pickup_type == "included":
                        if success_type == "retrieval":
                            mask_pickup_to_end = (trial_df[frame_index_col] > end_time_frame)
                            mask_exact_pickup = (trial_df[frame_index_col] == end_time_frame)

                            ### create a pickup at the end of the cluster
                            if check_pickup_validity(trial_df, pup_locations, pup_loc):
                                trial_df.loc[mask_exact_pickup, pickup_col] = True
                                trial_df.loc[mask_pickup_to_end, carrying_col] = True
                                trial_df.iloc[-1][retrieval_col] = True

                                print(f"Pickup at {end_time_frame}, Carry -> Retrieval at {trial_df[frame_index_col].max()}")
                            else:
                                #print(f"Pickup at {end_time_frame} is invalid, skipping")
                                raise ValueError(f"Pickup at {end_time_frame} for {mouse_id} - {day} - {trial_num} is invalid, Mouse does not move closer to the pup than 5 cm, skipping")
                        else:
                            mask_end_cluster_to_final = (trial_df[frame_index_col] >= end_time_frame)
                            # trial_df.loc[mask_end_cluster_to_final, pup_related_cols] = trial_df.iloc[-1][pup_related_cols].values
                            trial_df = propagate_pup_coords_and_recompute(trial_df, mask_end_cluster_to_final, pup_cols, pup_related_cols)
                
                            pup_locations[pup_loc]["end_time"] = trial_df[time_seconds_col].max()

                            print(f"Propagating pup coordinates to the end of the trial at {end_time_frame}")
                            print("Pup location end time updated to: ", trial_df[time_seconds_col].max())

            elif pickup_type == "none":

                # get distance to next cluster
                distance_to_next_cluster = pup_locations[pup_loc]["distance_to_next_cluster_cm"]
                max_distance_between_clusters_cm = self.distance_between_clusters_cm
                if distance_to_next_cluster!="none" and distance_to_next_cluster > max_distance_between_clusters_cm:
                    print("Distance to next cluster is too big: ", distance_to_next_cluster)

                    next_cluster_id = pup_locations_keys[i+1]

                    start_frame_next_cluster, end_frame_next_cluster = get_start_end_frames_cluster(trial_df, pup_locations, next_cluster_id)
                    
                    # insert a pickup at the end of the cluster, then carry to the next cluster, drop at the start of the next cluster
                    mask_exact_pickup = (trial_df[frame_index_col] == end_time_frame)
                    mask_carry_to_next_cluster = (trial_df[frame_index_col] > end_time_frame) & (trial_df[frame_index_col] < start_frame_next_cluster)
                    mask_drop_at_next_cluster = (trial_df[frame_index_col] == start_frame_next_cluster)
                    
                    ##### create a pickup at the end of the current cluster
                    if check_pickup_validity(trial_df, pup_locations, pup_loc):
                        trial_df.loc[mask_exact_pickup, pickup_col] = True
                        trial_df.loc[mask_carry_to_next_cluster, carrying_col] = True
                        trial_df.loc[mask_drop_at_next_cluster, drop_col] = True
                    else:
                        #print(f"Pickup at {end_time_frame} is invalid, skipping")
                        raise ValueError(f"Pickup at {end_time_frame} for {mouse_id} - {day} - {trial_num} is invalid, Mouse does not move closer to the pup than 5 cm, skipping")

                elif pup_loc == last_pup_location:
                    if success_type == "retrieval":
                        mask_exact_pickup = (trial_df[frame_index_col] == end_time_frame)
                        mask_carry_to_end = (trial_df[frame_index_col] > end_time_frame)

                        ### create a pickup at the end of the cluster
                        if check_pickup_validity(trial_df, pup_locations, pup_loc):
                            trial_df.loc[mask_exact_pickup, pickup_col] = True
                            trial_df.loc[mask_carry_to_end, carrying_col] = True
                            trial_df.iloc[-1][retrieval_col] = True
                        else:
                            #print(f"Pickup at {end_time_frame} is invalid, skipping")
                            raise ValueError(f"Pickup at {end_time_frame} for {mouse_id} - {day} - {trial_num} is invalid, Mouse does not move closer to the pup than 5 cm, skipping")

                    elif success_type == "failed":
                        # propagate pup coordinates to the end of the trial and adjust end time of cluster
                        mask_end_cluster_to_final = (trial_df[frame_index_col] >= end_time_frame)
                        # trial_df.loc[mask_end_cluster_to_final, pup_related_cols] = trial_df.iloc[-1][pup_related_cols].values
                        trial_df = propagate_pup_coords_and_recompute(trial_df, mask_end_cluster_to_final, pup_cols, pup_related_cols)
                        
                        pup_locations[pup_loc]["end_time"] = trial_df[time_seconds_col].max()

                        print(f"Propagating pup coordinates to the end of the trial at {end_time_frame}")
                        print("Pup location end time updated to: ", trial_df[time_seconds_col].max())
            
            print("\n===== After annotations edits:")
            print(trial_df[[drop_col, pickup_col, carrying_col, retrieval_col]].value_counts())

            ##### pup-directed behavioral analysis on the appropriate time windows #####
            print("\n===== General pup-directed analysis =====")
            start_time_window = pup_locations[pup_loc]["start_time"]
            end_time_window = pup_locations[pup_loc]["end_time"]
            mask_window = (trial_df[time_seconds_col] >= start_time_window) & (trial_df[time_seconds_col] <= end_time_window)
            print(f" -> Window: {start_time_window} to {end_time_window}")

            pup_cluster_df = trial_df.loc[mask_window].copy()
            annotated_cluster_df = self.label_pup_interaction_behaviors_trial(pup_cluster_df, trial_num, start_time_window, end_time_window, df_summary)
            trial_df.loc[mask_window, trial_df.columns] = annotated_cluster_df[trial_df.columns].values
            
            print("\n=====> After ALL edits:")
            print(trial_df[[approach_col, crouching_col, active_interaction_col]].value_counts())
            print(trial_df[[drop_col, pickup_col, carrying_col, retrieval_col]].value_counts())

        print(" **** Final pup locations: **** ")
        pprint.pprint(pup_locations)

        return trial_df, pup_locations

    def label_passive_behaviors_trial(self, trial_df, df_summary):
    
        pup_x_col = self.config["DLC_columns"]["pup"]["x"]
        pup_y_col = self.config["DLC_columns"]["pup"]["y"]
        mouse_speed_col = self.config["DLC_behaviour_columns"]["mouse_speed"]
        pixels_to_cm_ratio = self.config["pixels_to_cm_ratio"]
        mouse_x_col = self.config["DLC_columns"]["mouse_position"]["x"]
        mouse_y_col = self.config["DLC_columns"]["mouse_position"]["y"]
        in_nest_col = self.config["Behavioral_states"]["in_nest"]

        mouse_speed_cm = trial_df[mouse_speed_col] / pixels_to_cm_ratio
        mask_walking = (mouse_speed_cm > 5)
        mask_still = (mouse_speed_cm < 1)

        trial_df[self.config["Behavioral_states"]["walking"]] = mask_walking
        trial_df[self.config["Behavioral_states"]["still"]] = mask_still

        trial_df = self.BF_instance.flag_nest_coordinates(trial_df, in_nest_col = in_nest_col,
                                                        x = mouse_x_col, y = mouse_y_col,
                                                        nest_bounds = self.config["nest_bounds"])

        return trial_df

    def resolve_simultaneous_labels(self, trial_df, final_behavior_col = "behavior_annotation"):

        behaviors_dict = self.config["Behavioral_states"]
        single_event_states = [val for key, val in behaviors_dict.items() if key in self.config["single_event_states"]]

        ordered_behaviors = single_event_states + [behaviors_dict["carrying"]] + [behaviors_dict["in_nest"]] + [behaviors_dict["approach"]] + [behaviors_dict["active_interaction"]] + [behaviors_dict["crouching"]] + [behaviors_dict["still"]] + [behaviors_dict["walking"]]

        trial_df[final_behavior_col] = 'none'

        for behavior_name in ordered_behaviors:

            # define the mask for the current behavior out of available behavior slots
            mask_behavior = (trial_df[behavior_name]) & (trial_df[final_behavior_col] == 'none')
            # edit the single behavior column to be false for all the behaviors
            trial_df.loc[~mask_behavior, behavior_name] = False
            # edit the final behavior column to be the current behavior
            trial_df.loc[mask_behavior, final_behavior_col] = behavior_name

        return trial_df

    def plot_behavioral_annotations(self, trial_df_annotated, df_summary, pup_locations, 
                                    mouse_id, day, trial_num, start_time = None, end_time = None, add_USV_plot = False,
                                    add_head_angle_to_pup_plot = False, export_plot = False, plot_dir = "full_annotation_plots"):
        """Plots the labeled behaviors and USV data leading up to pup pickup"""

        trial_pickup_time = get_pick_up_time(df_summary, trial_num,
                                trial_num_col = self.DLC_summary_cols["trial_num"],
                                pick_up_time_col = self.DLC_summary_cols["mouse_first_pick_up"])
        time_seconds_col = self.DLC_cols["time"]
        trial_num_col = self.DLC_summary_cols["trial_num"]
        success_col = self.DLC_summary_cols["trial_success"]
        success = df_summary[df_summary[trial_num_col] == trial_num][success_col].values[0]

        if start_time is None:
            trial_start_time = trial_df_annotated[time_seconds_col].iloc[0]
        else:
            trial_start_time = start_time

        if end_time is None:
            trial_end_time = trial_df_annotated[time_seconds_col].iloc[-1]
        else:
            trial_end_time = end_time

        trial_df_annotated = trial_df_annotated[(trial_df_annotated[time_seconds_col] >= trial_start_time) & (trial_df_annotated[time_seconds_col] <= trial_end_time)]

        pup_x_col = self.DLC_cols["pup"]["x"]
        pup_y_col = self.DLC_cols["pup"]["y"]
        distance_to_pup_col = self.DLC_behaviour_cols["distance_mouse_pup"]
        pixels_to_cm_ratio = self.pixels_to_cm_ratio
        
        pickup_col = self.states["pickup"]
        drop_col = self.states["drop"]
        retrieval_col = self.states["retrieval"]
        std_frequency_col = self.config["USV_processing"]["output_columns"]["std_frequency"]

        states = [self.states["approach"], self.states["crouching"], self.states["active_interaction"],
                    self.states["pickup"], self.states["drop"], self.states["carrying"], self.states["retrieval"]]

        single_event_states = self.config["single_event_states"]
        
        trial_pickup_time_minutes = f"{str(int(trial_pickup_time//60))}:{int(trial_pickup_time%60)}" if trial_pickup_time is not None else "None"
        trial_start_time_minutes = f"{str(int(trial_start_time//60))}:{int(trial_start_time%60)}"  
        trial_end_time_minutes = f"{str(int(trial_end_time//60))}:{int(trial_end_time%60)}"
        
        # Create plot of distance to pup
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        trial_df_annotated[distance_to_pup_col+"_cm"] = trial_df_annotated[distance_to_pup_col] / pixels_to_cm_ratio
        trial_df_annotated.plot(x="time_seconds", y=[distance_to_pup_col+"_cm"], color="black", linewidth=1, ax=ax, zorder = 0)
        ax.set_ylabel("Distance of mouse to pup (cm)", color="black")
        ax.set_xlabel("Time (s)")

        if add_head_angle_to_pup_plot:
            ax2 = ax.twinx()
            trial_df_annotated.plot(x="time_seconds", y=["head_angle_to_pup_degrees"], color="red", linewidth=1, ax=ax2, zorder = 0, alpha=0.5)
            #ax2.set_ylabel("Head angle to pup (degrees)", color="red")

        # Add USV plot
        if add_USV_plot:
            ax3 = ax.twinx()
            
            # Plot average frequency with error bars if std exists
            if std_frequency_col in trial_df_annotated.columns:
                ax3.errorbar(x=trial_df_annotated["time_seconds"], 
                            y=trial_df_annotated["average_frequency"],
                            yerr=trial_df_annotated[std_frequency_col],
                            fmt='o', color="purple", markersize=3, 
                            alpha=0.5, zorder=0, elinewidth=1)
            else:
                # Original scatter plot if no std available
                trial_df_annotated.plot(x="time_seconds", y=["average_frequency"],
                        ax=ax3, kind="scatter", color="purple", zorder=0, s=5, alpha=0.5)
            
            ax3.set_ylabel("Average frequency of USVs (kHz)", color="purple")

        # for each state, pick a color
        dict_colors = {self.states["approach"]: "green",
                    self.states["crouching"]: "red",
                    self.states["active_interaction"]: "dodgerblue",
                    self.states["pickup"]: "magenta",
                    self.states["drop"]: "crimson",
                    self.states["carrying"]: "darkorange",
                    self.states["retrieval"]: "lime",
                    self.states["walking"]: "yellow",
                    self.states["still"]: "gray",
                    self.states["in_nest"]: "cyan"}

        pup_locations_keys = sorted(pup_locations.keys(), key=lambda x: pup_locations[x]["start_time"])
        gradient_blues = ["deepskyblue",  "royalblue",  "darkblue", "blue", "blueviolet"]
        colors_pup_locations = {pup_loc: gradient_blues[i] for i, pup_loc in enumerate(pup_locations_keys)}
        
        
        for pup_loc in pup_locations_keys:
            print("pup_loc", pup_loc)
            mask_pup_available = (trial_df_annotated[time_seconds_col] >= pup_locations[pup_loc]["start_time"]) & (trial_df_annotated[time_seconds_col] <= pup_locations[pup_loc]["end_time"])
            print("mask_pup_available", mask_pup_available.sum())
            ax.fill_between(trial_df_annotated[time_seconds_col], 0.95, 0.97, 
                    where=mask_pup_available,
                    transform=ax.get_xaxis_transform(),
                    facecolor=colors_pup_locations[pup_loc], alpha=1., zorder = 2)
        

        # Add behavior shading
        for behavior, color in dict_colors.items():
            # Get time points where behavior is True
            if behavior in trial_df_annotated.columns:

                behavior_times = trial_df_annotated.loc[trial_df_annotated[behavior], time_seconds_col]
                times = trial_df_annotated[time_seconds_col]
                # Create spans for each time point
                if behavior_times.size > 0:
                    if behavior in single_event_states:
                        for time in behavior_times:
                            ax.axvline(x=time, 
                                    ymin=0, ymax=0.875,  # Same vertical span as before
                                    color=color,
                                    linewidth=2,  # Makes the line thicker
                                    alpha=1., zorder = 2)
                    else:
                        ax.fill_between(times, 0, 0.875, 
                                where=trial_df_annotated[behavior],
                                transform=ax.get_xaxis_transform(),
                                facecolor=color, alpha=0.4, zorder = 1)

        # Add legend
        legend_elements_states_span = [mpatches.Patch(color=color, label=state, alpha=0.5) for state, color in dict_colors.items() if state not in single_event_states]
        legend_states_single_event = [mpatches.Patch(color=color, label=state, alpha=1.) for state, color in dict_colors.items() if state in single_event_states]
        legend_elements_pup_location = [mpatches.Patch(color=color, label=pup_loc, alpha=1) for pup_loc, color in colors_pup_locations.items()]

        legend_elements = legend_elements_states_span + legend_states_single_event
        #legend_elements.append(mlines.Line2D([], [], color='black', label='Distance to pup'))

        if add_USV_plot:
            if std_frequency_col in trial_df_annotated.columns:
                legend_elements.append(mlines.Line2D([], [], color='purple', marker='o', 
                             linestyle='None', label='Average frequency of USVs (with std)'))
            else:
                legend_elements.append(mlines.Line2D([], [], color='purple', marker='o', 
                             linestyle='None', label='Average frequency of USVs'))
        if add_head_angle_to_pup_plot:
            legend_elements.append(mlines.Line2D([], [], color='red', label='Head angle to pup'))

        # Create first legend in lower right with high zorder
        first_legend = ax.legend(handles=legend_elements, loc="lower right")
        # Add the first legend manually to the plot
        
        ax.add_artist(first_legend)
        second_legend = ax.legend(handles=legend_elements_pup_location, loc="upper right", bbox_to_anchor=(1.0, 0.95))
        ax.add_artist(second_legend)
        
        ax.set_title(f"{mouse_id} - {day} - Trial {trial_num} | Success was: {success} | Annotations from trial start at {trial_start_time_minutes} to end of trial at {trial_end_time_minutes}, pick up at {trial_pickup_time_minutes}")
        
        if export_plot:
            os.makedirs(f"plots/{plot_dir}", exist_ok = True)
            os.makedirs(f"plots/{plot_dir}/{mouse_id}/{day}", exist_ok = True)
            path = f"plots/{plot_dir}/{mouse_id}/{day}/{mouse_id}_{day}_trial_{trial_num}.png"
            plt.savefig(path)
            plt.show()
        else:
            plt.show()

    def export_trial(self, trial_df_annotated, pup_locations_annotated, df_summary,
                            mouse_id, day, trial_num, processed_data_dir = "annotated_data"):
        
        os.makedirs(processed_data_dir, exist_ok = True)
        os.makedirs(f"{processed_data_dir}/{mouse_id}/{day}/trials/", exist_ok = True)

        if df_summary is not None:
            df_summary.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/BehavSummary_{mouse_id}_{day}.csv", index = False)

        if trial_df_annotated is not None:
            path = f"{processed_data_dir}/{mouse_id}/{day}/trials/trial{trial_num}_DLC_annotated_{mouse_id}_{day}.csv"
            trial_df_annotated.to_csv(path, index=False)    

        if pup_locations_annotated is not None:
            path = f"{processed_data_dir}/{mouse_id}/{day}/trials/{mouse_id}_{day}_trial{trial_num}_pup_location_dict.json"
            with open(path, 'w') as f:
                json.dump(pup_locations_annotated, f, indent=4)
        
    def run_pup_directed_behavior_annotation(self, mouse_id, day, trial_num,
                                                trial_df, df_summary, pup_locations,
                                                processed_data_dir = "annotated_data", export = False):

        print(f" ==== Example: {mouse_id} - {day} - {trial_num} ==== ")

        # 1. create default columns
        print("===== * = * = 1. Creating default columns = * = * ======")
        trial_df = self.create_default_columns(trial_df)

        # 2. mark existing times
        print("===== * = * = 2. Marking existing times = * = * ======")
        trial_df = self.mark_existing_times(trial_df, df_summary, trial_num)

        # 3. Assigning pick up and success types
        print("===== * = * = 3. Assigning pick up and success types = * = * ======")
        pup_locations_assigned = self.assign_pick_up_and_success_types(pup_locations, df_summary, trial_num)

        # 4. compute distances
        print("===== * = * = 4. Computing distances = * = * ======")
        pup_locations_distances = self.compute_distances_clusters(pup_locations_assigned, trial_df)

        # 5. annotate trial
        print("===== * = * = 5. Annotating trial = * = * ======")
        trial_df_annotated, pup_locations_annotated = self.annotate_full_trial(trial_df, trial_num, df_summary, pup_locations_distances)

        # 8. plot passive behaviors
        print("===== * = * = 8.Plotting passive behaviors = * = * ======")
        trial_df_annotated = self.label_passive_behaviors_trial(trial_df_annotated, df_summary)

        # 9. resolve simultaneous labels
        print("===== * = * = 9.Resolving simultaneous labels = * = * ======")
        trial_df_annotated_resolved = self.resolve_simultaneous_labels(trial_df_annotated)

        # 6. export trial
        print("===== * = * = 6. Exporting trial = * = * ======")
        if export:
            self.export_trial(trial_df_annotated_resolved, pup_locations_annotated, df_summary,
                            mouse_id, day, trial_num, processed_data_dir)

        # 7. plot behavioral annotations
        print("===== * = * = 7.Plotting behavioral annotations = * = * ======")
        self.plot_behavioral_annotations(trial_df_annotated_resolved, df_summary, pup_locations_annotated,
                            mouse_id, day, trial_num,
                            start_time = None, end_time = None, add_USV_plot = False, plot_dir = "full_resolved_annotation_plots", export_plot = True)


        return trial_df_annotated, pup_locations_annotated

    def get_and_export_transition_paths_for_animal(self, processed_and_annotated_data, mouse_ids, days, export = False,
                                                    transition_path_export_dir = "transition_paths",
                                                    export_csv_dir = "annotated_cleaned_resolved_data",
                                                    plot_export_dir = "full_cleaned_resolved_annotation_plots"): 
        transition_paths_dict = {}
        for mouse_id in processed_and_annotated_data.keys():
            transition_paths_dict[mouse_id] = {}
            for day in days:
                transition_paths_dict[mouse_id][day] = {}
                for trial_num in processed_and_annotated_data[mouse_id][day]["trials"].keys():

                    trial_df_annotated = copy.deepcopy(processed_and_annotated_data[mouse_id][day]["trials"][trial_num]["dlc_data"])
                    df_summary = copy.deepcopy(processed_and_annotated_data[mouse_id][day]["Behavior"]["df_summary"])
                    pup_locations_annotated = copy.deepcopy(processed_and_annotated_data[mouse_id][day]["trials"][trial_num]["pup_locations"])

                    trial_df_annotated_cleaned, transition_path = self.get_transition_path_for_trial(trial_df_annotated,
                                                                final_behavior_col = "behavior_annotation")

                    transition_paths_dict[mouse_id][day][trial_num] = transition_path

                    if export:
                        self.export_trial(trial_df_annotated_cleaned, pup_locations_annotated, df_summary,
                                                mouse_id, day, trial_num, processed_data_dir = export_csv_dir)

                        self.plot_behavioral_annotations(trial_df_annotated_cleaned, df_summary, pup_locations_annotated,
                                    mouse_id, day, trial_num, plot_dir = plot_export_dir, export_plot = export)
            if export:
                os.makedirs(transition_path_export_dir, exist_ok=True)
                with open(f"{transition_path_export_dir}/{mouse_id}_transition_paths_dict.json", "w") as f:
                    mouse_specific_dict = {mouse_id: transition_paths_dict[mouse_id]}
                    json.dump(mouse_specific_dict, f, indent=4)

        return transition_paths_dict

    def get_transition_path_for_trial(self, trial_df, final_behavior_col = "behavior_annotation"):
    
        def is_valid_state(state):
            
            single_event_states = [val for key, val in self.config["Behavioral_states"].items() if key in self.config["single_event_states"]]
            duration_state = state['end_time'] - state['start_time']
            print(f"Duration state: {duration_state}")
            threshold_duration_state_secs =  0.7

            if state['behavior_name'] in single_event_states:
                return True
            elif duration_state > threshold_duration_state_secs:
                return True
            else:
                return False

        def remove_short_state(state, trial_df):

            mask_state = (trial_df[state['behavior_name']]) & \
                        (trial_df['time_seconds'] >= state['start_time']) & \
                        (trial_df['time_seconds'] < state['end_time'])
            print(f"Removing short state: {state['behavior_name']}, lasting {state['end_time'] - state['start_time']} seconds")

            # print value counts before and after
            print(f"Value counts before: {trial_df[state['behavior_name']].value_counts()}")
            print(f"Value counts before: {trial_df[final_behavior_col].value_counts()[state['behavior_name']] if state['behavior_name'] in trial_df[final_behavior_col].value_counts() else 0}")

            trial_df.loc[mask_state, final_behavior_col] = 'none'
            trial_df.loc[mask_state, state['behavior_name']] = False

            print(f"Value counts after: {trial_df[state['behavior_name']].value_counts()}")
            print(f"Value counts after: {trial_df[final_behavior_col].value_counts()[state['behavior_name']] if state['behavior_name'] in trial_df[final_behavior_col].value_counts() else 0}")
            return trial_df

        time_seconds_col = self.config["DLC_columns"]["time"]
        current_state = {'behavior_name': 'none', 'start_time': None, 'end_time': None}
        current_state_index = 0
        list_states = {}

        for i, bhv_name in enumerate(trial_df[final_behavior_col]):
            # close previous state if behavior name changes
            current_time = trial_df[time_seconds_col].iloc[i]
            
            if bhv_name != current_state['behavior_name']:
                if current_state['behavior_name'] != "none":
                    current_state['end_time'] = current_time
                    print(f"Current state: {current_state}")

                    if is_valid_state(current_state):
                        print(f"Valid state: {current_state['behavior_name']}, lasting {current_state['end_time'] - current_state['start_time']} seconds")
                        list_states[current_state_index] = current_state
                        current_state_index += 1
                    else:
                        print(f"Invalid state: {current_state['behavior_name']}, lasting {current_state['end_time'] - current_state['start_time']} seconds")
                        # remove the short state
                        trial_df = remove_short_state(current_state, trial_df)

                # open new state
                current_state = {'behavior_name': bhv_name, 'start_time': current_time, 'end_time': None}

        # a behavior was created or closed at the last frame
        current_state['end_time'] = trial_df[time_seconds_col].iloc[-1]
        if current_state['behavior_name'] != "none" and is_valid_state(current_state):
            list_states[current_state_index] = current_state
            current_state_index += 1

        sorted_list_state_keys = sorted(list_states.keys())

        transition_path = [list_states[key]['behavior_name'] for key in sorted_list_state_keys]
        print(f"Transition path: start: {transition_path}")

        return trial_df, transition_path
        
    def create_transition_matrices_from_transition_paths(self, mouse_ids, days, transition_paths_dict, category = ["Mother", "Virgin"]):
        
        states = list(self.config["Behavioral_states"].values())
        dict_transition_matrices = {animal: {day: create_default_counts_matrix(states) for day in days} for animal in category}

        for category in dict_transition_matrices.keys():

            first_char = category[0]
            mouse_ids_category = [mouse_id for mouse_id in mouse_ids if mouse_id.startswith(first_char)]

            for mouse_id in mouse_ids_category:
                for day in transition_paths_dict[mouse_id].keys():
                    for trial_num in transition_paths_dict[mouse_id][day].keys():
                        transition_path = transition_paths_dict[mouse_id][day][trial_num]
                        transition_matrix = get_transition_pairs(transition_path)
                        base_matrix = dict_transition_matrices[category][day]
                        dict_transition_matrices[category][day] = add_trial_to_counts_matrix(base_matrix, transition_matrix)
                
            
        for category in dict_transition_matrices.keys():
            for day in dict_transition_matrices[category].keys():
                dict_transition_matrices[category][day] = normalize_matrix(dict_transition_matrices[category][day])
                print(f"Final transition matrix for {category} - {day}:")
                display(dict_transition_matrices[category][day])

        return dict_transition_matrices

    def plot_transition_graph(self, prob_matrix,
                            title="Transition Probability Graph", threshold=0.0, ax=None):
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges with weights from probability matrix
        for source in prob_matrix.index:
            for target in prob_matrix.columns:
                prob = prob_matrix.loc[source, target]
                if prob > threshold:  # Only add edges above threshold
                    G.add_edge(source, target, weight=prob)
        
        # Define fixed positions for each state
        fixed_positions = {
            self.config["Behavioral_states"]["approach"]: (0, 0),
            self.config["Behavioral_states"]["crouching"]: (-1, 1), 
            self.config["Behavioral_states"]["active_interaction"]: (1, 1),
            self.config["Behavioral_states"]["pickup"]: (2, 0),
            self.config["Behavioral_states"]["drop"]: (2, -1),
            self.config["Behavioral_states"]["carrying"]: (1, -1),
            self.config["Behavioral_states"]["retrieval"]: (0, -1),
            self.config["Behavioral_states"]["walking"]: (-2, 0),
            self.config["Behavioral_states"]["still"]: (-1, -1),
            self.config["Behavioral_states"]["in_nest"]: (-2, 1)
        }
        
        # Define colors for each state based on annotation colors
        node_colors = {self.config["Behavioral_states"]["approach"]: "green",
                        self.config["Behavioral_states"]["crouching"]: "red",
                        self.config["Behavioral_states"]["active_interaction"]: "dodgerblue",
                        self.config["Behavioral_states"]["pickup"]: "magenta",
                        self.config["Behavioral_states"]["drop"]: "crimson",
                        self.config["Behavioral_states"]["carrying"]: "darkorange",
                        self.config["Behavioral_states"]["retrieval"]: "lime",
                        self.config["Behavioral_states"]["walking"]: "yellow",
                        self.config["Behavioral_states"]["still"]: "gray",
                        self.config["Behavioral_states"]["in_nest"]: "cyan"}
        
        colors = [node_colors[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, fixed_positions, node_color=colors,
                            node_size=3000, alpha=0.3, ax=ax)
        
        # Draw edges with varying width based on probability
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u,v in edges]  # Scale up weights for visibility
        
        # Draw edges with curved arrows and add labels along curves
        for (u, v), weight in zip(edges, weights):
            if u != v:
                # Create curved arrow
                rad = 0.3  # Controls curvature
                connectionstyle = f'arc3,rad={rad}'
                # Scale arrowsize with probability but with reduced scaling
                arrowsize = max(5, G[u][v]['weight'] * 10)  # Reduced max scaling from 20 to 10
                nx.draw_networkx_edges(G, fixed_positions, edgelist=[(u,v)], width=weight,
                                    edge_color='gray', arrowsize=arrowsize,
                                    connectionstyle=connectionstyle, ax=ax)
                
                # Calculate midpoint along the curved path for label placement
                start = fixed_positions[u]
                end = fixed_positions[v]
                
                # Calculate midpoint with curve offset
                mid_x = (start[0] + end[0])/2
                mid_y = (start[1] + end[1])/2
                # Offset perpendicular to edge direction to follow curve
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                mid_x += rad * dy * 0.5  # Offset x based on curve direction
                mid_y -= rad * dx * 0.5  # Offset y based on curve direction
                
                if G[u][v]['weight'] >= 0.3:  # Only label edges with prob >= 0.3
                    ax.annotate(f"{G[u][v]['weight']:.2f}",
                                xy=(mid_x, mid_y),
                                xytext=(0, 0),
                                textcoords='offset points',
                                ha='center',
                                va='center',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            else:
                # Draw self-loops normally
                arrowsize = max(5, G[u][v]['weight'] * 10)  # Reduced max scaling for self-loops too
                nx.draw_networkx_edges(G, fixed_positions, edgelist=[(u,v)], width=weight,
                                    edge_color='gray', arrowsize=arrowsize, ax=ax)
                if G[u][v]['weight'] >= 0.3:
                    # Get node position and place label above for self-loops
                    node_pos = fixed_positions[u]
                    ax.annotate(f"{G[u][v]['weight']:.2f}",
                                xy=(node_pos[0], node_pos[1] + 0.15),
                                xytext=(0, 0),
                                textcoords='offset points',
                                ha='center',
                                va='center',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Add node labels
        nx.draw_networkx_labels(G, fixed_positions, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
        
        if ax is None:
            plt.show()

### transition path functions ####

def load_transition_paths_dict(transition_path_export_dir, mouse_ids, days):
    transition_paths_dict = {}
    for mouse_id in mouse_ids:
        transition_paths_dict[mouse_id] = {}
        # read json file
        with open(f"{transition_path_export_dir}/{mouse_id}_transition_paths_dict.json", "r") as f:
            transition_paths_dict[mouse_id] = json.load(f)[mouse_id]
        
    return transition_paths_dict

# Convert transition sequences into from/to pairs
def get_transition_pairs(transition_path):
    counts_matrix = pd.crosstab(
        pd.Series(transition_path[:-1], name='from'),
        pd.Series(transition_path[1:], name='to'),
        normalize=False
    )
    return counts_matrix

def create_default_counts_matrix(states):

    default_matrix = pd.DataFrame(0,index=pd.Index(states, name='from'),
                        columns=pd.Index(states, name='to'))
    return default_matrix

def add_trial_to_counts_matrix(base_matrix, trials_matrix):
    return base_matrix.add(trials_matrix, fill_value=0)

def normalize_matrix(final_counts_matrix):
     # Create a copy to avoid modifying the original
    normalized = final_counts_matrix.copy()
    # Divide each row by its sum
    normalized = normalized.div(normalized.sum(axis=1), axis=0).fillna(0)
    return normalized


####   Previous analysis functions restricted to only successful trials ####


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