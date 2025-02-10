import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import warnings
import datetime
import re
from sklearn.cluster import DBSCAN
from matplotlib.collections import LineCollection
from colour import Color
import os
import pprint
# Suppress FutureWarning messages
warnings.filterwarnings('ignore') 


###### ------------------- Utility functions ------------------------- #####

def convert_seconds_to_frame(seconds, frame_rate = 30):
    return round(seconds*frame_rate)


###### ----------- Basic Feature extraction from DLC file ------------ #####
# Notes:

class BehaviourFeatureExtractor:

    def __init__(self, path_to_config_file):

        # read .json config file for 
        with open(path_to_config_file) as f:
            self.config = json.load(f)

        self.DLC_cols = self.config['DLC_columns']
        self.DLC_summary_cols = self.config['DLC_summary_columns']
        self.DLC_behaviour_cols = self.config['DLC_behaviour_columns']
        self.time_col = self.DLC_cols['time']
        self.frame_index_col = self.DLC_cols['frame']
        self.frame_rate = self.config['frame_rate_dlc']
        self.minimum_distance_to_nest = self.config['minimum_distance_to_nest']
        self.likelihood_threshold = self.config['likelihood_threshold']

    def compute_average_coordinates(self, df_DLC, parts_list, average_col_name):
        """
        Computes the average coordinates for the parts in the parts_list and adds them to the DataFrame.
        Parameters:
            - df_DLC (pd.DataFrame): DataFrame containing DeepLabCut (DLC) tracking data.
            - parts_list (list): List of body parts to compute the average coordinates for. The parts should be in the DLC_cols dictionary, see config.json.
            - average_col_name (str): Name of the column to store the average coordinates. The column name should be in the DLC_cols dictionary, see config.json.

        Returns:
            pd.DataFrame: DataFrame with the average coordinates added as a new column.
        """

        # compute average coordinates for the parts in the parts_list
        x_avg = df_DLC[[self.DLC_cols[part]["x"] for part in parts_list]].mean(axis=1)
        y_avg = df_DLC[[self.DLC_cols[part]["y"] for part in parts_list]].mean(axis=1)
        likelihood_avg = df_DLC[[self.DLC_cols[part]["likelihood"] for part in parts_list]].mean(axis=1)

        # add the average coordinates to the dataframe
        df_DLC[self.DLC_cols[average_col_name]["x"]] = x_avg
        df_DLC[self.DLC_cols[average_col_name]["y"]] = y_avg
        df_DLC[self.DLC_cols[average_col_name]["likelihood"]] = likelihood_avg

        return df_DLC
    
    def extract_pup_starting_position_bounds(self, df_summary, trial_num):
        """
        Extract the position of the pup in the nest for a given trial
        """
        # get the start and end time of the trial
        trial_num_col = self.DLC_summary_cols["trial_num"]
        pup_starting_position_col = self.DLC_summary_cols["pup_displacement_position"]
        corner_id = df_summary.loc[df_summary[trial_num_col] == trial_num, pup_starting_position_col].values[0]

        # return bounds from config file
        return self.config["pup_position_bounds"][str(corner_id)]

    def extract_trial_from_DLC(self, df_DLC, df_summary, 
                                trial_num):
        # get the trial start and end times
        trial_start_time = df_summary.loc[df_summary[self.DLC_summary_cols["trial_num"]] == trial_num, self.DLC_summary_cols["pup_displacement"]].values[0]
        trial_end_time = df_summary.loc[df_summary[self.DLC_summary_cols["trial_num"]] == trial_num, self.DLC_summary_cols["trial_end"]].values[0]

        # convert trial start and end times to frame indices
        start_frame, end_frame = convert_seconds_to_frame(trial_start_time, self.frame_rate), convert_seconds_to_frame(trial_end_time, self.frame_rate)
        
        # extract the trial data
        mask = (df_DLC[self.frame_index_col] >= start_frame) & (df_DLC[self.frame_index_col] <= end_frame)

        mask = (df_DLC[self.time_col] >= trial_start_time) & (df_DLC[self.time_col] <= trial_end_time)

        return df_DLC.loc[mask, :], mask
    
    def process_trial(self, trial_df_DLC, df_summary, trial_num, interpolate_low_likelihoods = True):
            """
            Processes a trial DataFrame by computing the speed, distance to pup, and head angle to pup

            Parameters:
                - trial_df_DLC (pd.DataFrame): DataFrame containing DeepLabCut (DLC) tracking data for a single trial.
                - trial_num (int): Trial number.
                - interpolate_low_likelihoods (bool, optional): Whether to interpolate low likelihood values. Default is True.
                        
            """

            trial_DLC = trial_df_DLC.copy()

            trial_DLC = self.check_and_insert_processed_columns(trial_DLC)
                
            ### 2. Remove and interpolate low likelihood values for all DLC columns, ignoring nest coordinates
            if interpolate_low_likelihoods == True:
                print("----> Interpolating low likelihood values")
                trial_DLC = self.filter_low_likelihoods_and_interpolate(trial_DLC, body_parts_dict = self.DLC_cols,
                                                                        interpolation_method=self.config["interpolation_method"],
                                                                        threshold = self.likelihood_threshold)
                
            
            ### 3. low level features behaviour features

            ## --- a) compute speed
            print("----> Computing speed")           
            trial_DLC = self.compute_speed(trial_DLC,
                                            x_col = self.DLC_cols["mouse_position"]["x"],
                                            y_col = self.DLC_cols["mouse_position"]["y"],
                                            speed_col = self.DLC_behaviour_cols["mouse_speed"])
            # denoising pup coordinates
            print("----> Denoising pup coordinates")
            trial_DLC, pup_dict = self.track_pup_coordinates_trial(trial_DLC, df_summary, trial_num)
            print("** Check ** NaN counts after denoising: ", trial_DLC[self.DLC_cols["pup"]["x"]].isna().sum())

            print("----> Recomputing distance to pup")
            trial_DLC = self.compute_distance_to_pup(trial_DLC,
                                                    x_col = self.DLC_cols["mouse_position"]["x"],
                                                    y_col = self.DLC_cols["mouse_position"]["y"],
                                                    pup_x_col = self.DLC_cols["pup"]["x"],
                                                    pup_y_col = self.DLC_cols["pup"]["y"],
                                                    distance_col = self.DLC_behaviour_cols["distance_mouse_pup"])
            
            trial_DLC = self.compute_distance_to_pup(trial_DLC,
                                                    x_col = self.DLC_cols["head_position"]["x"],
                                                    y_col = self.DLC_cols["head_position"]["y"],
                                                    pup_x_col = self.DLC_cols["pup"]["x"],
                                                    pup_y_col = self.DLC_cols["pup"]["y"],
                                                    distance_col = self.DLC_behaviour_cols["distance_head_pup"])

            print("----> Recomputing head angle to pup")
            trial_DLC = self.compute_head_angle_to_pup(trial_DLC, add_vector_columns = False,
                                                head_angle_to_pup_col = self.DLC_behaviour_cols["head_angle_to_pup"])

            # computing higher level bhv
            
            # d) add trial number to the dataframe
            trial_DLC[self.DLC_summary_cols["trial_num"]] == trial_num

            return trial_DLC, pup_dict

    #### Compute features ####  

    def compute_speed(self, df_DLC, x_col, y_col, speed_col):

        # compute speed
        distance = np.sqrt(np.diff(df_DLC[x_col])**2 + np.diff(df_DLC[y_col])**2)
        time = np.diff(df_DLC[self.time_col])
        speed = distance/time
        # add speed to the dataframe
        df_DLC[speed_col] = np.append(speed, 0)
        
        return df_DLC

    def compute_distance_to_pup(self, df_DLC,
                                x_col = 'msTop_x', y_col = 'msTop_y',
                                pup_x_col = 'pup_x', pup_y_col = 'pup_y',
                                distance_col = 'distance_to_pup'):
        # compute distance to pup

        distance = np.sqrt((df_DLC[x_col] - df_DLC[pup_x_col])**2 + (df_DLC[y_col] - df_DLC[pup_y_col])**2)
        df_DLC[distance_col] = distance 
        return df_DLC

    def compute_head_angle_to_pup(self, df_DLC, add_vector_columns = False,
                                head_angle_to_pup_col = 'head_angle_to_pup_degrees'):

        # define mouse head direction with respect to average of ears and nose
        earRight_x, earRight_y = df_DLC[self.DLC_cols["earRight"]["x"]], df_DLC[self.DLC_cols["earRight"]["y"]]
        earLeft_x, earLeft_y = df_DLC[self.DLC_cols["earLeft"]["x"]], df_DLC[self.DLC_cols["earLeft"]["y"]]
        nose_x, nose_y = df_DLC[self.DLC_cols["nose"]["x"]], df_DLC[self.DLC_cols["nose"]["y"]]
        pup_x, pup_y  = df_DLC[self.DLC_cols["pup"]["x"]], df_DLC[self.DLC_cols["pup"]["y"]]
        msTop_x, msTop_y = df_DLC[self.DLC_cols["msTop"]["x"]], df_DLC[self.DLC_cols["msTop"]["y"]]

        # compute between ears coordinate
        between_ears_x, between_ears_y = (earRight_x + earLeft_x)/2, (earRight_y + earLeft_y)/2

        # average of between_ears and msTop
        between_ears_x, between_ears_y = (between_ears_x + msTop_x)/2, (between_ears_y + msTop_y)/2

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
        df_DLC[head_angle_to_pup_col] = angle * 180/np.pi
        
        return df_DLC
    
    def flag_nest_coordinates(self, df_dlc, in_nest_col = "in_nest",
                                x = "mouse_x", y = "mouse_y",
                                nest_bounds = {"xmin": 168, "xmax": 300, "ymin": 30, "ymax": 160}):                                #nest_coord_x = "centerNest_x", nest_coord_y = "centerNest_y",
        
        """
        Flags the coordinates in the DataFrame that are within a specified distance to the nest.
        Parameters:
            df_dlc (pd.DataFrame): DataFrame containing the coordinates and nest information.
            in_nest_col (str, optional): Column name to store the flag indicating if the coordinates are in the nest. Default is "in_nest".
            x (str, optional): Column name for the x-coordinate. Default is "msTop_x".
            y (str, optional): Column name for the y-coordinate. Default is "msTop_y".
            nest_coord_x (str, optional): Column name for the x-coordinate of the nest center. Default is "centerNest_x".
            nest_coord_y (str, optional): Column name for the y-coordinate of the nest center. Default is "centerNest_y".
            minimum_distance_to_nest (float, optional): Minimum distance to the nest to consider the coordinates as in the nest. Default is 40.
        
        Returns:
            pd.DataFrame: DataFrame with an additional column indicating if the coordinates are in the nest.
        """

        in_nest = (df_dlc[x] > nest_bounds["xmin"]) & (df_dlc[x] < nest_bounds["xmax"]) & (df_dlc[y] > nest_bounds["ymin"]) & (df_dlc[y] < nest_bounds["ymax"])
        df_dlc[in_nest_col] = in_nest

        return df_dlc
    
    def filter_low_likelihoods_and_interpolate(self, df_DLC, body_parts_dict, interpolation_method = 'time', threshold=0.8):
        """
        Filter out rows with low likelihoods (when mouse is outside of nest) and interpolate the missing values.

        Parameters:
            - df_DLC (pd.DataFrame): DataFrame containing DeepLabCut (DLC) tracking data.
            - body_parts_dict (dict): Dictionary containing the body parts and their coordinates.
            - interpolation_method (str, optional): The method to use for interpolation. Default is 'time'.
            - threshold (float, optional): The likelihood threshold for filtering. Default is 0.8.

        Notes: df_DLC should contain in_nest and mouse_position (average of all mouse coordinates) columns.

        Returns:
        - pd.DataFrame: DataFrame with low likelihood values interpolated.
        """

        df = df_DLC.copy()
        out_of_nest = ~df[self.DLC_behaviour_cols["in_nest"]]
        body_parts = self.config["animal_coordinates"]

        # 0 - filtering low likelihoods for combined mouse position 
        mask = (df[self.DLC_cols["mouse_position"]["likelihood"]] < threshold) & out_of_nest

        ## 1 - marking low likelihood values as NaNs
        for i,val in enumerate(mask):
            if val:
                for body_part in body_parts:
                    # if body part is lower than threshold, mark as NaN
                    if df[body_parts_dict[body_part]["likelihood"]].iloc[i] < threshold:
                        df[body_parts_dict[body_part]["x"]].iloc[i] = np.nan
                        df[body_parts_dict[body_part]["y"]].iloc[i] = np.nan
        
        ## 2 - interpolate missing values
        for body_part in body_parts:
            # print nan count
            # print(f"Number of NaN values in x column {body_part}: ", df[body_parts_dict[body_part]["x"]][out_of_nest].isna().sum())
            # interpolate missing values unsing only outside of nest data
            df[body_parts_dict[body_part]["x"]][out_of_nest] = df[body_parts_dict[body_part]["x"]][out_of_nest].interpolate(method = interpolation_method).ffill().bfill()
            df[body_parts_dict[body_part]["y"]][out_of_nest] = df[body_parts_dict[body_part]["y"]][out_of_nest].interpolate(method = interpolation_method).ffill().bfill()

            # print(f"After interpolation Number of NaN values in x column {body_part}: ", df[body_parts_dict[body_part]["x"]][out_of_nest].isna().sum())


        ## 3 - recompute average coordinates head and mouse
        df = self.compute_average_coordinates(df, self.config["animal_coordinates"],
                                                average_col_name =  "mouse_position")
        df = self.compute_average_coordinates(df, self.config["head_coordinates"],  average_col_name = "head_position")

        return df

    #### PUP TRACKING functions

    def get_pick_up_time(self, df_summary, trial_num):
        """
        Get the pick up time for a given trial from the summary DataFrame.
        
        Parameters:
        -----------
        df_summary : pandas DataFrame
            DataFrame containing summary data
        trial_num : int
            Trial number
            
        Returns:
        --------
        float or None
            Pick up time if it exists, None otherwise
        """
        trial_num_col = self.DLC_summary_cols["trial_num"]
        pick_up_time_col = self.DLC_summary_cols["mouse_first_pick_up"]
        
        pick_up_time = df_summary[df_summary[trial_num_col] == trial_num][pick_up_time_col].values[0]
        return pick_up_time

    def compute_pup_box_center(self, df_summary, trial_num):
        # Get pup box coordinates from df_summary
        pup_bounds = self.extract_pup_starting_position_bounds(df_summary, trial_num)
        pup_box_x = pup_bounds["xmin"] + (pup_bounds["xmax"] - pup_bounds["xmin"]) / 2
        pup_box_y = pup_bounds["ymin"] + (pup_bounds["ymax"] - pup_bounds["ymin"]) / 2

        return pup_box_x, pup_box_y

    def reset_pup_cluster(self, trial_dlc, df_summary, trial_num,
                            start_time_cluster = None,
                            end_time_cluster = None, cluster_label = 0):
        # get summary columns
        start_time_col = self.DLC_summary_cols["pup_displacement"]
        success_col = self.DLC_summary_cols["trial_success"]
        pick_up_time_col = self.DLC_summary_cols["mouse_first_pick_up"]
        end_time_col = self.DLC_summary_cols["trial_end"]
        trial_num_col = self.DLC_summary_cols["trial_num"]
        cluster_label_col = "cluster_label"

        pup_x_col, pup_y_col = self.DLC_cols["pup"]["x"], self.DLC_cols["pup"]["y"]
        time_seconds_col = self.DLC_cols["time"]

        # get start and end times (if not provided)
        if start_time_cluster is None:
            start_time_cluster = df_summary[df_summary[trial_num_col] == trial_num][start_time_col].values[0]
        
        if end_time_cluster is None:
            success = df_summary[df_summary[trial_num_col] == trial_num][success_col].values[0]
            pickup_time = df_summary[df_summary[trial_num_col] == trial_num][pick_up_time_col].values[0]
            end_time_trial = df_summary[df_summary[trial_num_col] == trial_num][end_time_col].values[0]

            if pd.notna(pickup_time): # whether success or not
                end_time_cluster = pickup_time
            else:
                end_time_cluster = end_time_trial

        # time window
        time_window = (trial_dlc[time_seconds_col] >= start_time_cluster) & (trial_dlc[time_seconds_col] <= end_time_cluster)

        # get the pup box center
        pup_box_x, pup_box_y = self.compute_pup_box_center(df_summary, trial_num)

        # reset the pup cluster
        trial_dlc.loc[time_window, [pup_x_col, pup_y_col, cluster_label_col]] = pup_box_x, pup_box_y, cluster_label

        pup_coords = trial_dlc[time_window]

        # create a dictionary with the start and end times of the cluster
        pup_dict_instance = {
                    'cluster_label': cluster_label,
                    'start_time': start_time_cluster,
                    'end_time': end_time_cluster,
                    'data': pup_coords
                }

        print("====> Created one constant position cluster: pup_box_x = ", pup_box_x, " pup_box_y = ", pup_box_y)
        print("====> Number of pup coordinates in the cluster: ", pup_coords.shape[0])
        print("====> Start time: ", seconds_to_mmss(start_time_cluster), " End time: ", seconds_to_mmss(end_time_cluster))
        print("====> trial_dlc preview: -->", trial_dlc[time_window][[cluster_label_col, pup_x_col, pup_y_col]].head(3))

        return trial_dlc, pup_dict_instance

    def compute_distances_intra_coords(self, trial_dlc, cols = ["pupA", "pupB", "pupC"]):
        # computing distances
        dAB = cols[0], cols[1]
        dAC = cols[0], cols[2]
        dBC = cols[1], cols[2]

        for distance in dAB, dAC, dBC:
            point1, point2 = distance
            trial_dlc = self.compute_distance_to_pup(trial_dlc,
                                        x_col = self.DLC_cols[point1]["x"], y_col = self.DLC_cols[point1]["y"],
                                        pup_x_col = self.DLC_cols[point2]["x"], pup_y_col = self.DLC_cols[point2]["y"],
                                        distance_col = f'distance_{point1}_{point2}')
        
            # convert distances to cm
            trial_dlc[f'distance_{point1}_{point2}_cm'] = trial_dlc[f'distance_{point1}_{point2}'] / self.config["pixels_to_cm_ratio"]
            
        return trial_dlc

    def filter_speed_pup(self, trial_dlc, 
                        pup_cols = ["pupA", "pupB", "pupC"],
                        threshold_speed_pup_cm = 5):
        # recompute pup speed
        for pup_col in pup_cols:
            trial_dlc = self.compute_speed(trial_dlc, x_col = self.DLC_cols[pup_col]["x"],
                                                y_col = self.DLC_cols[pup_col]["y"],
                                                speed_col = f"{pup_col}_speed")
            # convert speed to cm
            trial_dlc[f"{pup_col}_speed_cm"] = trial_dlc[f"{pup_col}_speed"] / self.config["pixels_to_cm_ratio"]

            # label likelihood 0 for points with speed > threshold_speed_pup_cm
            mask = (trial_dlc[f"{pup_col}_speed_cm"] > threshold_speed_pup_cm)
            initially_high_likelihood = (trial_dlc[f"{pup_col}_likelihood"] > 0.7) & mask

            print(f"Number of implausible speed measurements for {pup_col}: {mask.sum()}, initially high likelihood: {initially_high_likelihood.sum()}")
            trial_dlc.loc[mask, f"{pup_col}_likelihood"] = 0

        return trial_dlc

    def filter_pick_up_constraint(self, trial_dlc,
                                df_summary,
                                trial_num,
                                pup_cols = ["pupA", "pupB", "pupC"]):

        # get the pick up time
        pick_up_time = self.get_pick_up_time(df_summary, trial_num)

        # get pup displacement bounds
        pup_disp_bounds = self.extract_pup_starting_position_bounds(df_summary, trial_num)

        # define the time window for the pup to stay in the initial position
        if pick_up_time is not None: # trial with pick up
            start_time = trial_dlc["time_seconds"].iloc[0]
            end_time = pick_up_time
        else: # failed trial
            start_time = trial_dlc["time_seconds"].iloc[0]
            end_time = trial_dlc["time_seconds"].iloc[-1]

        pre_pickup_times = (trial_dlc["time_seconds"] > start_time) & (trial_dlc["time_seconds"] < end_time)

        for pup_col in pup_cols:
            # out of pup displacement bounds
            out_of_bounds = (trial_dlc[f"{pup_col}_x"] < pup_disp_bounds["xmin"]) | (trial_dlc[f"{pup_col}_x"] > pup_disp_bounds["xmax"]) | \
                    (trial_dlc[f"{pup_col}_y"] < pup_disp_bounds["ymin"]) | (trial_dlc[f"{pup_col}_y"] > pup_disp_bounds["ymax"])

            implausible_coords = pre_pickup_times & out_of_bounds # whenever pup is out of pup displacement bounds before the pickup time
            initial_high_likelihood = (trial_dlc[f"{pup_col}_likelihood"] > 0.7) & implausible_coords
            print(f"Number of out of bounds coords before pickup for {pup_col}: {implausible_coords.sum()}, initially high likelihood: {initial_high_likelihood.sum()}")
            trial_dlc.loc[implausible_coords, f"{pup_col}_likelihood"] = 0

        return trial_dlc

    def filter_intra_pup_coords(self, trial_dlc,
                                threshold_intra_distance_pup_cm,
                                pup_cols = ["pupA", "pupB", "pupC"]):
        # label implausible pup coords with likelihood < threshold_likelihood_pup

        # ---- 1. outlier detection among the three coordinates
        # could you do a regex string matching for the format distance_pupX_pupY where X, Y are alphabetical characters?
        distance_cols = [col for col in trial_dlc.columns if re.match(r"distance_pup[A-Za-z]_pup[A-Za-z]_cm", col)]
        # print("Distance cols -->", distance_cols)

        for pup_col in pup_cols:
            # print("Pup coord -->", pup_col)
            proximity_distance_cols = [d_col for d_col in distance_cols if pup_col in d_col] # distance cols containing the coordinate
            # print("Total Proximity distance cols -->", proximity_distance_cols)
            dist_col_1, dist_col_2 = proximity_distance_cols[0], proximity_distance_cols[1] 
            # print("Chosen Proximity distance cols -->", dist_col_1, dist_col_2)
            other_distance_cols = [d_col for d_col in distance_cols if d_col not in proximity_distance_cols] # distance col not containing the coordinate
            dist_col_3 = other_distance_cols[0]

            # print("Other distance cols -->", other_distance_cols)
            # print("Distance col not containing the coordinate -->", dist_col_3)

        # mask is True for position where coord is an outlier but the other two coords are close
            mask_outlier = (trial_dlc[dist_col_1] > threshold_intra_distance_pup_cm) & \
                (trial_dlc[dist_col_2] > threshold_intra_distance_pup_cm) & \
                (trial_dlc[dist_col_3] < threshold_intra_distance_pup_cm)

            # outlier_indices = mask_outlier[mask_outlier].index[:3]
            # print(f"\nBefore modification - First 3 outliers for {pup_col}:")
            # for idx in outlier_indices:
            #     print(f"Index {idx}: {pup_col}_likelihood = {trial_dlc.loc[idx, f'{pup_col}_likelihood']}")


            # print("\nProximity distance cols -->", dist_col_1, dist_col_2)
            # print(f"Number of implausible positions for {pup_col}: {mask_outlier.sum()}")
            trial_dlc.iloc[mask_outlier.values, trial_dlc.columns.get_loc(f"{pup_col}_likelihood")] = 0

            # print(f"\nAfter modification - First 3 outliers for {pup_col}:")
            # for idx in outlier_indices:
            #     print(f"Index {idx}: {pup_col}_likelihood = {trial_dlc.loc[idx, f'{pup_col}_likelihood']}")


        # ---- 2. general filter for implausible triple coordinates
        # mask is True for positions where ALL coordinates are far apart from each other
        mask_general = (trial_dlc[dist_col_1] > threshold_intra_distance_pup_cm) & (trial_dlc[dist_col_2] > threshold_intra_distance_pup_cm) & (trial_dlc[dist_col_3] > threshold_intra_distance_pup_cm)

        # print(f"\nNumber of implausible positions for all coordinates: {mask_general.sum()}")
        
        general_indices = mask_general[mask_general].index[:3]
        # print("\nBefore modification - First 3 general outliers:")
        # for idx in general_indices:
        #     for pup_col in cols:
        #         print(f"Index {idx}: {pup_col}_likelihood = {trial_dlc.loc[idx, f'{pup_col}_likelihood']}")

        # label positions with likelihood = 0
        for pup_col in pup_cols:
            trial_dlc.iloc[mask_general.values, trial_dlc.columns.get_loc(f"{pup_col}_likelihood")] = 0        
        # trial_dlc.loc[mask_general, "pup_likelihood"] = 0

        # Verify the modification
        # print("\nAfter modification - Same general outliers:")
        # for idx in general_indices:
        #     for pup_col in cols:
        #         print(f"Index {idx}: {pup_col}_likelihood = {trial_dlc.loc[idx, f'{pup_col}_likelihood']}")


        return trial_dlc

    def detect_pup_clusters(self, trial_dlc, df_summary, trial_num, threshold_likelihood_pup = 0.7, show_plot=False):

        # filter out pup coordinates with likelihood < threshold_likelihood_pup, set x, y, time_seconds
        pup_x_col = self.DLC_cols["pup"]["x"]
        pup_y_col = self.DLC_cols["pup"]["y"]
        time_seconds_col = self.DLC_cols["time"]
        pup_likelihood_col = self.DLC_cols["pup"]["likelihood"]
        cluster_label_col = "cluster_label"

        ## filter out pup coordinates with likelihood < threshold_likelihood_pup
        mask = (trial_dlc[pup_likelihood_col] >= threshold_likelihood_pup)
        pup_coords = trial_dlc.copy()[mask]

        ## Clean up temporary columns
        trial_dlc = trial_dlc.drop(columns=[pup_x_col, pup_y_col])

        ## check empty case
        if pup_coords.empty:
            print("====> No pup coordinates with likelihood above threshold")
            _ , pup_dict_instance = self.reset_pup_cluster(trial_dlc.copy(), df_summary, trial_num,
                                                    start_time_cluster = None,
                                                    end_time_cluster = None, 
                                                    cluster_label = 0)

            clustered_pup_coords = pup_dict_instance["data"][[time_seconds_col, pup_x_col, pup_y_col, cluster_label_col]]                                         

            print(clustered_pup_coords.head(3))

        else:
            pup_coords = pup_coords[[pup_x_col, pup_y_col, time_seconds_col]]

            ## cluster the pup coordinates
            clustered_pup_coords = self.cluster_pup_positions(pup_coords)

        ## reassign the pup_x, pup_y, time_seconds, cluster_label based on the time_seconds
        trial_dlc = trial_dlc.merge(clustered_pup_coords[[time_seconds_col,
                                                        pup_x_col, pup_y_col,
                                                        cluster_label_col]], 
                                    on=time_seconds_col,  # Match on time_seconds column in both DataFrames
                                    how="left")           # Keep all rows from trial_dlc

        print("====== preview trial dlc after clustering")
        print(trial_dlc.loc[trial_dlc[cluster_label_col]==0, [time_seconds_col, pup_x_col, pup_y_col, cluster_label_col]].head(3))
        print("====== in detect_pup_clusters: trial_dlc NaN counts -->", trial_dlc[pup_x_col].isna().sum())
        return trial_dlc

    def cluster_pup_positions(self, df, eps=7, min_samples=7, show_plot=False):
        """
        Cluster pup positions using DBSCAN algorithm.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing pup_x, pup_y, and time_seconds columns
        eps : float
            The maximum distance between two samples for them to be considered neighbors
        min_samples : int
            The minimum number of samples in a neighborhood for a point to be considered a core point
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with original coordinates, time_seconds, and cluster labels
        """
        
        # Extract coordinates for clustering
        X = df[['pup_x', 'pup_y']].values
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'pup_x': df['pup_x'],
            'pup_y': df['pup_y'],
            'time_seconds': df['time_seconds'],
            'cluster_label': cluster_labels
        })

        # Print count of data points in each cluster
        print(f"Number of data points in each cluster: {result_df['cluster_label'].value_counts()}")
        
        # Visualize clusters
        if show_plot:
            plt.figure(figsize=(10, 8))
            
            # Plot points colored by cluster
            scatter = plt.scatter(df['pup_x'], df['pup_y'], 
                                c=cluster_labels, 
                                cmap='viridis',
                                alpha=0.6)
            
            plt.title('Pup Position Clusters')
            plt.xlabel('Pup X Position')
            plt.ylabel('Pup Y Position')
            plt.colorbar(scatter, label='Cluster Label')
            
            # Add note about noise points (label = -1)
            if -1 in cluster_labels:
                plt.text(0.02, 0.98, 'Note: Cluster -1 represents noise points', 
                        transform=plt.gca().transAxes, 
                        fontsize=10, 
                        verticalalignment='top')
            
            plt.show()
        
        return result_df

    def detect_implausible_speeds(self, trial_dlc, start_time, end_time, time_column, pup_x, pup_y, 
                                speed_threshold_cms, pixel_to_cm):
        """
        Detect implausible speeds in a given time window using both x and y coordinates.
        
        Parameters:
        -----------
        trial_dlc : pandas DataFrame
            DataFrame containing the data
        start_time, end_time : float
            Time window to analyze
        time_column : str
            Column name for time data
        pup_x : str
            Column name for x coordinate
        pup_y : str
            Column name for y coordinate
        speed_threshold_cms : float
            Maximum plausible speed in centimeters per second
        pixel_to_cm : float
            Conversion ratio from pixels to centimeters
            
        Returns:
        --------
        int
            Number of implausible speeds detected
        """
        # Get data within time window
        mask = (trial_dlc[time_column] >= start_time) & (trial_dlc[time_column] <= end_time)
        window_data = trial_dlc.loc[mask].copy()
        
        # Calculate positions and times (dropping NaNs but keeping track of indices)
        valid_data = window_data[[time_column, pup_x, pup_y]].dropna()
        if len(valid_data) < 2:
            return 0
        
        # Calculate Euclidean distances between consecutive points (in pixels)
        x_diff = np.diff(valid_data[pup_x].values)
        y_diff = np.diff(valid_data[pup_y].values)
        distances_pixels = np.sqrt(x_diff**2 + y_diff**2)
        
        # Convert distances to centimeters
        distances_cm = distances_pixels / pixel_to_cm
        
        # Calculate time differences in seconds
        time_diff = np.diff(valid_data[time_column].values)
        
        # Calculate speeds in cm/s
        speeds_cms = distances_cm / time_diff
        print("Speeds max: ", max(speeds_cms))
        print("Speeds min: ", min(speeds_cms))
        
        return np.sum(speeds_cms > speed_threshold_cms)

    def has_overlap(self, window1, window2, time_overlap_tolerance=0.2):
        """
        Check if two time windows overlap, including a tolerance buffer.
        
        Parameters:
        -----------
        window1, window2 : dict
            Dictionaries containing 'start_time' and 'end_time'
        time_overlap_tolerance : float
            Time buffer in seconds to consider windows as overlapping
            
        Returns:
        --------
        bool
            True if windows overlap (including tolerance buffer), False otherwise
        """
        # Add tolerance buffer to all times
        start_time_1 = window1['start_time'] - time_overlap_tolerance
        end_time_1 = window1['end_time'] + time_overlap_tolerance
        start_time_2 = window2['start_time'] - time_overlap_tolerance
        end_time_2 = window2['end_time'] + time_overlap_tolerance

        # Check for overlap - this works regardless of window order
        # Two windows overlap if the later window starts before the earlier window ends
        return max(start_time_1, start_time_2) <= min(end_time_1, end_time_2)

    def process_pup_clusters(self, trial_dlc, df_summary, trial_num,
                                speed_threshold_cms=25,
                                time_overlap_tolerance=0.3,
                                distance_between_clusters_cm=5,
                                distance_to_pup_box_cm=10):
        """
        Process pup clusters with speed-based validation and cluster harmonization.
        
        Parameters:
        -----------
        trial_dlc : pandas DataFrame
            DataFrame containing time_seconds, pup_x, pup_y, and cluster_label columns
        df_summary : pandas DataFrame
            DataFrame containing summary data
        speed_threshold_cms : float
            Maximum plausible speed for pup movement in cm/s
        time_overlap_tolerance : float
            Time buffer in seconds to consider clusters as overlapping (default: 0.3s)
            
        Returns:
        --------
        tuple (dict, pandas DataFrame)
            - Dictionary with processed pup clusters
            - Updated trial_dlc DataFrame
        """

        def determine_furthest_cluster(group, trial_dlc, coord_x, coord_y):
            avg_distances = {}
            for cluster in group:
                cluster_mask = trial_dlc['cluster_label'] == cluster['cluster']
                cluster_data = trial_dlc[cluster_mask]
                avg_distances[cluster['cluster']] = compute_distance((cluster_data['pup_x'], cluster_data['pup_y']), (coord_x, coord_y))
            
            # Find cluster with maximum average distance
            furthest_cluster = max(group, key=lambda x: avg_distances[x['cluster']])
            max_distance = avg_distances[furthest_cluster["cluster"]]
            print(f"Furthest cluster: {furthest_cluster['cluster']}")

            return furthest_cluster, max_distance

        def remove_cluster(trial_dlc, cluster_label):
            mask = trial_dlc['cluster_label'] == cluster_label
            trial_dlc.loc[mask, ['pup_x', 'pup_y', 'cluster_label']] = np.nan
            return trial_dlc

        def compute_distance(coords_1, coords_2):
            coords_1_x, coords_1_y = coords_1
            coords_2_x, coords_2_y = coords_2

            return np.sqrt((coords_1_x - coords_2_x)**2 + (coords_1_y - coords_2_y)**2).mean()


        # Make a copy of trial_dlc to modify
        trial_dlc = trial_dlc.copy()
        pup_dict = {}
            
        pick_up_time = self.get_pick_up_time(df_summary, trial_num)

        # Get pup box coordinates from df_summary
        pup_box_x, pup_box_y = self.compute_pup_box_center(df_summary, trial_num)
        
        # Get pixel to cm conversion ratio from config
        pixel_to_cm = self.config["pixels_to_cm_ratio"]
        
        ## Get unique valid clusters, non -1 and non NaN
        valid_clusters = sorted(trial_dlc[(trial_dlc['cluster_label'] != -1) & \
                                (~trial_dlc['cluster_label'].isna())]['cluster_label'].unique())

        # turn all pup_x, pup_y for NaN and -1 clusters to pup         
        invalid_clusters = [-1] + [np.nan]
        trial_dlc.loc[trial_dlc['cluster_label'].isin(invalid_clusters), ['pup_x', 'pup_y', 'cluster_label']] = np.nan
        
        ## Extract time windows for each cluster
        cluster_windows = []
        for cluster in valid_clusters:
            pup_cluster_mask = trial_dlc['cluster_label'] == cluster
            window = {
                'cluster': cluster,
                'start_time': trial_dlc.loc[pup_cluster_mask, 'time_seconds'].min(),
                'end_time': trial_dlc.loc[pup_cluster_mask, 'time_seconds'].max(),
                'n_points': pup_cluster_mask.sum()
            }
            cluster_windows.append(window)
        
        # Sort clusters by start time
        cluster_windows.sort(key=lambda x: x['start_time'])

        print("Cluster windows: ", len(cluster_windows))
        pprint.pprint(cluster_windows)
        
        ## Group overlapping clusters
        overlap_groups = []
        current_group = []
        for window in cluster_windows:
            if not current_group or any(self.has_overlap(window, w, time_overlap_tolerance) for w in current_group):
                current_group.append(window)
            else:
                overlap_groups.append(current_group)
                current_group = [window]
        if current_group:
            overlap_groups.append(current_group)
        
        print("== * == * Groups: ", len(overlap_groups))
        pprint.pprint(overlap_groups)
        
        for i, group in enumerate(overlap_groups, 1):

            ## Cleaning the cluster, removing implausible time-overlapping clusters
            while len(group) > 1:
                # Calculate average distance to pup_box for each cluster      
                furthest_cluster, max_distance = determine_furthest_cluster(group, trial_dlc, pup_box_x, pup_box_y)
                max_distance_cm = max_distance / pixel_to_cm
                smallest_cluster = min(group, key=lambda x: x['n_points'])
                print(f"Max distance to pup box: {max_distance_cm} cm")

                speeds = self.detect_implausible_speeds(
                    trial_dlc,
                    min(w['start_time'] for w in group),
                    max(w['end_time'] for w in group),
                    'time_seconds',
                    'pup_x',
                    'pup_y',
                    speed_threshold_cms,
                    pixel_to_cm
                )

                if speeds > 0:

                    coords_furthest = (trial_dlc.loc[trial_dlc['cluster_label'] == furthest_cluster['cluster'], ['pup_x', 'pup_y']]).mean()
                    coords_smallest = (trial_dlc.loc[trial_dlc['cluster_label'] == smallest_cluster['cluster'], ['pup_x', 'pup_y']]).mean()

                    distance_smallest_to_furthest = compute_distance((coords_smallest['pup_x'], coords_smallest['pup_y']),
                                                                    (coords_furthest['pup_x'], coords_furthest['pup_y']))
                                                                    
                    distance_smallest_to_furthest_cm = distance_smallest_to_furthest / pixel_to_cm
                    
                    print(f"Distance between smallest and furthest cluster: {distance_smallest_to_furthest_cm} cm")
                    
                    if distance_smallest_to_furthest > distance_between_clusters_cm:
                        # remove the furthest cluster, as it is too far from the pup box
                        cluster_to_remove = furthest_cluster
                        print("Removed furthest cluster - too far from the pup box")
                    else:
                        # remove the smallest cluster, as it close to the furthest cluster but not as representative
                        cluster_to_remove = smallest_cluster
                        if distance_smallest_to_furthest == 0:
                            print(f"Furthest cluster = Smallest cluster")
                        else:
                            print("Removed smallest cluster - was close to the furthest cluster")

                    trial_dlc = remove_cluster(trial_dlc = trial_dlc, cluster_label = cluster_to_remove['cluster'])
                    group.remove(cluster_to_remove)
                    print(f"---> Removed cluster was {cluster_to_remove['cluster']}")
                else:
                    break
            
            remaining_clusters = [w['cluster'] for w in group]
            print(f"Remaining clusters: {remaining_clusters}")
            new_label = min(remaining_clusters)
            to_remove = []

            ## Cleaning the cluster, removing implausible intra-group clusters
            for cluster in group:
                removed = False
                cluster_id = cluster['cluster']
                mask_cluster = trial_dlc['cluster_label'] == cluster_id
                # logic to check if the cluster is too far from the pup box when there was no interaction with the pup
                if pd.isna(pick_up_time):
                    print("No pick up time")
                    # get average coordinate over the cluster
                    pup_x_avg, pup_y_avg = trial_dlc.loc[mask_cluster, ['pup_x', 'pup_y']].mean()
                    mean_distance_to_pup_box = compute_distance((pup_x_avg, pup_y_avg), (pup_box_x, pup_box_y))
                    mean_distance_to_pup_box_cm = mean_distance_to_pup_box / pixel_to_cm

                    print(f"Mean distance to pup box: {mean_distance_to_pup_box_cm} cm")
                    if mean_distance_to_pup_box_cm > distance_to_pup_box_cm:
                        trial_dlc = remove_cluster(trial_dlc = trial_dlc, cluster_label = cluster_id)
                        to_remove.append(cluster_id)
                        removed = True
                        print(f"Removed cluster {cluster_id} from group")

                if not removed:
                    ## Cluster is harmonized with the unified new label
                    start_subcluster = cluster['start_time']
                    end_subcluster = cluster['end_time']
                    mask = (trial_dlc['time_seconds'] >= start_subcluster) & (trial_dlc['time_seconds'] <= end_subcluster)
                    trial_dlc.loc[mask, 'cluster_label'] = new_label
            
            group = [cluster for cluster in group if cluster['cluster'] not in to_remove]
            
            ## Special constraints for the first group of clusters 'pup1'
            if i == 1 and len(group) != 0:
                first_cluster = trial_dlc.loc[trial_dlc['cluster_label'] == new_label]
                print("First cluster: ", new_label)

                # get average distance to pup box
                pup_x_avg, pup_y_avg = first_cluster['pup_x'].mean(), first_cluster['pup_y'].mean()
                mean_distance_to_pup_box = compute_distance((pup_x_avg, pup_y_avg), (pup_box_x, pup_box_y))
                mean_distance_to_pup_box_cm = mean_distance_to_pup_box / pixel_to_cm
                print(f"Mean distance to pup box: {mean_distance_to_pup_box_cm} cm")

                if mean_distance_to_pup_box_cm > distance_to_pup_box_cm:
                    trial_dlc = remove_cluster(trial_dlc = trial_dlc, cluster_label = new_label)
                    group.remove(cluster)
                    print(f"Removed cluster {new_label} from group - first cluster too far from the pup box")

                # if the first cluster occurs after the pick up time, then remove it
                elif pd.notna(pick_up_time) and first_cluster['time_seconds'].min() > pick_up_time:
                    trial_dlc = remove_cluster(trial_dlc = trial_dlc, cluster_label = new_label)
                    group.remove(cluster)
                    print(f"Removed cluster {new_label} from group - found first cluster starting after pick up time")
                    
            ##  Saving the cluster
            if group:
                start_time = min(w['start_time'] for w in group)
                end_time = max(w['end_time'] for w in group)
                mask_final_cluster = (trial_dlc['time_seconds'] >= start_time) & (trial_dlc['time_seconds'] <= end_time)

                # nan filling
                trial_dlc.loc[mask_final_cluster, ['pup_x', 'pup_y', 'cluster_label']] = trial_dlc.loc[mask_final_cluster, ['pup_x', 'pup_y', 'cluster_label']].ffill().bfill()
                        
                # Get the data for this time window
                pup_data = trial_dlc[mask_final_cluster].copy()

                pup_dict[f'pup_location{i}'] = {
                    'cluster_label': new_label,
                    'start_time': start_time,
                    'end_time': end_time,
                    'data': pup_data
                }

        if 'pup_location1' not in pup_dict:

            print("Resetting first cluster pup1")
            # reset the first cluster
            start_time_cluster = None
            end_time_cluster = None
            cluster_label = 0

            if pup_dict != {}:
                next_pup_label = min(pup_dict.keys())
                start_time_next_cluster = pup_dict[next_pup_label]['start_time']
                end_time_conventional = pick_up_time if pd.notna(pick_up_time) else trial_dlc['time_seconds'].max()
                end_time_cluster = min(end_time_conventional, start_time_next_cluster)

            trial_dlc, pup1_instance = self.reset_pup_cluster(trial_dlc, df_summary, trial_num,
                                                        start_time_cluster = start_time_cluster,
                                                        end_time_cluster = end_time_cluster, 
                                                        cluster_label = cluster_label)

            pup_dict['pup_location1'] = pup1_instance
        
        return pup_dict, trial_dlc

    def compute_pup_average(self, row, 
                            pup_cols = ["pupA", "pupB", "pupC"],
                            threshold_likelihood_pup=0.7):
        """
        Computes average of pup coordinates whose likelihood exceeds the threshold.
        Uses pandas operations for better efficiency.

        Parameters:
            row (pd.Series): Row of the dataframe containing pup coordinates and likelihoods
            pup_coords_dict (dict): Dictionary containing column names for each pup's coordinates
            likelihood_threshold (float): Minimum likelihood value to include coordinate in average

        Returns:
            tuple: (x_avg, y_avg, likelihood_avg) averaged coordinates and likelihood
        """
        # Create lists of coordinate columns

        DLC_columns = self.DLC_cols

        valid_info = {pup_col: (row[DLC_columns[pup_col]["x"]], row[DLC_columns[pup_col]["y"]], row[DLC_columns[pup_col]["likelihood"]]) for pup_col in pup_cols if row[DLC_columns[pup_col]["likelihood"]] >= threshold_likelihood_pup}
    
        all_info = {pup_col: (row[DLC_columns[pup_col]["x"]], row[DLC_columns[pup_col]["y"]], row[DLC_columns[pup_col]["likelihood"]]) for pup_col in pup_cols}
        # if no pup clusters are detected
        if len(valid_info) == 0:
            x_avg = np.mean([all_info[pup_col][0] for pup_col in all_info])
            y_avg = np.mean([all_info[pup_col][1] for pup_col in all_info])
            likelihood_avg = 0
            return x_avg, y_avg, likelihood_avg
        
        # compute the average coords for the pup
        x_avg = np.mean([valid_info[pup_col][0] for pup_col in valid_info])
        y_avg = np.mean([valid_info[pup_col][1] for pup_col in valid_info])
        likelihood_avg = np.mean([valid_info[pup_col][2] for pup_col in valid_info])
        
        #print("Number of valid pup coords: ", len(valid_info))
        return x_avg, y_avg, likelihood_avg
     
    def track_pup_coordinates_trial(self, trial_dlc, df_summary, trial_num):
        """
        Tracks the coordinates of the pup in the trial.
        """
        print("Before filtering:")
        for col in ["pupA_likelihood", "pupB_likelihood", "pupC_likelihood"]:
            print(f"Zeros in {col}: {(trial_dlc[col] == 0).sum()}")

        ######## 1. compute distances between pup coordinates
        trial_dlc = self.compute_distances_intra_coords(trial_dlc)

        ######## 2. filter out implausible pup coordinates
        trial_dlc = self.filter_intra_pup_coords(trial_dlc, threshold_intra_distance_pup_cm=2.5)

        print("\nAfter intra pup filter:")
        for col in ["pupA_likelihood", "pupB_likelihood", "pupC_likelihood"]:
            print(f"Zeros in {col}: {(trial_dlc[col] == 0).sum()}")

        ######## 3. filter out implausible pup speeds
        trial_dlc = self.filter_speed_pup(trial_dlc,
                                        threshold_speed_pup_cm = 5)
        print("\nAfter speed filter:")
        for col in ["pupA_likelihood", "pupB_likelihood", "pupC_likelihood"]:
            print(f"Zeros in {col}: {(trial_dlc[col] == 0).sum()}")

        ######## 4. recompute the pup average
        pup_coords = { key:values for key, values in self.config["DLC_columns"].items() if key.startswith("pup") and len(key) == 4}
        trial_dlc[['pup_x','pup_y', 'pup_likelihood']] = trial_dlc.apply(lambda row: self.compute_pup_average(row, threshold_likelihood_pup = 0.7), axis = 1, result_type = 'expand')
        # count number of NaNs in pup_average
        print(f"Number of NaNs in pup_average: {trial_dlc['pup_x'].isna().sum()}")

        ms_id = df_summary[self.DLC_summary_cols["animal_id"]].values[0]
        d = df_summary[self.DLC_summary_cols["day"]].values[0]
        
        ######## 5. visualize the pup trajectory
        visualize_pup_trajectory(mouse_id = ms_id, day = d, trial_num = trial_num,
                                cmap = "viridis",
                                trial_dlc = trial_dlc,
                                skip_average = True,
                                df_summary = df_summary, 
                                df_dlc = trial_dlc,
                                path_dir = None,  
                                BF_instance = self,
                                BF_config = self.config,
                                threshold_likelihood_pup = 0.7)


        ######## 6. detect pup clusters
        trial_dlc = self.detect_pup_clusters(trial_dlc, df_summary, trial_num, threshold_likelihood_pup = 0.7, show_plot=True)
        
        ######## 7. filter out implausible pup coordinates (at this point NaNs will be added into pup_x, pup_y)
        pup_dict, trial_dlc = self.process_pup_clusters(trial_dlc, df_summary, trial_num)

        # pprint.pprint(pup_dict)

        ######## 8. re-visualize the pup trajectory
        visualize_pup_trajectory(mouse_id = ms_id, day = d, trial_num = trial_num,
                            cmap = "magma",
                            trial_dlc = trial_dlc,
                            pup_dict = pup_dict,
                            skip_average = True,
                            df_summary = df_summary, 
                            df_dlc = trial_dlc,
                            path_dir = f"plots/pup_trajectory_transformed/{ms_id}",  
                            BF_instance = self,
                            BF_config = self.config,
                            threshold_likelihood_pup = 0.7)

        return trial_dlc, pup_dict
    
    #### Process the DLC data ####
    
    def process_DLC(self, df_DLC, df_summary, interpolate_low_likelihoods = True):
        """
        Extracts base parameters such as speed, distance to pup, and head angle to pup for each trial 
        from the given DataFrame. Updates a dictionary mapping trial number to the extracted trial data.
        Parameters:
        df_DLC (pd.DataFrame): DataFrame containing DeepLabCut (DLC) tracking data with a 'frame_index' column.
        df_summary (pd.DataFrame): DataFrame containing summary information for each trial, including 
                                'BehavRecdTrialEndSecs' and 'PupDispDropSecs' columns.
        Returns:
        pd.DataFrame: Updated DataFrame with additional columns for speed, 
                    distance to pup and head angle to pup.
                    These columns have default NaN values for frames that don't belong to any trial.
        trials_dict (dict): A dictionary containing the extracted trial data for each trial.
        """ 

        df_DLC = df_DLC.copy()
        modified_df_DLC = df_DLC.copy()

        modified_df_DLC = self.check_and_insert_processed_columns(modified_df_DLC)
        original_df_DLC = self.check_and_insert_processed_columns(df_DLC.copy())

        # print("** Check ** original_df_DLC columns:", original_df_DLC.columns)
        # print("** Check ** original_df_DLC NaN counts:", original_df_DLC["pup_x"].isna().sum())

        # print("** Check ** modified_df_DLC columns:", modified_df_DLC.columns)
        # print("** Check ** modified_df_DLC NaN counts:", modified_df_DLC["pup_x"].isna().sum())

        trials_dict = {}
        trial_nums = df_summary[self.DLC_summary_cols["trial_num"]]

        relevant_cols = list(self.DLC_behaviour_cols.values())
        # add the new columns to the modified_df_DLC    
        # for col in relevant_cols:
        #     if col not in modified_df_DLC.columns:
        #         print(f"** ADDED NEW COLUMN ** column {col} not in modified_df_DLC")
        #         modified_df_DLC[col] = np.nan

        # iterate over each trial
        for trial_num in  trial_nums:

                trial_DLC, mask_DLC = self.extract_trial_from_DLC(original_df_DLC, df_summary, trial_num)
                trial_DLC, pup_dict = self.process_trial(trial_DLC, df_summary, trial_num, interpolate_low_likelihoods = interpolate_low_likelihoods)
                
                trials_dict[trial_num] = trial_DLC #update the dictionary with the trial data
                print("** Check ** trial_DLC NaN counts:", trial_DLC["pup_x"].isna().sum())
                # print("** Check ** trial_DLC columns:", trial_DLC.columns)
                # print("Preview first three rows of trial_DLC:")
                # print(trial_DLC[["pup_x", "pup_y", "pup_likelihood", "cluster_label", "head_angle_to_pup_degrees", "distance_head_to_pup"]].head(3))
                # print("Indices of the first three rows in trial_DLC:", trial_DLC.index.values[:3])

                if "cluster_label" not in trial_DLC.columns:
                    print("** ADDED NEW COLUMN ** column cluster_label not in trial_DLC")
                    modified_df_DLC["cluster_label"] = np.nan

                # modified_df_DLC.loc[mask_DLC, col] = trial_DLC[col]
                # print(f"** ADDED COLUMN ** column {col} added to modified_df_DLC")

                # Then update the values using loc
                modified_df_DLC.loc[mask_DLC, modified_df_DLC.columns] = trial_DLC[modified_df_DLC.columns].values
                #print("** Check ** modified_df_DLC columns:", modified_df_DLC.columns)
                print("** Check ** modified_df_DLC NaN counts:", modified_df_DLC[mask_DLC]["pup_x"].isna().sum())
                # print("** Check ** modified_df_DLC NaN counts TOTAL:", modified_df_DLC["pup_x"].isna().sum())

                # print("Preview first three rows of trial in modified_df_DLC:")
                # print(modified_df_DLC.loc[mask_DLC][["pup_x", "pup_y", "pup_likelihood", "cluster_label", "head_angle_to_pup_degrees", "distance_head_to_pup"]].head(3)) 

                # export the pup_location_dict as a json file, export only the start_time, end_time, and cluster_label of the clusters, and save it in the same directory as the trial_DLC
                ms_id = df_summary[self.DLC_summary_cols["animal_id"]].values[0]
                d = df_summary[self.DLC_summary_cols["day"]].values[0]

                # create the directory if it doesn't exist
                os.makedirs(f"processed_data/{ms_id}/{d}/trials/", exist_ok=True)

                with open(f"processed_data/{ms_id}/{d}/trials/{ms_id}_{d}_trial{trial_num}_pup_location_dict.json", "w") as f:
                    # export only the start_time, end_time, and cluster_label of the clusters
                    pup_dict_to_export = {key: {"start_time": value["start_time"], "end_time": value["end_time"], "cluster_label": value["cluster_label"]} for key, value in pup_dict.items()}
                    json.dump(pup_dict_to_export, f)

        return modified_df_DLC, trials_dict
    
    def check_and_insert_processed_columns(self, df):
        """
        Checks if the processed or extra columns are present in the DataFrame.
        Columns are:
            If any of the following columns are missing, they are computed and added to the DataFrame.
            - mouse_position (average of all mouse coordinates)
            - head_position (average of all head coordinates)
            - in_nest (flag indicating if the mouse is in the nest)

            If any of the following columns are missing, they are added to the DataFrame with NaN values:
            - trial_num (trial number)
            - mouse_speed (speed of the mouse)
            - pup_speed (speed of the pup)
            - distance_mouse_pup (distance between the mouse and the pup)
            - distance_head_pup (distance between the head and the pup)
            - head_angle_to_pup (angle between the head direction and the pup direction)
            
        Parameters:
            - df (pd.DataFrame): DataFrame to check. Can be either a trial DataFrame or the full DLC DataFrame.
        Returns:
            bool: True if all columns are present, False otherwise.

        """
        df = df.copy()

        coordinates_cols = [self.DLC_cols["mouse_position"]["likelihood"],
                            self.DLC_cols["head_position"]["likelihood"], 
                            self.DLC_cols["pup"]["likelihood"]]
        trial_num_col = [self.DLC_summary_cols["trial_num"]] #+ ["cluster_label"]
        behavioural_cols = list(self.DLC_behaviour_cols.values()) #+ ["cluster_label"]

        columns = coordinates_cols + behavioural_cols + trial_num_col

        for col in columns:
            if col not in df.columns:
                if col == self.DLC_cols["mouse_position"]["likelihood"]:
                    df = self.compute_average_coordinates(df, self.config["animal_coordinates"],
                                                average_col_name =  "mouse_position")
                
                elif col == self.DLC_cols["head_position"]["likelihood"]:
                    df = self.compute_average_coordinates(df, self.config["head_coordinates"], 
                                                    average_col_name = "head_position")
                    
                elif col == self.DLC_cols["pup"]["likelihood"]:
                    df = self.compute_average_coordinates(df, self.config["pup_coordinates"], 
                                                    average_col_name = "pup")

                elif col == self.DLC_behaviour_cols["in_nest"]:
                    df = self.flag_nest_coordinates(df, in_nest_col = self.DLC_behaviour_cols["in_nest"], 
                                                    x = self.DLC_cols["mouse_position"]["x"], y = self.DLC_cols["mouse_position"]["y"],
                                                    nest_bounds = self.config["nest_bounds"])

                else:
                    df[col] = np.nan
            
        return df



#### ------------ Plotting utility functions ------------- #####

def plot_mouse_angle_to_pup(trial_df_DLC,
                            ylim = None,
                            xlim = None,
                            frame_index_number = None,
                            frame_index_col = 'frame_index',
                            msTop_x_col = 'msTop_x',
                            msTop_y_col = 'msTop_y',
                            earRight_x_col = 'earRight_x',
                            earRight_y_col = 'earRight_y',
                            earLeft_x_col = 'earLeft_x',
                            earLeft_y_col = 'earLeft_y',
                            nose_x_col = 'nose_x',
                            nose_y_col = 'nose_y',
                            pup_x_col = 'pup_x',
                            pup_y_col = 'pup_y',
                            between_ears_x_col = 'between_ears_x',
                            between_ears_y_col = 'between_ears_y',
                            head_angle_to_pup = 'head_angle_to_pup_degrees'):

    # select a random frame in the trial
    if frame_index_number is None:
        frame_index_min, frame_index_max = trial_df_DLC[frame_index_col].min(), trial_df_DLC[frame_index_col].max()
        frame_index_number = random.randint(frame_index_min, frame_index_max)
    trial_1_DLC_frame = trial_df_DLC.loc[trial_df_DLC[frame_index_col] == frame_index_number, :]

    # plot the frame
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if xlim is None or ylim is None:
        xlim, ylim =  max(trial_df_DLC[msTop_x_col].max(), trial_df_DLC[pup_x_col].max()), max(trial_df_DLC[msTop_y_col].max(), trial_df_DLC[pup_y_col].max())

    trial_1_DLC_frame.plot(x=earRight_x_col, y=earRight_y_col, style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'black')
    trial_1_DLC_frame.plot(x=earLeft_x_col, y=earLeft_y_col, style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'black')
    trial_1_DLC_frame.plot(x=nose_x_col, y=nose_y_col, style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'red')
    trial_1_DLC_frame.plot(x=pup_x_col, y=pup_y_col, style='o', ax=ax,xlim=(0, xlim), ylim=(0, ylim), color = 'purple')
    trial_1_DLC_frame.plot(x=between_ears_x_col, y=between_ears_y_col, style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'blue')

    # draw lines
    # draw line from nose to between ears in blue
    ax.plot([trial_1_DLC_frame[nose_x_col].values[0], trial_1_DLC_frame[between_ears_x_col].values[0]],
            [trial_1_DLC_frame[nose_y_col].values[0], trial_1_DLC_frame[between_ears_y_col].values[0]], 'k-', color = 'blue')

    # draw line from between ears to pup in green
    ax.plot([trial_1_DLC_frame[between_ears_x_col].values[0], trial_1_DLC_frame[pup_x_col].values[0]],
            [trial_1_DLC_frame[between_ears_y_col].values[0], trial_1_DLC_frame[pup_y_col].values[0]], 'k-', color = 'green')

    # add angle value to the plot in the title
    angle = trial_1_DLC_frame[head_angle_to_pup].values[0]

    ax.set_title("Angle between mouse head direction and pup direction: " + str(angle) + " degrees")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

def seconds_to_mmss(seconds):
    if pd.isna(seconds):
        return "N/A"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

def get_pick_up_time(df_summary, trial_num, BF):
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
        trial_num_col = BF.config["DLC_summary_columns"]["trial_num"]
        pick_up_time_col = BF.config["DLC_summary_columns"]["mouse_first_pick_up"]
        return df_summary.loc[df_summary[trial_num_col] == trial_num, pick_up_time_col].values[0]

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def visualize_pup_trajectory(mouse_id, day, trial_num,
                            df_summary, df_dlc, 
                            path_dir, 
                            BF_instance,
                            BF_config,
                            cmap = "plasma",
                            pup_dict = None,
                            skip_average = False,
                            trial_dlc = None,
                            threshold_likelihood_pup = 0.7): 
    if trial_dlc is None:
        trial_dlc, _ = BF_instance.extract_trial_from_DLC(df_dlc, df_summary, trial_num = trial_num)

    trial_success = df_summary.loc[df_summary[BF_config["DLC_summary_columns"]["trial_num"]] == trial_num,
                                    BF_config["DLC_summary_columns"]["trial_success"]].values[0]
    pup_disp_pos = df_summary.loc[df_summary[BF_config["DLC_summary_columns"]["trial_num"]] == trial_num,
                                    BF_config["DLC_summary_columns"]["pup_displacement_position"]].values[0]
    pup_position_bounds = BF_instance.extract_pup_starting_position_bounds(df_summary, trial_num)

    pick_up_time = BF_instance.get_pick_up_time(df_summary, trial_num)
    start_trial = trial_dlc["time_seconds"].iloc[0]
    end_trial = trial_dlc["time_seconds"].iloc[-1]

    fig, ax = plt.subplots(figsize = (10,10))

    if not skip_average:
        ##### plot the average of all ABC
        pup_average_likelihood = np.mean([trial_dlc[BF_config["DLC_columns"]["pupA"]["likelihood"]],
                                        trial_dlc[BF_config["DLC_columns"]["pupB"]["likelihood"]],
                                        trial_dlc[BF_config["DLC_columns"]["pupC"]["likelihood"]]], axis = 0)
        pup_average_y = np.mean([trial_dlc[BF_config["DLC_columns"]["pupA"]["y"]], 
                                trial_dlc[BF_config["DLC_columns"]["pupB"]["y"]],
                                trial_dlc[BF_config["DLC_columns"]["pupC"]["y"]]], axis = 0)
        pup_average_x = np.mean([trial_dlc[BF_config["DLC_columns"]["pupA"]["x"]],
                                trial_dlc[BF_config["DLC_columns"]["pupB"]["x"]],
                                trial_dlc[BF_config["DLC_columns"]["pupC"]["x"]]], axis = 0)

        trial_dlc["pup_x"] = pup_average_x
        trial_dlc["pup_y"] = pup_average_y
        trial_dlc["pup_likelihood"] = pup_average_likelihood
    
    color = (trial_dlc["time_seconds"] - trial_dlc["time_seconds"].iloc[0])

    # plot the time color on the trajectory
    lines = colored_line(trial_dlc["pup_y"], trial_dlc["pup_x"], color, ax, linewidth=2, zorder = 5, cmap=cmap, alpha = 0.7)
    cbar = fig.colorbar(lines, label="Time (s)")  # store colorbar reference

    # Add lines for each cluster's start/end times
    if pup_dict:
        # Get colorbar axis
        cax = cbar.ax
                
        # Create different colors for different clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple']  # extend if needed
        
        for i, (cluster_name, cluster_data) in enumerate(pup_dict.items()):
            start_time = cluster_data['start_time']
            end_time = cluster_data['end_time']
            
            # Convert times to normalized positions
            start_pos = start_time - start_trial
            end_pos = end_time - start_trial
            
            # Add horizontal lines at start and end times
            color = colors[i % len(colors)]
            cax.axhline(y=start_pos, color=color, linestyle='--', linewidth=1)
            cax.axhline(y=end_pos, color=color, linestyle='--', linewidth=1)
            
            # Add text labels
            cax.text(1.5, start_pos, f'{cluster_name} start', 
                    color=color, ha='left', va='center')
            cax.text(1.5, end_pos, f'{cluster_name} end',
                    color=color, ha='left', va='center')

    # add pickup point indicator
    if not pd.isna(pick_up_time):
        cax = cbar.ax
        pickup_pos = pick_up_time - start_trial
        cax.axhline(y=pickup_pos, color="magenta", linestyle='--', linewidth=1)
        cax.text(1.5, pickup_pos, f'pickup: {pickup_pos:.0f}s', color="magenta", ha='left', va='center')


    # only plot points > likelihood threshold
    mask = trial_dlc["pup_likelihood"] < threshold_likelihood_pup
    trial_dlc[~mask].plot(x = "pup_y", y = "pup_x",
                kind = "scatter",
                marker = "o",
                ax = ax,
                s = 30,
                alpha = 0.4,
                zorder = 7,
                color = "green", label = "pup_ABC")
    
    # add a red cross if likelihood is below threshold_likelihood_pup
    trial_dlc.loc[mask].plot(x = "pup_y",
                            y = "pup_x",
                            marker = "x",
                            kind = "scatter",
                            ax = ax,
                            alpha = 0.4,
                            s = 30,
                            color = "red",
                            zorder = 5,
                            label = f"pup_low_likelihood < {threshold_likelihood_pup}")
    
    trial_dlc.iloc[0:1].plot(x = "pup_y", y = "pup_x",
                        kind = "scatter",
                        marker = "*",
                        alpha = 1.,
                        s = 200,
                        edgecolors='black',
                        ax = ax,
                        color = "cyan", label = "start",zorder = 10)
    trial_dlc.iloc[-1:].plot(x = "pup_y", y = "pup_x",
                            kind = "scatter",
                            marker = "*",
                            alpha = 1.,
                            s = 200,
                            edgecolors='black',
                            ax = ax,
                            color = "green", label = "end", zorder = 10)
        
    # set x and y labels
    ax.set_xlabel("pup y")
    ax.set_ylabel("pup x")

    arena_bounds = BF_config["arena_bounds"]
    nest_bounds = BF_config["nest_bounds"]
        
    # set x and y limits
    ax.set_xlim([arena_bounds["ymin"], arena_bounds["ymax"]])
    ax.set_ylim([arena_bounds["xmin"], arena_bounds["xmax"]])
        
    # plot the arena
    ax.plot([nest_bounds["ymin"], nest_bounds["ymax"]], [nest_bounds["xmin"], nest_bounds["xmin"]], 'k--')
    ax.plot([nest_bounds["ymin"], nest_bounds["ymax"]], [nest_bounds["xmax"], nest_bounds["xmax"]], 'k--')
    ax.plot([nest_bounds["ymin"], nest_bounds["ymin"]], [nest_bounds["xmin"], nest_bounds["xmax"]], 'k--')
    ax.plot([nest_bounds["ymax"], nest_bounds["ymax"]], [nest_bounds["xmin"], nest_bounds["xmax"]], 'k--', label = "nest_bounds")

    # plot the pup position bounds
    ax.plot([pup_position_bounds["ymin"], pup_position_bounds["ymax"]], [pup_position_bounds["xmin"], pup_position_bounds["xmin"]], 'r--')
    ax.plot([pup_position_bounds["ymin"], pup_position_bounds["ymax"]], [pup_position_bounds["xmax"], pup_position_bounds["xmax"]], 'r--')
    ax.plot([pup_position_bounds["ymin"], pup_position_bounds["ymin"]], [pup_position_bounds["xmin"], pup_position_bounds["xmax"]], 'r--')
    ax.plot([pup_position_bounds["ymax"], pup_position_bounds["ymax"]], [pup_position_bounds["xmin"], pup_position_bounds["xmax"]], 'r--', label = "pup_position_bounds")

    ax.legend()
        
    start_trial_s = seconds_to_mmss(start_trial)
    pick_up_time_s = seconds_to_mmss(pick_up_time)

    title = f"Mouse id: {mouse_id}, Trial : {trial_num} Session : {day}, Success : {trial_success}, Start time: {start_trial_s}, Pick-up time: {pick_up_time_s}"
    ax.set_title(title)
    
    if path_dir is not None:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        plt.savefig(f"{path_dir}/{mouse_id}_{day}_trial_{trial_num}.png")
    plt.show()

