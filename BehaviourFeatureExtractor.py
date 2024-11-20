import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import warnings
import datetime

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

        return df_DLC.loc[mask, :], mask
    
    def process_trial(self, trial_df_DLC, trial_num, interpolate_low_likelihoods = True):
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
            trial_DLC = self.compute_speed(trial_DLC, x_col = self.DLC_cols["pup"]["x"],
                                            y_col = self.DLC_cols["pup"]["y"],
                                            speed_col = self.DLC_behaviour_cols["pup_speed"])

            ## --- b) compute distance to pup
            print("----> Computing distance to pup")
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
            
            ## --- c) compute head angle to pup
            print("----> Computing head angle to pup")
            trial_DLC = self.compute_head_angle_to_pup(trial_DLC, add_vector_columns = False,
                                                head_angle_to_pup_col = self.DLC_behaviour_cols["head_angle_to_pup"])
            

            # denoising pup coordinates

            # computing higher level bhv
            
            # d) add trial number to the dataframe
            trial_DLC[self.DLC_summary_cols["trial_num"]] == trial_num

            return trial_DLC

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

        df_DLC = self.check_and_insert_processed_columns(df_DLC)

        trials_dict = {}
        trial_nums = df_summary[self.DLC_summary_cols["trial_num"]]

        # iterate over each trial
        for trial_num in  trial_nums:

                trial_DLC, mask_DLC = self.extract_trial_from_DLC(df_DLC, df_summary, trial_num)
                trial_DLC = self.process_trial(trial_DLC, trial_num, interpolate_low_likelihoods = interpolate_low_likelihoods)
                
                trials_dict[trial_num] = trial_DLC # update the dictionary with the trial data
                df_DLC.loc[mask_DLC, :] = trial_DLC # update the dataframe

        return df_DLC, trials_dict
    
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
        trial_num_col = [self.DLC_summary_cols["trial_num"]]
        behavioural_cols = list(self.DLC_behaviour_cols.values())

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


