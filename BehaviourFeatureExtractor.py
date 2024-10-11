import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings('ignore') 


###### ------------------- Utility functions ------------------------- #####

def convert_seconds_to_frame(seconds, frame_rate = 30):
    return round(seconds*frame_rate)




###### ----------- Basic Feature extraction from DLC file ------------ #####
# Notes:
# - the following utility functions assume that the input is a processed DLC file.

class BehaviourFeatureExtractor:

    def __init__(self, path_to_config_file):

        # read .json config file for DLC
        self.config = pd.read_json(path_to_config_file)
        self.DLC_cols = self.config['DLC_columns']
        self.DLC_summary_cols = self.config['DLC_summary_columns']
        self.DLC_behaviour_cols = self.config['DLC_behaviour_columns']
        self.time_col = self.DLC_cols['time']
        self.frame_index_col = self.DLC_cols['frame']
        self.frame_rate = self.config['frame_rate']

    def extract_trial_from_DLC(self, df_DLC, df_summary, 
                                trial_num):
        # get the trial start and end times
        trial_start_time = df_summary.loc[df_summary[self.DLC_summary_cols["trial_num"]] == trial_num, self.DLC_summary_cols["pup_displacement"]].values[0]
        trial_end_time = df_summary.loc[df_summary[self.DLC_summary_cols["trial_num"]] == trial_num, self.DLC_summary_cols["trial_end"]].values[0]

        # convert trial start and end times to frame indices
        start_frame, end_frame = convert_seconds_to_frame(trial_start_time, self.frame_rate), convert_seconds_to_frame(trial_end_time, self.frame_rate)
        
        # extract the trial data
        mask = (df_DLC[self.frame_index_col] >= start_frame) & (df_DLC[self.frame_index_col] <= end_frame)

        return df_DLC.loc[mask, :]


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
        earRight_x, earRight_y = df_DLC[self.DLC_cols["earRight"]["x"]], df_DLC[self.DLC_cols["earRight"]["x"]]
        earLeft_x, earLeft_y = df_DLC[self.DLC_cols["earLeft"]["x"]], df_DLC[self.DLC_cols["earLeft"]["y"]]
        nose_x, nose_y = df_DLC[self.DLC_cols["nose"]["x"]], df_DLC[self.DLC_cols["nose"]["y"]]
        pup_x, pup_y  = df_DLC[self.DLC_cols["pup"]["x"]], df_DLC[self.DLC_cols["pup"]["y"]]

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
        df_DLC[head_angle_to_pup_col] = angle * 180/np.pi
        
        return df_DLC

    def extract_base_parameters(self, df_DLC, df_summary,
                                frame_index_col = 'frame_index'
                                ):
        """
        Extracts base parameters such as speed, distance to pup, and head angle to pup for each trial 
        from the given DataFrame. Updates a dictionary mapping trial number to the extracted trial data.
        Parameters:
        df_DLC (pd.DataFrame): DataFrame containing DeepLabCut (DLC) tracking data with a 'frame_index' column.
        df_summary (pd.DataFrame): DataFrame containing summary information for each trial, including 
                                'BehavRecdTrialEndSecs' and 'PupDispDropSecs' columns.
        Returns:
        pd.DataFrame: Updated DataFrame with additional columns for speed ('speed_cm/s'), 
                    distance to pup ('distance_to_pup'), and head angle to pup ('head_angle_to_pup_degrees').
                    These columns have default NaN values for frames that don't belong to any trial.
        trials_dict (dict): A dictionary containing the extracted trial data for each trial.
        """ 


        pup_speed_col = self.DLC_behaviour_cols["pup_speed"]
        mouse_speed_col = self.DLC_behaviour_cols["mouse_speed"]
        distance_col_mouse_pup = self.DLC_behaviour_cols["distance_mouse_pup"]
        distance_col_head_pup = self.DLC_behaviour_cols["distance_head_pup"]
        head_angle_to_pup_col = self.DLC_behaviour_cols["head_angle_to_pup"]

        # for each trial, get the start and end frames
        trial_end_col = self.DLC_summary_cols["trial_end"]
        trial_start_col = self.DLC_summary_cols["pup_displacement"]
        trial_num_col = self.DLC_summary_cols["trial_num"]

        end_times, start_times, trial_nums = df_summary[trial_end_col], df_summary[trial_start_col], df_summary[trial_num_col]

        # create NaN columns for speed, distance to pup and head angle to pup
        df_DLC[mouse_speed_col] = np.nan
        df_DLC[pup_speed_col] = np.nan
        df_DLC[distance_col_head_pup] = np.nan
        df_DLC[distance_col_mouse_pup] = np.nan
        df_DLC[head_angle_to_pup_col] = np.nan

        trials_dict = {}

        # iterate over each trial
        for end, start, trial_num in zip(end_times, start_times, trial_nums):
                # compute frame indices
                end_frame, start_frame = convert_seconds_to_frame(end), convert_seconds_to_frame(start)

                print(f"Processing trial {trial_num} Start frame: {start_frame} End frame: {end_frame}")

                # get the data for the trial
                mask = (df_DLC[frame_index_col] >= start_frame) & (df_DLC[frame_index_col] <= end_frame)
                trial_DLC = df_DLC.loc[mask, :] 

                # compute speed
                trial_DLC = self.compute_speed(trial_DLC,
                                               x_col = self.DLC_cols["msTop"]["x"],
                                               y_col = self.DLC_cols["msTop"]["y"],
                                               speed_col = mouse_speed_col)
                trial_DLC = self.compute_speed(trial_DLC, x_col = self.DLC_cols["pup"]["x"],
                                               y_col = self.DLC_cols["pup"]["y"],
                                               speed_col = pup_speed_col)

                # compute distance to pup
                trial_DLC = self.compute_distance_to_pup(trial_DLC,
                                                        x_col = self.DLC_cols["msTop"]["x"],
                                                        y_col = self.DLC_cols["msTop"]["y"],
                                                        pup_x_col = self.DLC_cols["pup"]["x"],
                                                        pup_y_col = self.DLC_cols["pup"]["y"],
                                                        distance_col = distance_col_mouse_pup)
                
                trial_DLC = self.compute_distance_to_pup(trial_DLC,
                                                        x_col = self.DLC_cols["endHeadbar"]["x"],
                                                        y_col = self.DLC_cols["endHeadbar"]["y"],
                                                        pup_x_col = self.DLC_cols["pup"]["x"],
                                                        pup_y_col = self.DLC_cols["pup"]["y"],
                                                        distance_col = distance_col_head_pup)
                
                # compute head angle to pup
                trial_DLC = self.compute_head_angle_to_pup(trial_DLC, add_vector_columns = False,
                                                    head_angle_to_pup_col = head_angle_to_pup_col)
                
                # update the dictionary with the trial data
                trials_dict[trial_num] = trial_DLC
                
                # update the dataframe
                df_DLC.loc[mask, :] = trial_DLC


        return df_DLC, trials_dict



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

    trial_1_DLC_frame.plot(x=self.DLC_cols["earRight"]["x"], y=self.DLC_cols["earLeft"], style='o', ax=ax, xlim=(0, xlim), ylim=(0, ylim), color = 'black')
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


