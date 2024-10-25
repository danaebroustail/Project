import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import warnings
import datetime
from BehaviourFeatureExtractor import BehaviourFeatureExtractor

# Suppress FutureWarning messages
warnings.filterwarnings('ignore') 

class VocalFeatureExtractor:

    def __init__(self, path_to_config_file):

        # read .json config file for 
        with open(path_to_config_file) as f:
            self.config = json.load(f)

        self.DLC_cols = self.config['DLC_columns']
        self.DLC_summary_cols = self.config['DLC_summary_columns']
        self.DLC_behaviour_cols = self.config['DLC_behaviour_columns']

        self.USV_input_cols = self.config['USV_processing']['input_columns']
        self.USV_output_cols = self.config['USV_processing']['output_columns']
        self.begin_time_USV = self.USV_input_cols['begin_time_col']
        self.end_time_USV = self.USV_input_cols['begin_time_col']
        self.frames_per_bout = self["number_of_frames_per_bout"]

        self.time_col = self.DLC_cols['time']
        self.frame_index_col = self.DLC_cols['frame']
        self.frame_rate = self.config['frame_rate']

        self.BF = BehaviourFeatureExtractor(path_to_config_file)

    def extract_trial_USV(self, df_USV, df_summary, trial_num):
        
        # keep only accepted calls
        df_USV = df_USV[df_USV[self.USV_input_cols["accepted"]] == True]

        trial_begin_time = df_summary[df_summary[self.DLC_summary_cols["trial_num"]] == trial_num][self.DLC_summary_cols["pup_displacement"]].values[0]    
        trial_end_time = df_summary[df_summary[self.DLC_summary_cols["trial_num"]] == trial_num][self.DLC_summary_cols["pup_displacement"]].values[0]
        
        # get the calls in the trial
        mask_trial_window = (df_USV[self.begin_time_USV] >= trial_begin_time) & (df_USV[self.end_time_usv] <= trial_end_time)
        trial_USV = df_USV[mask_trial_window]   
        
        return trial_USV
    
    def assign_bout_index_to_DLC(self, trial_DLC):  

        USV_window_col = self.USV_output_cols["bout_window_index"]
        for i in range(0, len(trial_DLC), self.frames_per_bout):
            trial_DLC[USV_window_col][i:i+self.frames_per_bout] = i//self.frames_per_bout
        
        return trial_DLC

    def assign_bout_index_to_USV(self, trial_USV, trial_DLC):

        def most_frequent(l:np):
            l = l.tolist()
            return max(set(l), key=l.count)

        for i in range(len(trial_USV)):
    
            row = trial_USV.iloc[i]
            begin_time_usv = row['BeginTime_s_']
            end_time_usv = row['EndTime_s_']
            #duration = row['CallLength_s_']

            begin_time_usv_frame = self.BF.convert_seconds_to_frame(begin_time_usv, self.frame_rate)
            end_time_usv_frame = self.BF.convert_seconds_to_frame(end_time_usv, self.frame_rate)

            # print("Begin time: {}, End time: {}, Duration: {}".format(begin_time_usv, end_time_usv, duration))
            # print("Begin time frame: {}, End time frame: {}".format(begin_time_usv_frame, end_time_usv_frame))

            bout_window = trial_DLC[(trial_DLC['time_seconds'] >= begin_time_usv) & (trial_DLC['time_seconds'] <= end_time_usv)]['bout_window'].values
            bout_window = trial_DLC[(trial_DLC['frame_index'] >= begin_time_usv_frame) & (trial_DLC['frame_index'] <= end_time_usv_frame)]['bout_window'].values

            # print("bout window = ", bout_window)
            if len(bout_window) > 0:
                # pick most frequent bout window
                most_frequent_index = most_frequent(bout_window)
                # print("most_frequent_index = ", most_frequent_index)
                # print("index = ", i)
                trial_USV['bout_window'].iloc[i] = most_frequent_index

        return trial_USV
    
    def merge_USV_DLC(self, trial_DLC, trial_USV):

        # group usv trial by bout window
        df_Avi_USV_trial_grouped = trial_USV.groupby("bout_window").agg({self.USV_input_cols["call_duration"]: "mean",
                                                                         self.USV_input_cols["amplitude_col"]: "mean",
                                                                         self.USV_output_cols["bout_window_index"]: "count"})
        # renames the columns
        df_Avi_USV_trial_grouped.columns = [ self.USV_output_cols["average_duration"],
                                            self.USV_output_cols["average_frequency"],
                                            self.USV_output_cols["call_number"]]
        
        for bout_window in trial_DLC["bout_window"].unique():
            bout = df_Avi_USV_trial_grouped[df_Avi_USV_trial_grouped.index == bout_window]
            if len(bout) > 0:
                for output_col in [ self.USV_output_cols["average_duration"],
                                    self.USV_output_cols["average_frequency"],
                                    self.USV_output_cols["call_number"]]:
                    trial_DLC.loc[trial_DLC['bout_window'] == bout_window, output_col] = bout[output_col].values[0]

        return trial_DLC

    def process_trial_USV(self, trial_USV, trial_DLC):

        df_DLC = self.check_and_insert_columns_USV(df_DLC)

        # 1. assign bout index to trial_DLC
        trial_DLC = self.assign_bout_index_to_DLC(trial_DLC)

        # 2. assign bout index to trial_USV 
        trial_USV = self.assign_bout_index_to_USV(trial_USV, trial_DLC)

        # 3. combine trial_USV and trial_DLC
        trial_DLC = self.merge_USV_DLC(trial_DLC, trial_USV)


        return trial_DLC, trial_USV

    def process_USV(self, df_USV, df_summary, df_DLC):

        # insert and check if required output columns are present
        df_DLC = self.check_and_insert_columns_USV(df_DLC)

        # iterate over all trials


        pass

    def check_and_insert_columns_USV(self, df_DLC, df_USV):

        # check if required columns are present
        for col in self.USV_output_cols.values():
            if col not in df_DLC.columns:
                df_DLC[col] = np.nan

        for col in [self.USV_output_cols["bout_window_index"]]:
            if col not in df_USV.columns:
                df_USV[col] = np.nan

        return df_DLC



#### -----------  Plotting utilities  ----------- ####
