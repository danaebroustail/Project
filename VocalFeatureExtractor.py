import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import warnings
import datetime

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
        self.time_col = self.DLC_cols['time']
        self.frame_index_col = self.DLC_cols['frame']
        self.frame_rate = self.config['frame_rate']
        self.minimum_distance_to_nest = self.config['minimum_distance_to_nest']
        self.likelihood_threshold = self.config['likelihood_threshold']