import pandas as pd
import os
import json

class DataLoader:

    def __init__(self, data_dir, path_to_config_file,
                 file_extensions = ['.csv', '.xlsx']):
        
        with open(path_to_config_file) as f:
            self.config = json.load(f)

        # DLC processing parameters
        self.frame_index_col = self.config["DLC_columns"]["frame"]
        self.time_seconds_col = self.config["DLC_columns"]["time"]
        self.frame_rate = self.config["frame_rate_dlc"]
        self.frame_index_to_drop = self.config["frame_index_to_drop"]   
        self.filters_dlc = self.config["dlc_file_tags"] # filters for DLC files
        
        # Data directory and file extensions
        self.data_dir = data_dir
        self.file_extensions = file_extensions

        # Dataframes and videos
        df_dict, video_dict = self.find_paths_main(data_dir)
        self.df_dict = df_dict
        self.video_dict = video_dict

    def find_paths_main(self, data_dir):
        """
        Main function to find paths in the given directory.
        This function initializes an empty dictionary and calls the helper function
        `find_paths_helper` to populate it with paths found in the specified directory.
        Args:
            data_dir (str): The directory path where the search for paths will be conducted.
        Returns:
            dict: A dictionary containing the paths found in the specified directory.
            Keys are the file paths and values are dictionaries containing the filenames and DataFrames.
        """

        d, d_video = {}, {}
        self.find_paths_helper(data_dir, d, d_video)
        return d, d_video

    def find_paths_helper(self, path_dir,  d = {},
                                           d_video = {}):
        """
        Recursively finds files with specific extensions in a directory and its subdirectories,
        converts them to DataFrames, and stores them in a dictionary.
        Args:
            path_dir (str): The directory path to search for files.
            d (dict, optional): A dictionary to store the file paths, filenames, and DataFrames. Defaults to an empty dictionary.
        Returns:
            None: The function updates the provided dictionary in place with the file paths as keys and dictionaries containing
                    filenames and DataFrames as values.
        Notes:
            - The function prints the items found in each directory for debugging purposes.
            - The function checks if an item is a directory and recursively calls itself if true.
            - If an item is a file with a specified extension, it converts the file to a DataFrame and stores it in the dictionary.
        """


        items = os.listdir(path_dir)
        #print(f"Found items {items} at directory {path_dir}")
        for item in items:
            # directory
            path_item = os.path.join(path_dir, item)
            
            if os.path.isdir(path_item):
                #print(f"Found a directory at path {path_dir}")
                self.find_paths_helper(path_item, d, d_video)

            # videos
            elif item.endswith('.mp4'):
                path_item = os.path.join(path_dir, item)
                mouse_id, day = item.split("_")[0], item.split("_")[1].split("DLC")[0]
                if mouse_id not in d_video:
                    d_video[mouse_id] = {day: path_item}
                else:
                    d_video[mouse_id][day] = path_item

            # convertible format
            elif item.endswith(tuple(self.file_extensions)):
                #print(f"Found a file at path {path_dir}")
                path_item = os.path.join(path_dir, item)
                df = self.convert_to_df(path_item)

                d[path_item] = {"filename": item, "df": df}

    def process_behav_dlc(self, path,
                        num_header_rows = 3,
                        change_column_names = True,
                        drop_wall_columns = True,
                        drop_inital_frames = True,
                        frame_index_to_drop = 150,
                        create_time_seconds = True,
                        frame_rate = 30,
                        frame_index_col = 'frame_index',
                        time_seconds_col = 'time_seconds'):
        
        """
        Processes a DataFrame containing behavioral data from DeepLabCut (DLC).
        Parameters:
            path (str): The file path to the DLC data file.
            num_header_rows (int, optional): The number of header rows in the file. Defaults to 3.
            change_column_names (bool, optional): Whether to change the column names. Defaults to True.
            drop_wall_columns (bool, optional): Whether to drop columns containing 'wall'. Defaults to True.
            drop_inital_frames (bool, optional): Whether to drop rows until frame_index 150. Defaults to True.
            frame_index_to_drop (int, optional): The frame index until which rows should be dropped. Defaults to 150.
            create_time_seconds (bool, optional): Whether to create a time_seconds column from the frame_index column. Defaults to True.
            framerate (int, optional): The framerate of the video. Defaults to 30.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        # read first 3 rows of the file containing feature information
        headers = pd.read_csv(path, nrows=num_header_rows)
        # extract bodyparts and coordinates names
        bodyparts, coords = headers.iloc[0, 1:], headers.iloc[1, 1:]
        # read the file data
        df = pd.read_csv(path, skiprows=num_header_rows-1)
                        
        # ----- 1. Change column names
        if change_column_names:
            # concatenate the first two rows and join by by '_' to generate new column names
            new_columns = [frame_index_col] + [el1 + '_' + el2 for el1, el2 in zip(bodyparts, coords)]
            # rename the columns
            df.columns = new_columns

        # ----- 2. Drop rows until frame_index 150
        if drop_inital_frames:
            df = df[df[frame_index_col] > frame_index_to_drop]

        # ----- 3. Create time_seconds column
        if create_time_seconds:
            df[time_seconds_col] = df[frame_index_col] / frame_rate
            # redefine order of columns in dataframe
            df = df[[frame_index_col, time_seconds_col] + [col for col in df.columns if col not in [frame_index_col, time_seconds_col]]]

        # ----- 4. Drop columns 
        if drop_wall_columns:
            columns_to_drop = [col for col in df.columns if 'wall' in col.lower()]
            df = df.drop(columns=columns_to_drop)

        return df
        
    def convert_to_df(self, path):
        """
        Converts a file at the given path to a pandas DataFrame.
        Parameters:
            - path (str): The file path to the data file. The file can be a CSV or an Excel file.
        Returns:
        pandas.DataFrame: The data from the file as a pandas DataFrame.
        Notes:
            - If the file is a CSV and contains 'resnet50' or 'dlc' in its name (case insensitive), 
        the DataFrame will be processed by the `process_behav_dlc` function.
            - The function currently supports only CSV and Excel files.
        """

        if path.endswith('.csv'):
            if any([f in path.lower() for f in self.filters_dlc]):
                df = self.process_behav_dlc(path, 
                                            frame_index_to_drop=self.frame_index_to_drop,
                                            frame_index_col = self.frame_index_col,
                                            time_seconds_col = self.time_seconds_col,
                                            frame_rate = self.frame_rate)
            else:
                df = pd.read_csv(path)

        elif path.endswith('.xlsx'):
            df = pd.read_excel(path)

        return df
        
    def filter_files(self, filters):
        """
        Filters files based on provided criteria.
        Args:
            filters (list): A list of strings to filter the files by.
        Returns:
            list: List of file paths that match all the provided filters.
            None: If results_dict is not provided.
        """
                
        if self.df_dict is  None:
            self.df_dict = self.find_paths_main(self.data_dir)

        files = [k for k, v in self.df_dict.items() if all(f in k for f in filters)]
        return files

    def get_data_for_experiment(self, mouse_id: str,
                    day = None,
                    extra_param_1 = None,
                    extra_param_2 = None):
        
        """
        Retrieves DLC, Avisoft, BehavSummary data for an experiment session based on specified filters.
        Parameters:
        -----------
        - mouse_id (str): The mouse ID to filter by.
        - day (str): The day to filter by.
        - extra_param_1 (str): Optional. Extra parameter to filter by.
        - extra_param_2 (str): Optional. Extra parameter to filter by.

        Returns:
        --------
        dict
            A dictionary containing the experiment data with the following structure:
            {
                'Behavior': {
                    'df_dlc': DataFrame or None,
                    'df_summary': DataFrame or None
                },
                'Avisoft': {
                    'df': DataFrame or None
                }
            }
            The DataFrame values are populated based on the filtered files.

        Examples:
        ---------
        # Retrieve data for a specific mouse ID and day
        data = get_data_for_experiment(mouse_id='M1', day='d1')    
        """

        filters = [f for f in [mouse_id, day, extra_param_1, extra_param_2] if f is not None] 
        files = self.filter_files(filters)

        experiment_data = {'Behavior': {"df_dlc": None ,
                            "df_summary": None},
                            "Avisoft": {"df": None}}

        for f in files:
            if all(filter in f.lower() for filter in self.filters_dlc):
                experiment_data['Behavior']['df_dlc'] = self.df_dict[f]['df'].copy()
            elif 'summary' in f.lower():
                experiment_data['Behavior']['df_summary'] = self.df_dict[f]['df'].copy()
            elif 'avisoft' in f.lower():
                experiment_data['Avisoft']['df'] = self.df_dict[f]['df'].copy()
        
        return experiment_data

    def get_processed_data_for_experiment(self, processed_data_dir, mouse_id, day, trial_num):
        
        experiment_data_processed = {'Behavior': {
                                        "df_summary": None,
                                        "df_dlc": None} , 
                                    "trials": {}
                                    }
        
        # load data
        experiment_data_processed['Behavior']['df_summary'] = pd.read_csv(f"{processed_data_dir}/{mouse_id}/{day}/BehavSummary_{mouse_id}_{day}.csv")
        experiment_data_processed[mouse_id][day]["Behavior"]["df_dlc"] = pd.read_csv(f"{processed_data_dir}/{mouse_id}/{day}/DLC_original_{mouse_id}_{day}.csv")
        
        for trial_num in experiment_data_processed[mouse_id][day]["trials"]:
            dlc_data = pd.read_csv(f"{processed_data_dir}/{mouse_id}/{day}/trials/trial{trial_num}_DLC_processed_{mouse_id}_{day}.csv")
            experiment_data_processed[mouse_id][day]["trials"][trial_num] = {"dlc_data": dlc_data}

    def collect_and_process_experiment_data(self, mouse_ids, days, BF_instance, VF_instance, processed_data_dir = None, export = False):

        experiment_data = {}

        for mouse_id in mouse_ids:
            experiment_data[mouse_id] = {}
            for day in days:
                # Load data 
                data = self.get_data_for_experiment(mouse_id = mouse_id,
                                                day = day)
                if data is None:
                    print("Data not found for mouse", mouse_id, "on day", day)
                    continue

                experiment_data[mouse_id][day] = data

                df_DLC = experiment_data[mouse_id][day]["Behavior"]["df_dlc"].copy()
                df_Avi = experiment_data[mouse_id][day]["Avisoft"]["df"].copy()
                df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"].copy()

                # align USV data to DLC data
                df_DLC_processed, _ = BF_instance.process_DLC(df_DLC.copy(), df_summary)
                trials, df_DLC, df_USV = VF_instance.process_USV(df_Avi, df_summary, df_DLC_processed)
                experiment_data[mouse_id][day]["trials"] =  trials

                if export and processed_data_dir is not None:
                     # export the summary data
                    df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"]
                    df_summary.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/BehavSummary_{mouse_id}_{day}.csv", index = False)

                    # export the original DLC data
                    df_DLC = experiment_data[mouse_id][day]["Behavior"]["df_dlc"]
                    df_DLC.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/DLC_original_{mouse_id}_{day}.csv", index = False)

                    for trial_num in experiment_data[mouse_id][day]["trials"]:
                        dlc_data = experiment_data[mouse_id][day]["trials"][trial_num]["dlc_data"]
                        # export the processed DLC data
                        dlc_data.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/trials/trial{trial_num}_DLC_processed_{mouse_id}_{day}.csv", index = False)

        return experiment_data


def export_processed_data(processed_data_dir, experiment_data):
    os.makedirs(processed_data_dir, exist_ok = True)

    for mouse_id in experiment_data:
        os.makedirs(f"{processed_data_dir}/{mouse_id}", exist_ok = True)

        for day in experiment_data[mouse_id]:   
            os.makedirs(f"{processed_data_dir}/{mouse_id}/{day}", exist_ok = True)
            os.makedirs(f"{processed_data_dir}/{mouse_id}/{day}/trials/", exist_ok = True)

            # export the summary data
            df_summary = experiment_data[mouse_id][day]["Behavior"]["df_summary"]
            df_summary.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/BehavSummary_{mouse_id}_{day}.csv", index = False)

            # export the original DLC data
            df_DLC = experiment_data[mouse_id][day]["Behavior"]["df_dlc"]
            df_DLC.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/DLC_original_{mouse_id}_{day}.csv", index = False)

            for trial_num in experiment_data[mouse_id][day]["trials"]:
                dlc_data = experiment_data[mouse_id][day]["trials"][trial_num]["dlc_data"]
                # export the processed DLC data
                dlc_data.to_csv(f"{processed_data_dir}/{mouse_id}/{day}/trials/trial{trial_num}_DLC_processed_{mouse_id}_{day}.csv", index = False)

def load_processed_data(processed_data_dir, mouse_ids, days):

    experiment_data_processed = {}  

    for mouse_id in mouse_ids:
        experiment_data_processed[mouse_id] = {}
        for day in days:
            experiment_data_processed[mouse_id][day] = {}
            experiment_data_processed[mouse_id][day]["Behavior"] = {}
            experiment_data_processed[mouse_id][day]["trials"] = {}
        
            experiment_data_processed[mouse_id][day]["Behavior"]["df_summary"] = pd.read_csv(f"{processed_data_dir}/{mouse_id}/{day}/BehavSummary_{mouse_id}_{day}.csv")
            experiment_data_processed[mouse_id][day]["Behavior"]["df_dlc"] = pd.read_csv(f"{processed_data_dir}/{mouse_id}/{day}/DLC_original_{mouse_id}_{day}.csv")
            
            files = os.listdir(f"{processed_data_dir}/{mouse_id}/{day}/trials/")

            for file in files:
                if "trial" in file:
                    trial_num = file.split("_")[0].split("trial")[1]
                    trial_num = int(trial_num)
                    dlc_data = pd.read_csv(f"{processed_data_dir}/{mouse_id}/{day}/trials/{file}")
                    experiment_data_processed[mouse_id][day]["trials"][trial_num] = {"dlc_data": dlc_data}

    return experiment_data_processed    