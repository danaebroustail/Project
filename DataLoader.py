import pandas as pd
import os

class DataLoader:

    def __init__(self, data_dir, file_extensions = ['.csv', '.xlsx']):
        self.data_dir = data_dir
        self.file_extensions = file_extensions
        self.df_dict = self.find_paths_main(data_dir)

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

        d = {}
        self.find_paths_helper(data_dir, d)
        return d

    # write a function that goes in the data_dir and finds all of the paths to the files
    def find_paths_helper(self, path_dir,  d = {}):
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
        print(f"Found items {items} at directory {path_dir}")
        for item in items:
            # directory
            path_item = os.path.join(path_dir, item)
            
            if os.path.isdir(path_item):
                #print(f"Found a directory at path {path_dir}")
                self.find_paths_helper(path_item, d)

            # convertible format
            elif item.endswith(tuple(self.file_extensions)):
                #print(f"Found a file at path {path_dir}")
                path_item = os.path.join(path_dir, item)
                df = self.convert_to_df(path_item)

                d[path_item] = {"filename": item, "df": df}

    def process_behav_dlc(self, path,
                        num_header_rows = 3,
                        change_column_names=True,
                        drop_wall_columns = True,
                        drop_inital_frames = True,
                        frame_index_to_drop = 150,
                        create_time_seconds = True,
                        framerate = 30):
        
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
            new_columns = ['frame_index'] + [el1 + '_' + el2 for el1, el2 in zip(bodyparts, coords)]
            # rename the columns
            df.columns = new_columns

        # ----- 2. Drop rows until frame_index 150
        if drop_inital_frames:
            df = df[df['frame_index'] > frame_index_to_drop]

        # ----- 3. Create time_seconds column
        if create_time_seconds:
            df['time_seconds'] = df['frame_index'] / framerate
            # redefine order of columns in dataframe
            df = df[['frame_index', 'time_seconds'] + [col for col in df.columns if col not in ['frame_index', 'time_seconds']]]

        # ----- 4. Drop columns 
        if drop_wall_columns:
            columns_to_drop = [col for col in df.columns if 'wall' in col.lower()]
            df = df.drop(columns=columns_to_drop)

        return df
        

    def convert_to_df(self, path):
        """
        Converts a file at the given path to a pandas DataFrame.
        Parameters:
        path (str): The file path to the data file. The file can be a CSV or an Excel file.
        Returns:
        pandas.DataFrame: The data from the file as a pandas DataFrame.
        Notes:
        - If the file is a CSV and contains 'resnet50' or 'dlc' in its name (case insensitive), 
        the DataFrame will be processed by the `process_behav_dlc` function.
        - The function currently supports only CSV and Excel files.
        """

        if path.endswith('.csv'):
            if 'resnet50' in path.lower() or 'dlc' in path.lower():
                df = self.process_behav_dlc(path)
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
                    day:str,
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
            if 'resnet50' in f.lower():
                experiment_data['Behavior']['df_dlc'] = self.df_dict[f]['df']
            elif 'summary' in f.lower():
                experiment_data['Behavior']['df_summary'] = self.df_dict[f]['df']
            elif 'avisoft' in f.lower():
                experiment_data['Avisoft']['df'] = self.df_dict[f]['df']
        
        return experiment_data

    
    