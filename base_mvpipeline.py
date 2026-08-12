import pandas as pd
import os
import re
import json
import yaml
import glob

from typing import Literal
from pathlib import Path
from typing import Callable
from dateutil import tz
from datetime import datetime
from datetime import timedelta
from utils.algorithms import do_algorithm, ganz_algorithm, mixed_threshold_algorithm, changepoints_algorithm
import logging
import numpy as np
import concurrent.futures as cf

valid_df_name = Literal['ground_truth', 'do', 'ganz', 'mixed_threshold', 'changepoints']
valid_df_name_set = {'ground_truth', 'do', 'ganz', 'mixed_threshold', 'changepoints'}
eastern = tz.gettz('US/Eastern')
utc = tz.gettz('UTC')

do_threshold_default = 0.001
ganz_threshold_default = 0.05
mixed_threshold_relative_threshold_default = 0.0005 # 0.1
mixed_threshold_absolute_difference_default = 0

class BaseMVPipeline:
    def __init__(self, config, base_file_path, start_time = "2023-11-01 00:00:00", end_time = "2024-11-30 23:59:59", output_path = None):
        self.config = config
        self.base_file_path = base_file_path
        self.type_to_prefix = {'o': 'per_outage', 'c': 'per_county', 'z': 'per_zipcode'} 
        self.geomap = {}
        self._load_geo_mapping()

        self._dataset_lower_bound = None
        self._dataset_upper_bound = None
        self._set_timeframe(start_time = start_time, end_time=end_time)

        self._per_outage = pd.DataFrame({})
        self._per_county = pd.DataFrame({})

        self.methods_to_run = []

        self._ground_truth_df = pd.DataFrame({})
        self._do_df = pd.DataFrame({})
        self._do_threshold = 0.001
        self._ganz_df = pd.DataFrame({})
        self._ganz_threshold = 0.05
        self._mixed_threshold_df = pd.DataFrame({})
        self._mixed_threshold_values = {
            'relative_threshold': 0.0005, #0.1,
            'absolute_difference': 0 
        }

        self._changepoints_df = pd.DataFrame({})

        self._name_to_df = { # holds naming scheme for the file names as well
            "ground_truth": self._ground_truth_df,
            "do": self._do_df,
            "ganz": self._ganz_df,
            "mixed_threshold": self._mixed_threshold_df,
            "changepoints": self._changepoints_df
        }
        self._comparison_table = pd.DataFrame()

        self._output_path = None
        self._set_output_path(output_path = output_path)
        
        self._output_path.mkdir(parents=True, exist_ok=True)
        pass

    def _load_geo_mapping(self):
        try:
            with open('/root/autodl-tmp/mvpipeline/utils/zip_to_county_name.json', 'r') as json_file:
                self.geomap['zip_to_county_name'] = json.load(json_file)
            with open('/root/autodl-tmp/mvpipeline/utils/zip_to_county_fips.json', 'r') as json_file:
                self.geomap['zip_to_county_fips'] = json.load(json_file)
            with open('/root/autodl-tmp/mvpipeline/utils/zip_to_state_name.json', 'r') as json_file:
                self.geomap['zip_to_state_name'] = json.load(json_file)
        except Exception as e:
            print(f"An error occurred during geo map loading: {e}") 
            raise
    
    def _construct_raw_file_path(self):
        '''
        Constructing the file path for both per_outage and per_county'''
        # file_prefix = self.type_to_prefix[self.config['type']]

        per_county_file_path = f"{self.base_file_path}/{self.config['state']}/layout_{self.config['layout']}/per_county_{self.config['name']}.csv"
        per_outage_file_path = f"{self.base_file_path}/{self.config['state']}/layout_{self.config['layout']}/per_outage_{self.config['name']}.csv"

        return per_county_file_path.replace('//', '/'), per_outage_file_path.replace('//', '/')

    def _set_output_path(self, output_path = None):
        if not output_path:
            self._output_path = Path(__file__).parent / 'output' / 'transformed_data' / self.config['state'] / f'layout_{self.config["layout"]}' / self.config['name']
        else:
            self._output_path = output_path

    def _construct_exported_df_path(self, df_name: valid_df_name):
            '''
            Taking in a specific dataframe name (ground_truth, do, ganz, mixed_threshold, changepoints, etc.) (HAS TO MATCH ONE OF valid_df_name),
            generates a path for the specified dataframe (to export to or to retrieve from)
            '''
            if df_name not in valid_df_name_set:
                raise Exception(
                    f"df_name {df_name} is not one of the names of the dataframes. "
                    f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
                )

            lower_date = self._dataset_lower_bound.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'
            upper_date = self._dataset_upper_bound.strftime('%Y-%m-%d') 
            return self._output_path / f'{self.config["name"]}_{df_name}__{lower_date}_to_{upper_date}.csv'

    def _construct_comparison_table_path(self):
            '''
            Taking in a specific dataframe name (ground_truth, do, ganz, mixed_threshold, changepoints, etc.) (HAS TO MATCH ONE OF valid_df_name),
            generates a path for the specified dataframe (to export to or to retrieve from)
            '''
            lower_date = self._dataset_lower_bound.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'
            upper_date = self._dataset_upper_bound.strftime('%Y-%m-%d') 
            return self._output_path / f'{self.config["name"]}_comparison_table__{lower_date}_to_{upper_date}.csv'

    def _set_timeframe(self, start_time = "2023-11-01 00:00:00", end_time = "2024-11-30 23:59:59"):
        self._dataset_lower_bound = pd.Timestamp(start_time).tz_localize(eastern) 
        self._dataset_upper_bound = pd.Timestamp(end_time).tz_localize(eastern)

    def _construct_and_get_df_variable(self, df_name: valid_df_name):
        '''
        Pass in a specific name representing one of the algorithms and get returned the df value associated with that name
        'ground_truth' = self._ground_truth_df
        'do' = self._do_df
        'ganz'= self._ganz_df
        'mixed_threshold' = self._mixed_threshold_df
        'changepoints' = self._changepoints_df
        
        '''
        if df_name not in valid_df_name_set:
            raise Exception(
                f"df_name {df_name} is not one of the names of the dataframes. "
                f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
            )
        variable_name = None
        variable_name = f"_{df_name}_df"
        dataframe = getattr(self, variable_name, None)
        return dataframe

    def _load_raw_data(self):
        ''' Loading the raw per_county and per_outage datasets '''
        try:
            per_county_file_path, per_outage_file_path = self._construct_raw_file_path()
            self._per_county = pd.read_csv(per_county_file_path)
            self._per_outage = pd.read_csv(per_outage_file_path)
        except Exception as e:
            logging.warning(f"An error occurred during file loading: {e}. There may not be an existing per_outage or per_county file.")
            raise

    def _load_aggregated_df(self, df_name: valid_df_name):
        try:
            if df_name not in valid_df_name_set:
                raise Exception(
                    f"df_name {df_name} is not one of the names of the dataframes. "
                    f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
                )

            df_path = self._construct_exported_df_path(df_name)

            exists = False
            if df_path.exists():
                print(f"The computed dataframe for {df_name} exists. Loading into self._{df_name}_df")
                variable_name = f"_{df_name}_df"
                if hasattr(self, variable_name):
                    read_df = pd.read_csv(df_path)
                    setattr(self, variable_name, read_df)
                    self._update_name_to_df_map(df_name= df_name, df=read_df)
            else:
                # print(f"The path {df_path} does not exist. No dataframe file has been exported before or a file with a different time span exists. Check to make sure the right lower-bound timeframe and upper-bound timeframe is set.")
                # print("Generating groundtruth")
                # self.create_groundtruth(force_agg=True)
                raise Exception(f'There is no file: {df_path}. Either this is on purpose or you forgot to generate and/or export the df associated with {df_name}. If on purpose, then please use the respective method to get the aggregated dataframe instead of _load_aggregated_df')
                
        except Exception as e:
            logging.warning(f"An error occurred during file loading: {e}. Something may have went wrong with loading the already exported dataframes. ")

    def _transform_loaded_aggregated_df(self, df_name: valid_df_name):
        def transform_outage_point(outage_point_str): 
            if not isinstance(outage_point_str, str): 
                return outage_point_str 
            # Regular expression to match tuples 
            tuple_pattern = re.compile(r'\((-?\d+\.?\d*|None|<NA>),\s*(-?\d+\.?\d*|None|<NA>)\)') 
            tuples = [] 
            for match in tuple_pattern.findall(outage_point_str): 
                # Replace 'None' and '<NA>' with None, and convert numbers to float 
                tuple_values = [] 
                for val in match: 
                    if val in ['None', '<NA>']: 
                        tuple_values.append(None) 
                    else: 
                        tuple_values.append(float(val)) 
                tuples.append(tuple(tuple_values)) 
            
            return tuples
        
        
        if df_name not in valid_df_name_set:
            raise Exception(
                f"df_name {df_name} is not one of the names of the dataframes. "
                f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
            )

        dataframe = self._construct_and_get_df_variable(df_name)
        
        if dataframe is None:
            raise Exception(
                f"Instance Variable ._{df_name}_df holds no value. Make sure it has the corresponding aggregated df. Maybe run _load_aggregated_df()?"
            )
        if dataframe.empty:
            raise Exception(
                f"Instance Variable ._{df_name}_df is an empty dataframe. Something may have went wrong with aggregating the raw file or when loading a preexisting aggregated df (like the raw data or imported file were originally empty)"
            )

        dataframe['start_time'] = pd.to_datetime(dataframe['start_time'], utc=True).dt.tz_convert(eastern)
        dataframe['end_time'] = pd.to_datetime(dataframe['end_time'], utc=True).dt.tz_convert(eastern)

        dataframe['weighted_customers_out'] = pd.to_numeric(dataframe['weighted_customers_out'])
        dataframe['average_customers_out'] = pd.to_numeric(dataframe['average_customers_out'])
        dataframe['duration'] = pd.to_numeric(dataframe['duration'])

        if df_name == 'ground_truth':
            dataframe['duration_timestamp_diff'] = pd.to_numeric(dataframe['duration_timestamp_diff'])
            dataframe['duration_mean'] = pd.to_numeric(dataframe['duration_mean'])

            # dataframe['outage_point'] = dataframe['outage_point'].apply(lambda x: ast.literal_eval(x))
            # dataframe['outage_point'] = dataframe['outage_point'].apply(safe_literal_eval)
            
            dataframe['outage_point'] = dataframe['outage_point'].apply(transform_outage_point)

        # if df_name == 'changepoints':
        #     dataframe['penalty'] = pd.to_numeric(dataframe['penalty'])
        #     if (isinstance(dataframe['duration'].iloc[-1], str) and 'days' in dataframe['duration'].iloc[-1]) or isinstance(dataframe['duration'].iloc[-1], pd.Timedelta): 
        #         # hack fix for some changepoint df having duration as timedelta instead of numeric minutes
        #         # changing it to minutes
        #         num_strings = dataframe['duration'].apply(lambda x: isinstance(x, str)).sum()
        #         print(f"{num_strings} out of {len(dataframe)} 'duration' rows are strings")
        #         dataframe['duration'] = pd.to_timedelta(dataframe['duration'])
        #         dataframe['duration'] = (dataframe['duration'].total_seconds()) / 60


        setattr(self, f"_{df_name}_df", dataframe)

        return

    def _load_comparison_table(self):
        try:
            comparison_table_path = self._construct_comparison_table_path()

            exists = False
            if comparison_table_path.exists():
                print(f"A previously exported comparison table exists. Loading into self._comparison_table")
                variable_name = f"_comparison_table"
                if hasattr(self, variable_name):
                    read_df = pd.read_csv(comparison_table_path)
                    setattr(self, variable_name, read_df)
            else:
                print(f"The path {comparison_table_path} does not exist. No dataframe file has been exported before or a file with a different time span exists. Check to make sure the right lower-bound timeframe and upper-bound timeframe is set.")
                print("Generating new comparison table")
                # breakpoint()
                self._comparison_table = self.generate_comparison_table()
                # raise Exception
                
                
        except Exception as e:
            logging.warning(f"An error occurred during file loading: {e}. Something went wrong with loading the already exported dataframes")
            raise        

    def _update_name_to_df_map(self, df_name: valid_df_name, df):
        # might not need this?
        if df_name not in valid_df_name_set:
            raise Exception(
                f"{df_name} is not one of the names of the dataframes. "
                f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
            )
        else:
            self._name_to_df[df_name] = df

    def _transform_per_county(self):
        '''
        Cleaning and transforming the per_county dataset and prepare it to be passed to the methods:
        - Turning the dataset into a uniform attribute schema used across all layouts
        - Doing transformations
            - Datetime transformations

        Schema:
        - timestamp = column containing the time the webscraper scraped the data
        - county = column containing the county the outage occurs in
        - customersServed = utility company's customer population
        - customersOutNow = # of customers without power
        - utilityProvider = company providing power (EMC, etc)
        Includes datetime transformation
        '''
        if self._per_county.empty:
            print("Error: per_county is empty and is probably not loaded. Running ._load_raw_data") 
            self._load_raw_data()
        return

    def _transform_per_outage(self):
        '''
        Cleaning and transforming the per_outage dataset to be prepared to have _create_groundtruth() performed on it
        '''  
        if self._per_outage.empty:
            print("Error: per_outage is empty and is probably not loaded. Running ._load_raw_data") 
            self._load_raw_data()
        return

    def transform_datasets(self):
        '''
        Runs both transform functions 

        Filters out desired time frame for both per_outage and per_county

        '''
        if self._per_county.empty or self._per_outage.empty:
            print("Error: per_county and/or per_outage are not loaded. Running self._load_raw_data(). All transformations performed on both per_outage and per_county performed are reset, and both datasets are reset back to their raw forms.")
            self._load_raw_data()
        self._transform_per_county()
        self._transform_per_outage()

        self._per_county = self._per_county[
            (self._per_county['timestamp'] >= self._dataset_lower_bound) &
            (self._per_county['timestamp'] <= self._dataset_upper_bound) 
        ]

        self._per_outage = self._per_outage[
            (self._per_outage['timestamp'] >= self._dataset_lower_bound) &
            (self._per_outage['timestamp'] <= self._dataset_upper_bound) 
        ]

        if self._per_county.empty:
            print(f"In transform_datasets(): Filtered per_county dataset is empty. Either the original dataset is empty or there is no data within {self._dataset_lower_bound} to {self._dataset_upper_bound}")
        if self._per_outage.empty:
            print(f"In transform_datasets(): Filtered per_outage dataset is empty. Either the original dataset is empty or there is no data within {self._dataset_lower_bound} to {self._dataset_upper_bound}")
        
        return

    def create_groundtruth(self, force_agg=False):
        '''
        General:
        Aggregating per_outage into a dataset where each row is an spatio-temporal outage event
        Includes datetime transformation
        Input: per_outage dataset of provider
        Output: ground-truth dataset to compare against

        base_mvpipeline specific:
        Checks whether the ground truth exported file already exists and/or the aggregation should be done anyway
        Parent_function specific output: Whether the ground truth dataset already exists or not (whether through previous invoking of the function or through the file already existing locally)
        '''
        groundtruth_path = self._construct_exported_df_path('ground_truth')

        if self._per_outage.empty:
            raise Exception("per_outage is empty and is probably not loaded. Make sure ._load_raw_data is run and also run ._transform_per_outage after loading. Stopping execution.")
        
        exists = False
        if groundtruth_path.exists():
            print(f"The file {groundtruth_path} exists.")
            self._ground_truth_df = pd.read_csv(groundtruth_path)
            exists = True
        
        if force_agg:
            print("Forcing groundtruth aggregation")
            exists = False

        return exists

    def do_method(self, force_agg=False, threshold = None):
        '''
        Input:
        - force_agg = whether you want to force the algorithm to run again
        - threshold = the threshold for the do algorithm
        Output:
        - None (stores the df to self._do_df)
        '''
        if threshold is None:
            self._do_threshold = do_threshold_default
        elif isinstance(threshold, (int, float)) and threshold <= 1 and threshold >= 0:
            self._do_threshold = threshold
        else:
            raise Exception(f"Do threshold value {threshold} is either not a number or not a value between 0 and 1. Please edit and run do_method() again")
        
        do_path = self._construct_exported_df_path('do')
        if self._per_county.empty:
            raise Exception("_per_county is empty and is probably not loaded. Make sure ._load_raw_data is run and also run ._transform_per_county after loading. Stopping execution.")
        # TODO: modify the thresholds
        do_df = pd.DataFrame({}) # empty df to hold the do algorithm results
        for _do_thres in np.arange(0, 1.001, step=0.1):
            _do_thres = max(_do_thres, 0.001)
            if force_agg:
                print("Forcing do algorithm")
                do_df = pd.concat([do_df, do_algorithm(self, threshold=_do_thres)])
            elif do_path.exists():
                print(f"The file {do_path} already exists. Reading file in.")
                self._do_df= pd.read_csv(do_path)
                self._update_name_to_df_map('do', self._do_df)
                return
            else:
                do_df = pd.concat([do_df, do_algorithm(self, threshold=_do_thres)])
        if not do_df.empty:
            self._do_df = do_df
        self._update_name_to_df_map('do', self._do_df)
        
        return

    def ganz_method(self, force_agg=False, threshold = None):
        '''
        Input:
        - force_agg = whether you want to force the algorithm to run again
        - threshold = the threshold for the ganz algorithm 
        Output:
        - None (stores the df to self._ganz_df)
        '''
        if threshold is None:
            self._ganz_threshold = ganz_threshold_default
        elif isinstance(threshold, (int, float)) and threshold <= 1 and threshold >= 0:
            self._ganz_threshold = threshold
        else:
            raise Exception(f"Ganz threshold value {threshold} is either not a number or not a value between 0 and 1. Please edit and run ganz_method() again")
        

        ganz_path = self._construct_exported_df_path('ganz')
        if self._per_county.empty:
            raise Exception("_per_county is empty and is probably not loaded. Make sure ._load_raw_data is run and also run ._transform_per_county after loading. Stopping execution.")
        # TODO: modify the thresholds
        # Define the ranges and number of steps
        start = 0.001  # Minimum threshold
        end = 0.3      # Maximum threshold (similar to mixed_threshold method)
        num_steps = 10 # Number of uniform steps

        # Create uniform steps using np.linspace
        ganz_thresholds = np.linspace(start, end, num_steps)

        # Use the uniform steps in the loop
        ganz_df = pd.DataFrame({})
        for _ganzthres in np.arange(0, 1.001, step=0.1):
            _ganzthres = max(_ganzthres, 0.001)
            if force_agg:
                print("Forcing ganz algorithm")
                ganz_df = pd.concat([ganz_df, ganz_algorithm(self, threshold=_ganzthres)])
            elif ganz_path.exists():
                print(f"The file {ganz_path} already exists. Reading file in.")
                self._ganz_df = pd.read_csv(ganz_path)
                self._update_name_to_df_map('ganz', self._ganz_df)
                return
            else:
                ganz_df = pd.concat([ganz_df, ganz_algorithm(self, threshold=_ganzthres)])


        if not ganz_df.empty:
            self._ganz_df = ganz_df
        self._update_name_to_df_map('ganz', self._ganz_df)

        return
        
    def mixed_threshold_method(self, force_agg=False, relative_threshold = None, absolute_difference = None):
        '''
        import optuna

        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2

        study = optuna.create_study()
        study.optimize(objective, n_trials=100)

        study.best_params  # E.g. {'x': 2.002108042}
        '''
        '''
        Input:
        - force_agg = whether you want to force the algorithm to run again
        - relative_threshold = the relative threshold for the mixed threshold algorithm
        - absolute_difference = the absolute difference for the mixed threshold algorithm
        Output:
        - None (stores the df to self._mixed_threshold_df)
        '''
        if relative_threshold is None:
            self._mixed_threshold_values['relative_threshold'] = mixed_threshold_relative_threshold_default
        elif isinstance(relative_threshold, (int, float)) and relative_threshold >= 0:
            self._mixed_threshold_values['relative_threshold'] = relative_threshold
        else:
            raise Exception(f"Mixed Threshold's Relative Threshold value {relative_threshold} is either not a number or not a value above 0. Please edit and run do_method() again")
        
        if absolute_difference is None:
            self._mixed_threshold_values['absolute_difference'] = mixed_threshold_absolute_difference_default
        elif isinstance(absolute_difference, (int, float)) and absolute_difference >= 0:
            self._mixed_threshold_values['relative_threshold'] = relative_threshold
        else:
            raise Exception(f"Mixed Threshold's Relative Threshold value {relative_threshold} is either not a number or not a value above 0. Please edit and run do_method() again")
        

        mixed_threshold_path = self._construct_exported_df_path('mixed_threshold')
        if self._per_county.empty:
            raise Exception("_per_county is empty and is probably not loaded. Make sure ._load_raw_data is run and also run ._transform_per_county after loading. Stopping execution.")
        


        relative_thresh = self._mixed_threshold_values['relative_threshold']
        absolute_diff = self._mixed_threshold_values['absolute_difference']
        _mixed_df = pd.DataFrame({})
        # TODO: modify the thresholds, split the search range into 10 thresholds uniformly
        start = 0.05
        end = 0.3
        for _relthres in np.arange(start, end + 0.001, step=(end - start) / 10):
            for _abs_diff in range(1,11):
                
                if force_agg:
                    print("Forcing mixed_threshold algorithm")
                    _mixed_df = pd.concat([_mixed_df, mixed_threshold_algorithm(self, relative_threshold=_relthres, absolute_diff=_abs_diff)])
                elif mixed_threshold_path.exists():
                    print(f"The file {mixed_threshold_path} already exists. Reading file in.")
                    self._mixed_threshold_df= pd.read_csv(mixed_threshold_path)
                    self._update_name_to_df_map('mixed_threshold', self._mixed_threshold_df)
                    return
                else:
                    _mixed_df = pd.concat([_mixed_df, mixed_threshold_algorithm(self, relative_threshold=_relthres, absolute_diff=_abs_diff)])
        self._mixed_threshold_df = _mixed_df
        self._update_name_to_df_map('mixed_threshold', self._mixed_threshold_df)
        return
        
    def changepoints_method(self, force_agg=False):
        '''
        Input:
        - force_agg = whether you want to force the algorithm to run again
        Output:
        - None (stores the df to self._change_points_df)
        '''
        changepoints_path = self._construct_exported_df_path('changepoints')
        if self._per_county.empty:
            raise Exception("_per_county is empty and is probably not loaded. Make sure ._load_raw_data is run and also run ._transform_per_county after loading. Stopping execution.")
        if self._ground_truth_df.empty:
            raise Exception("_ground_truth_df is empty. Make sure _per_outage is loaded, _transform_per_outage() is run, and .create_groundtruth is run")

        if force_agg:
            print("Forcing changepoints algorithm")
            self._changepoints_df = changepoints_algorithm(self)
        elif changepoints_path.exists():
            print(f"The file {changepoints_path} already exists. Reading file in.")
            self._changepoints_df = pd.read_csv(changepoints_path)
        else: 
            self._changepoints_df = changepoints_algorithm(self)
        
        self._update_name_to_df_map('changepoints', self._changepoints_df)

        return

    def set_methods_to_run(self, methods_to_run = ["create_groundtruth, do_method", "ganz_method", "mixed_threshold_method","changepoints_method"]):
        self.methods_to_run = methods_to_run
        return

    def run_methods(self, methods_to_run = ["create_groundtruth", "do_method", "ganz_method", "mixed_threshold_method", "changepoints_method"], force_agg = False):
        self.set_methods_to_run(methods_to_run)

        for method_name in self.methods_to_run:
            method = getattr(self, method_name, None)
            if callable(method):
                method(force_agg)
            else:
                print(f"Method '{method_name}' not found or not callable")

    def generate_comparison_table(self, to_compare_list = list(valid_df_name_set)):
        ''' Creating the comparison table in which the different models are compared to see which one performs the best'''
        '''
        Input: list of algorithm-aggregated df's to put into the comparison table
        Returns: A table holding the average of the stats (number of outages, duration, number of customers out, weighted customers out) for the different methods applied (do, ganz, mixed_threshold, changepoints)
        '''
        
        def generate_summary_row(df_name: valid_df_name, dataframe):
            if df_name not in valid_df_name_set:
                raise Exception(
                    f"df_name {df_name} is not one of the names of the dataframes. "
                    f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
                )
            '''
            Generating new row for the comparison table
            '''
            # result_df = pd.DataFrame(dataframe).copy()
            # zero_dur_outage_count = len(result_df[result_df['duration'] == 0])
            
            # result_df = result_df[result_df['duration'] > 0]
            # frequency = len(result_df)
            # total_duration = result_df['duration'].sum()
            # total_avg_customers_out = result_df['average_customers_out'].sum()
            # total_weighted_customers_out =  result_df['weighted_customers_out'].sum()
            
            result_df = pd.DataFrame(dataframe).copy()
            result_df = result_df[result_df['duration'] > 0]
            zero_dur_outage_count = len(result_df[result_df['duration'] == 0])
            if result_df.empty:
                logging.warning(f"result_df {df_name} | param: {dataframe['params'].iloc[0]} is empty after filtering out zero dur")
                return pd.DataFrame()  # Return an empty DataFrame if result_df is empty
            model_name = None
            new_comparison_table_row = pd.DataFrame()
            # breakpoint()
            if df_name == 'ground_truth' or df_name == 'do' or df_name == 'ganz' or df_name == 'mixed_threshold':
                if df_name == 'ground_truth':
                    model_name = df_name
                # if df_name == 'do':
                #     model_name = f'Threshold: {self._do_threshold}'
                # if df_name == 'ganz':
                #     model_name = f'Threshold: {self._ganz_threshold}'
                # if df_name == 'mixed_threshold':
                #     model_name = f"Relative Threshold: {self._mixed_threshold_values['relative_threshold']}, Absolute Difference: {self._mixed_threshold_values['absolute_difference']}"
                elif df_name in ['do', 'ganz', 'mixed_threshold']:
                    model_name = dataframe['params'].iloc[0]
                frequency = len(result_df)
                total_duration = result_df['duration'].sum()
                total_avg_customers_out = result_df['average_customers_out'].sum()
                total_weighted_customers_out =  result_df['weighted_customers_out'].sum()
                try:
                    params = "N/A" if ('params' not in result_df.columns) else dataframe['params'].iloc[0]
                
                    new_row = pd.Series({
                        'model_type': df_name,
                        'model_name': model_name,
                        'frequency': frequency,
                        'average_duration': total_duration / frequency,
                        'average_customers_out': total_avg_customers_out / frequency,
                        'total_customer_outage_time': total_weighted_customers_out,
                        'average_customer_outage_time': total_weighted_customers_out / frequency,
                        'params': params,
                    })
                except Exception as e:
                    breakpoint()
                    logging.error(f"An error occurred: {e}")
                new_comparison_table_row = pd.concat([new_comparison_table_row, new_row.to_frame().T], axis=0, ignore_index=True)
            elif df_name == 'changepoints':
                # breakpoint()
                unique_models = result_df['params'].unique()
                for _param in unique_models:
                    subset = result_df[result_df['params'] == _param]

                    # penalty = subset['penalty'].iloc[0]
                    frequency = len(subset)
                    total_duration = subset['duration'].sum()
                    total_avg_customers_out = subset['average_customers_out'].sum()
                    total_weighted_customers_out = subset['weighted_customers_out'].sum()
                    new_row = pd.Series({
                        'model_type': df_name,
                        'model_name': _param,
                        'params': _param,
                        'frequency': frequency,
                        'average_duration': total_duration / frequency,
                        'average_customers_out': total_avg_customers_out / frequency,
                        'total_customer_outage_time': total_weighted_customers_out,
                        'average_customer_outage_time': total_weighted_customers_out / frequency
                    })
                    new_comparison_table_row = pd.concat([new_comparison_table_row, new_row.to_frame().T],axis=0, ignore_index=True)
            else:
                pass
            
            new_comparison_table_row.insert(0, 'state', self.config['state'])
            new_comparison_table_row.insert(1, 'layout', self.config['layout'])      
            new_comparison_table_row.insert(2, 'utility', self.config['name'])      
            new_comparison_table_row.insert(3, 'timeframe', f'{self._dataset_lower_bound} - {self._dataset_upper_bound}')
            new_comparison_table_row.insert(4, 'customer_population_per_county', self.config['customer_population'])

            return new_comparison_table_row

        
        comparison_table = pd.DataFrame()
        df_list = to_compare_list
        for df_name in df_list:
            # Adding df currently generated and stored in pipeline df_variable
            df = self._construct_and_get_df_variable(df_name)
            if df is None or df.empty:
                # Algorithm transformation method not run
                try:
                    self._load_aggregated_df(df_name)
                    self._transform_loaded_aggregated_df(df_name)
                except Exception as e:
                    if df_name == 'ground_truth':
                        raise
                    continue
            df = self._construct_and_get_df_variable(df_name)

            # add option for generating multiple DF's of each type and adding results to the 
            # breakpoint()
            # groupby model params
            if df_name in ['do', 'ganz', 'mixed_threshold']:
                grouped_df = df.groupby('params')
                for name, group in grouped_df:
                    new_table_row = generate_summary_row(df_name, dataframe=group)
                    if new_table_row.empty:
                        # continue  # HACK
                        logging.error(f"new_table_row is empty for {df_name} with params {name}. Skipping this group.")
                    comparison_table = pd.concat([comparison_table, new_table_row], ignore_index=True)
            else:
                new_table_row = generate_summary_row(df_name, dataframe=df)
                new_table_row = pd.DataFrame(new_table_row)

                comparison_table = pd.concat([comparison_table, new_table_row], ignore_index=True)
        
        
        # Adding ranks
        ground_truth_row = comparison_table[comparison_table['model_type'] == 'ground_truth'].copy().squeeze()

        comparison_table['frequency__diff'] = abs(ground_truth_row['frequency'] - comparison_table['frequency'])
        comparison_table['average_duration__diff'] = abs(ground_truth_row['average_duration'] - comparison_table['average_duration'])
        comparison_table['average_customers_out__diff'] = abs(ground_truth_row['average_customers_out'] - comparison_table['average_customers_out'])
        comparison_table['total_customer_outage_time__diff'] = abs(ground_truth_row['total_customer_outage_time'] - comparison_table['total_customer_outage_time'])

        comparison_table['frequency__percentage_difference'] = (comparison_table['frequency__diff'] / ground_truth_row['frequency']) * 100
        comparison_table['average_duration__percentage_difference'] = (comparison_table['average_duration__diff'] / ground_truth_row['average_duration']) * 100
        comparison_table['average_customers_out__percentage_difference'] = (comparison_table['average_customers_out__diff'] / ground_truth_row['average_customers_out']) * 100
        comparison_table['total_customer_outage_time__percentage_difference'] = (comparison_table['total_customer_outage_time__diff'] / ground_truth_row['total_customer_outage_time']) * 100
        
        # Setting 
        try:
            comparison_table['average_duration'] = comparison_table['average_duration'].apply(lambda x: timedelta(minutes = x))
            comparison_table['total_customer_outage_time'] = comparison_table['total_customer_outage_time'].apply(lambda x: timedelta(minutes = x))
            comparison_table['average_customer_outage_time'] = comparison_table['average_customer_outage_time'].apply(lambda x: timedelta(minutes = x))
        except Exception as e:
            breakpoint()
            logging.error(f"An error occurred: {e}")

        
        # self._comparison_table = comparison_table
        return comparison_table
        # model_rankings about 75% to 80% of the way through the file
        

    def compare(self):
        '''
        Runs transformation, standardization, ground truth creation, methods, and generates comparison table
        '''
        self.transform_datasets()
        self.run_methods()
        self._comparison_table = self.generate_comparison_table()
    
    def export_file(self, df_name:valid_df_name, output_folder = None, replace = False):
        '''
        Exports the file for the specified dataframe (ground_truth, do, ganz, mixed_threshold, changepoints, etc.) to the specified output folder
        Input: self, df_name (valid_df_name), output_folder (str), replace (bool)
        Output: None
        '''
        if df_name not in valid_df_name_set:
            raise Exception(
                f"df_name {df_name} is not one of the names of the dataframes. "
                f"Please make sure it is one of the following: {', '.join(valid_df_name_set)}"
            )
        
        self._set_output_path(output_path=output_folder)

        dataframe = self._construct_and_get_df_variable(df_name)
        if dataframe is None or dataframe.empty:
            raise Exception(
                f"The dataframe for {df_name} is None or empty. Make sure the respective method is run"
            )

        lower_date = self._dataset_lower_bound.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'
        upper_date = self._dataset_upper_bound.strftime('%Y-%m-%d') 

        logging.info(f"Exporting file for {df_name} for {self.config['name']}")

        df_file_path = self._construct_exported_df_path(df_name=df_name)
        # breakpoint()
        logging.info(f"Exporting {df_name} for {self.config['name']} to {df_file_path}")
        if os.path.isfile(df_file_path):
            if not replace:
                print(f"{df_file_path} already exists and not replacing. Not exporting...")
            else:
                print(f"{df_file_path} already exists but replacing. Exporting...")
                os.remove(df_file_path)
                dataframe.to_csv(df_file_path, index=False)
        
        else:
            # check if similar name file exists
            df_file_similar = glob.glob(os.path.join(self._output_path, f'*{self.config["name"]}_{df_name}*'))

            if df_file_similar:
                if replace:
                    print(f"Files similar to {self.config['name']}_{df_name} found but replacing them")
                    for file in df_file_similar:
                        os.remove(file)
                else:
                    print(f"Files similar to {self.config['name']}_{df_name} found but NOT replacing them")
            
            print(f"Exporting {df_name}...")
            dataframe.to_csv(df_file_path, index=False)


    def export_comparison_table(self, output_folder = None, replace = False):
        '''
        Exports the comparison table to the specified output folder
        Input: self, output_folder (str), replace (bool)
        Output: None
        '''
        self._set_output_path(output_path=output_folder)

        dataframe = self._comparison_table
        if dataframe is None or dataframe.empty:
            raise Exception(
                f"The dataframe for the comparison table is None or empty. Make sure self.generate_comparison_table is run"
            )

        lower_date = self._dataset_lower_bound.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'
        upper_date = self._dataset_upper_bound.strftime('%Y-%m-%d') 

        print(f"Exporting file for comparison table for {self.config['name']}")

        comparison_table_path = self._construct_comparison_table_path()

        if os.path.isfile(comparison_table_path):
            if not replace:
                print(f"{comparison_table_path} already exists and not replacing. Not exporting...")
            else:
                print(f"{comparison_table_path} already exists but replacing. Exporting...")
                os.remove(comparison_table_path)
                dataframe.to_csv(comparison_table_path, index=False)
        
        else:
            # check if similar name file exists
            df_file_similar = glob.glob(os.path.join(self._output_path, f'*{self.config["name"]}_comparison_table*'))

            if df_file_similar:
                if replace:
                    print(f"Files similar to {self.config['name']}_comparison_table found but replacing them")
                    for file in df_file_similar:
                        os.remove(file)
                else:
                    print(f"Files similar to {self.config['name']}_comparison_table found but NOT replacing them")
            
            print(f"Exporting comparison table...")
            dataframe.to_csv(comparison_table_path, index=False)

    def export_multiple_files(self, df_list, output_folder = None, replace = False):
        pass
