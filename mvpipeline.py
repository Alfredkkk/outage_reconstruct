import pandas as pd
import os
import json
import re
from typing import Literal
from datetime import datetime
from datetime import timedelta
from dateutil import tz
from base_mvpipeline import BaseMVPipeline

valid_df_name = Literal['ground_truth', 'do', 'ganz', 'mixed_threshold', 'changepoints']
valid_df_name_set = {'ground_truth', 'do', 'ganz', 'mixed_threshold', 'changepoints'}
eastern = tz.gettz('US/Eastern')
utc = tz.gettz('UTC')

class GA1TX8(BaseMVPipeline):
    def _transform_per_county(self):
        ''' 
        Add here any layout-specific per_county transformations
        '''
        super()._transform_per_county()
        self._per_county.rename(columns={
            'name': 'county',
            'customersOutNow': 'customersOutNow',
            'customersServed': 'customersServed',
            'timestamp': 'timestamp',
            'EMC': 'utilityProvider'
        }, inplace=True)
        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        self._per_county['timestamp'] = pd.to_datetime(self._per_county['timestamp'], utc=True).dt.tz_convert(eastern) 
        
        return

    def _transform_per_outage(self):
        '''
        Add here any layout_specific per_outage transformations 
        - dateTime transformations
        - error fixing (NA's, nulls)
        '''
        super()._transform_per_outage()
        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        self._per_outage['timestamp'] = pd.to_datetime(self._per_outage['timestamp'], utc=True).dt.tz_convert(eastern) 
        self._per_outage['outageStartTime'] = pd.to_datetime(self._per_outage['outageStartTime'], format='mixed', utc=True).dt.tz_convert(eastern)
        self._per_outage['zip'] = self._per_outage['zip'].replace(to_replace=["unknown", "Outage scale too large to extract zipcodes"], value=pd.NA)
        self._per_outage['outagePoint'] = self._per_outage['outagePoint'].apply(lambda x: json.loads(x.replace("'", '"')))
        self._per_outage[['lat', 'long']] = self._per_outage['outagePoint'].apply(lambda x: pd.Series([x['lat'], x['lng']]))

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
        variable_name = f"_{df_name}_df"
        dataframe = getattr(self, variable_name, None)
        if dataframe is None:
            raise Exception(
                f"Instance Variable ._{df_name}_df holds no value. Make sure it has the corresponding aggregated df. Maybe run _load_aggregated_df()?"
            )
        if dataframe.empty:
            raise Exception(
                f"Instance Variable ._{df_name}_df is an empty dataframe. Something may have went wrong with aggregating the raw file or when loading a preexisting aggregated df (like the raw data or imported file were originally empty)"
            )
        
        def safe_to_datetime(val): 
            try: 
                dt = pd.to_datetime(val) 
                return dt
            except (ValueError, pd.errors.OutOfBoundsDatetime): 
                print(f"Error converting: {val}") 
                return pd.NaT # or use np.nan if preferred
        
        # dataframe['start_time'] = pd.to_datetime(dataframe['start_time'], format='mixed', utc=True).dt.tz_convert(eastern)
        dataframe['start_time'] = dataframe['start_time'].apply(safe_to_datetime)
        dataframe['end_time'] = dataframe['end_time'].apply(safe_to_datetime)
        # dataframe['end_time'] = pd.to_datetime(dataframe['end_time'], format='mixed', utc=True).dt.tz_convert(eastern)
        dataframe['weighted_customers_out'] = pd.to_numeric(dataframe['weighted_customers_out'])
        dataframe['average_customers_out'] = pd.to_numeric(dataframe['average_customers_out'])
        dataframe['duration'] = pd.to_numeric(dataframe['duration'])

        if df_name == 'ground_truth':
            dataframe['duration_timestamp_diff'] = pd.to_numeric(dataframe['duration_timestamp_diff'])
            dataframe['duration_mean'] = pd.to_numeric(dataframe['duration_mean'])
            # dataframe['outage_point'] = dataframe['outage_point'].apply(lambda x: ast.literal_eval(x))
            # dataframe['outage_point'] = dataframe['outage_point'].apply(safe_literal_eval)
            
            dataframe['outage_point'] = dataframe['outage_point'].apply(transform_outage_point)



        setattr(self, variable_name, dataframe)

        return


    def create_groundtruth(self, force_agg = False):
        def _agg_vars(group):
            # the same as the compute_metric() method from aggregating county level data

            # Timestamp
            first_timestamp = group['timestamp'].min()
            last_timestamp = group['timestamp'].max()


            # Duration and start/end times
            duration_ts_diff = ((last_timestamp - first_timestamp).total_seconds()) / 60 
            start_time = group['outageStartTime'].min()
            duration_timestamp_start = ((last_timestamp - start_time).total_seconds()) / 60
            duration_approx = duration_timestamp_start

            end_time = start_time + timedelta(minutes=duration_approx)

            duration_max = max(duration_timestamp_start + 15, duration_ts_diff + 15) 
            duration_mean = (duration_timestamp_start + duration_max + duration_ts_diff) / 3


            ## Customers
            cust_restored = group['customersRestored'].max()

            # Customers Affected
                # Assumes time weight begins from current timestamp up to next timestamp
                # Left bounded?
            # group['duration_weight_lb'] = group['timestamp'].shift(-1) - group['timestamp']
            # group.loc[group.index[-1], 'duration_weight_lb'] = timedelta(minutes=15)

            group['duration_weight_lb'] = (
                group['timestamp'].shift(-1)
                .sub(group['timestamp'])
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_lb = (group['customersOutNow'] * group['duration_weight_lb']).sum()
            cust_a_mean_lb = cust_affected_x_duration_lb / group['duration_weight_lb'].sum()

                # Assumes time weight begins from previous timestamp up to current timestamp
                # Right bounded?
            group['duration_weight_rb'] = (
                group['timestamp'].diff()
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_rb = (group['customersOutNow'] * group['duration_weight_rb']).sum()
            cust_a_mean_rb = cust_affected_x_duration_rb / group['duration_weight_rb'].sum()

            cust_a_mean = round((cust_a_mean_lb + cust_a_mean_rb) / 2)

            cust_affected_x_duration_mean = (cust_affected_x_duration_rb + cust_affected_x_duration_lb) / 2


            # Location
            unique_coords = list(set(zip(group['long'], group['lat'])))
            lat = group['lat'].iloc[-1]
            longi = group['long'].iloc[-1]

            lat2 = group['lat'].unique()
            long2 = group['long'].unique()

            zipcode_map = self.geomap['zip_to_county_name']
            zipcode_list = group['zip'].unique()
            county_list = [
                self.geomap['zip_to_county_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA 
                for zipcode in zipcode_list
            ]
            county_fip_list = [
                self.geomap['zip_to_county_fips'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
                for zipcode in zipcode_list
            ]
            state_list = [
                self.geomap['zip_to_state_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
                for zipcode in zipcode_list
            ]
            # county_name = self.geomap['zip_to_county_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
            # county_fips = self.geomap['zip_to_county_fips'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
            # state = self.geomap['zip_to_state_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA

            utility_provider = group['EMC'].iloc[-1]

            return pd.Series({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration_approx,
                'duration_timestamp_diff': duration_ts_diff,
                'duration_mean': duration_mean,
                'duration_max': duration_max,
                'average_customers_out': cust_a_mean,
                'customer_restored_max': cust_restored,
                'weighted_customers_out': cust_affected_x_duration_mean,
                'outage_point': unique_coords,
                'zipcode': zipcode_list,
                'county_name': county_list,
                'county_fips': county_fip_list,
                'state': state_list,
                'utility_provider': utility_provider
            })

        exists = super().create_groundtruth(force_agg=force_agg)    
        if not exists:
            aggregated = self._per_outage.groupby('outageRecID').apply(_agg_vars).reset_index()
            self._ground_truth_df = aggregated

            self._update_name_to_df_map('ground_truth', self._ground_truth_df)
        
        return

class GA11TX12(BaseMVPipeline):
    def _transform_per_county(self):
        super()._transform_per_county()
        self._per_county.rename(columns={
            'Name': 'county',
            'out': 'customersOutNow',
            'count': 'customersServed',
            'timestamp': 'timestamp',
            'EMC': 'utilityProvider'
        }, inplace=True)

        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        self._per_county['timestamp'] = pd.to_datetime(self._per_county['timestamp'], utc=True).dt.tz_convert(eastern) 
        
        return
    
    def _transform_per_outage(self):
        def _reformat_start_date(row):
            month_day, time, ampm = row['start_time'].split(' ')
            s_month, s_day = month_day.split('/')
            year = None
            # Determining year using timestamp as start_date does not include year
            if pd.notna(row['timestamp']): 
                timestamp_components = row['timestamp'].split(' ')
                ts_date_comp = timestamp_components[0].split('-')
                t_month, t_day = ts_date_comp[0], ts_date_comp[1]
                t_year = pd.to_numeric(ts_date_comp[2])
                if t_month == '01' and s_month == '12':
                    year = str(int(t_year) - 1)
                else:
                    year = t_year 
            else:
            # for Walton, Tri-State, Oconee, and Mitchell, the na timestamps are in march 2023
                year = '2023'

            hour, minute = time.split(':')

            if 'am' in ampm.lower() and hour == '12':
                hour = '00' 
            if 'pm' in ampm.lower():
                hour = str(int(hour) + 12) if int(hour) < 12 else hour

            # Add leading zeros if necessary
            hour = hour.zfill(2)
            minute = minute.zfill(2)

            reformatted_date = f'{s_month}-{s_day}-{year} {hour}:{minute}:00'
            return reformatted_date

        def _reformat_update(row):
            month_day, time, ampm = row['updateTime'].split(',') 
            u_month, u_day = month_day.split(' ')
            month_dict = { 'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 
                        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12' }
            u_month = month_dict[u_month]
            
            year = None
            if pd.notna(row['timestamp']):
                timestamp_components = row['timestamp'].split(' ')
                ts_date_comp = timestamp_components[0].split('-')
                t_month, t_day = ts_date_comp[0], ts_date_comp[1]
                t_year = pd.to_numeric(ts_date_comp[2])
                if t_month == '01' and u_month == '12':
                    year = str(int(t_year) - 1)
                else:
                    year = t_year 
            else:
                year = '2023'

            hour, minute = time.split()
            if 'am' in ampm.lower() and hour == '12':
                hour = '00' 
            if 'pm' in ampm.lower():
                hour = str(int(hour) + 12) if int(hour) < 12 else hour
            hour = hour.zfill(2)
            minute = minute.zfill(2)

            reformatted_date = f'{u_month}-{u_day}-{year} {hour}:{minute}:00'
            return reformatted_date

        super()._transform_per_outage()
        self._per_outage = self._per_outage.rename(columns={
            'incident_id':'outage_id',
            'start_date': 'start_time',
            'zip_code':'zipcode',
            'consumers_affected': 'customer_affected'
        })


        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        try:
            self._per_outage['start_time'] = self._per_outage.apply(_reformat_start_date, axis=1)
            self._per_outage['start_time'] = pd.to_datetime(self._per_outage['start_time'], utc=True).dt.tz_convert(eastern)
            self._per_outage['updateTime'] = self._per_outage.apply(_reformat_update, axis=1) 
            self._per_outage['updateTime'] = pd.to_datetime(self._per_outage['updateTime'], utc=True).dt.tz_convert(eastern)
            self._per_outage['duration'] = pd.to_timedelta(self._per_outage['duration'])
            self._per_outage['duration'] = self._per_outage['duration'].apply(lambda x: x.total_seconds() / 60)
            self._per_outage['timestamp'] = pd.to_datetime(self._per_outage['timestamp'], utc=True).dt.tz_convert(eastern) 
            
        except Exception as e:
            print("An error occurred during _transform_per_outage for GA11TX12")
            print(f"Error: {e}")
            raise

        return
    
    def create_groundtruth(self, force_agg = False):
        def _agg_vars(group):
            """
            Overwriting superclass _compute_metrics as duration is given in this layout is more accurate
            """

            # Time and duration
            first_timestamp = group['timestamp'].min()
            last_timestamp = group['timestamp'].max()

            duration_dur_max = group['duration'].max() # 'duration' should be a number representing minutes
            duration_ts_diff = ((last_timestamp - first_timestamp).total_seconds()) / 60 
            duration_max = max(duration_dur_max + 15, duration_ts_diff + 15) # because 15 minute update intervals

            duration_approx = duration_dur_max

            duration_mean = (duration_max + duration_dur_max + duration_ts_diff) / 3

            duration_15 = 15 * len(group)
                        
            start_time = group['start_time'].min()
            end_time = start_time + timedelta(minutes=duration_approx)


            # group['duration_weight_lb'] = group['timestamp'].shift(-1) - group['timestamp']
            # group.loc[group.index[-1], 'duration_weight_lb'] = timedelta(minutes=15)

            # Customers affected
            group['duration_weight_lb'] = (
                group['timestamp'].shift(-1)
                .sub(group['timestamp'])
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_lb = (group['customer_affected'] * group['duration_weight_lb']).sum()
            cust_a_mean_lb = cust_affected_x_duration_lb / group['duration_weight_lb'].sum()

            group['duration_weight_rb'] = (
                group['timestamp'].diff()
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_rb = (group['customer_affected'] * group['duration_weight_rb']).sum()
            cust_a_mean_rb = cust_affected_x_duration_rb / group['duration_weight_rb'].sum()

            cust_affected_x_duration_mean = (cust_affected_x_duration_rb + cust_affected_x_duration_lb) / 2
            cust_a_mean = round((cust_a_mean_lb + cust_a_mean_rb) / 2)

            # location
            zipcode_map = self.geomap['zip_to_county_name']

            zipcode_list = group['zipcode'].unique()
            county_list = [
                self.geomap['zip_to_county_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA 
                for zipcode in zipcode_list
            ]
            county_fip_list = [
                self.geomap['zip_to_county_fips'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
                for zipcode in zipcode_list
            ]
            state_list = [
                self.geomap['zip_to_state_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
                for zipcode in zipcode_list
            ]
            utility_provider = group['EMC'].iloc[-1]


            return pd.Series({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration_approx,
                'duration_timestamp_diff': duration_ts_diff,
                'duration_mean': duration_mean,
                'duration_max': duration_max,
                'average_customers_out': cust_a_mean,
                'weighted_customers_out': cust_affected_x_duration_mean,
                'outage_point': list(set(zip(group['lon'], group['lat']))),
                'zipcode': zipcode_list,
                'county_name': county_list, 
                'county_fips': county_fip_list,
                'state': state_list,
                'utility_provider': utility_provider
            })
        
        exists = super().create_groundtruth(force_agg=force_agg)  
        if not exists:  
            aggregated = self._per_outage.groupby('outage_id').apply(_agg_vars).reset_index()
            self._ground_truth_df = aggregated
        return 

class GA3TX16(BaseMVPipeline):
    def _transform_per_county(self):
        super()._transform_per_county()
        self._per_county.rename(columns={
            'CountyName': 'county',
            'CustomersAffected': 'customersOutNow',
            'CustomersServed': 'customersServed',
            'timestamp': 'timestamp',
            'EMC': 'utilityProvider'
        }, inplace=True)

        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        self._per_county['timestamp'] = pd.to_datetime(self._per_county['timestamp'], utc=True).dt.tz_convert(eastern) 
        return
    
    def _transform_per_outage(self):
        super()._transform_per_outage()

        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')

        try:
            self._per_outage['timestamp'] = pd.to_datetime(self._per_outage['timestamp'], utc=True).dt.tz_convert(eastern) 
            self._per_outage['OutageTime'] = pd.to_datetime(self._per_outage['OutageTime'], utc=True).dt.tz_convert(eastern) 

            self._per_outage = self._per_outage.rename(columns={
                'X': 'long',
                'Y': 'lat'
            })
            
        except Exception as e:
            print("An error occurred during _transform_per_outage for GA11TX12")
            print(f"Error: {e}")
            raise

    def create_groundtruth(self, force_agg = False):
        def _agg_vars(group):
            """
            Overwriting superclass _compute_metrics as duration is given in this layout is more accurate
            """
            # Time and duration
            first_timestamp = group['timestamp'].min()
            last_timestamp = group['timestamp'].max()
            first_outage_time = group['OutageTime'].min()

            duration_ts_diff = ((last_timestamp - first_timestamp).total_seconds()) / 60 
            duration_outage_time_ts_diff = ((last_timestamp - group['OutageTime'].min()).total_seconds() / 60)
            duration_max = max(duration_ts_diff + 15, duration_outage_time_ts_diff + 15) # because 15 minute update intervals

            duration_approx = duration_ts_diff

            duration_mean = (duration_max + duration_ts_diff + duration_outage_time_ts_diff) / 3

            duration_15 = 15 * len(group)
                        
            start_time = first_outage_time
            end_time = start_time + timedelta(minutes=duration_approx)


            # group['duration_weight_lb'] = group['timestamp'].shift(-1) - group['timestamp']
            # group.loc[group.index[-1], 'duration_weight_lb'] = timedelta(minutes=15)

            # Customers affected
            group['duration_weight_lb'] = (
                group['timestamp'].shift(-1)
                .sub(group['timestamp'])
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_lb = (group['CutomersAffected'] * group['duration_weight_lb']).sum()
            cust_a_mean_lb = cust_affected_x_duration_lb / group['duration_weight_lb'].sum()

            group['duration_weight_rb'] = (
                group['timestamp'].diff()
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_rb = (group['CutomersAffected'] * group['duration_weight_rb']).sum()
            cust_a_mean_rb = cust_affected_x_duration_rb / group['duration_weight_rb'].sum()

            cust_affected_x_duration_mean = (cust_affected_x_duration_rb + cust_affected_x_duration_lb) / 2
            cust_a_mean = round((cust_a_mean_lb + cust_a_mean_rb) / 2)

            utility_provider = group['EMC'].iloc[-1]


            return pd.Series({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration_approx,
                'duration_timestamp_diff': duration_ts_diff,
                'duration_mean': duration_mean,
                'duration_max': duration_max,
                'average_customers_out': cust_a_mean,
                'weighted_customers_out': cust_affected_x_duration_mean,
                'outage_point': list(set(zip(group['long'], group['lat']))),
                'utility_provider': utility_provider

            })
        
        exists = super().create_groundtruth(force_agg=force_agg)    
        if not exists:
            aggregated = self._per_outage.groupby(['CaseNumber', 'OutageTime'], as_index=False).apply(_agg_vars).reset_index()
            self._ground_truth_df = aggregated
      
        return 
    


    
class CAInvestor(BaseMVPipeline):
    def _transform_per_county(self):
        ''' 
        Add here any layout-specific per_county transformations
        '''
        super()._transform_per_county()
        self._per_county.rename(columns={
            'county': 'county',
            'customersOutNow': 'customersOutNow',
            'customersServed': 'customersServed',
            'timestamp': 'timestamp',
            'EMC': 'utilityProvider'
        }, inplace=True)
        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        self._per_county['timestamp'] = pd.to_datetime(self._per_county['timestamp'], utc=True).dt.tz_convert(eastern) 
        
        return

    def _transform_per_outage(self):
        '''
        Add here any layout_specific per_outage transformations 
        - dateTime transformations
        - error fixing (NA's, nulls)
        '''
        super()._transform_per_outage()
        eastern = tz.gettz('US/Eastern')
        utc = tz.gettz('UTC')
        self._per_outage['timestamp'] = pd.to_datetime(self._per_outage['timestamp'], utc=True).dt.tz_convert(eastern) 
        # it's StartDate instead of outageStartTime
        self._per_outage['outageStartTime'] = pd.to_datetime(self._per_outage['StartDate'], format='mixed', utc=True).dt.tz_convert(eastern)
        # filter nan StartDate
        self._per_outage = self._per_outage[self._per_outage['outageStartTime'].notna()]
        # HACK we ignore zip here
        self._per_outage['zip'] = 0
        # self._per_outage['zip'] = self._per_outage['zip'].replace(to_replace=["unknown", "Outage scale too large to extract zipcodes"], value=pd.NA)
        # ignore outage point
        # self._per_outage['outagePoint'] = self._per_outage['outagePoint'].apply(lambda x: json.loads(x.replace("'", '"')))
        self._per_outage.rename(
            columns={
                "x": "long",
                "y": "lat"
            },
            inplace=True
        )

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
        variable_name = f"_{df_name}_df"
        dataframe = getattr(self, variable_name, None)
        if dataframe is None:
            raise Exception(
                f"Instance Variable ._{df_name}_df holds no value. Make sure it has the corresponding aggregated df. Maybe run _load_aggregated_df()?"
            )
        if dataframe.empty:
            raise Exception(
                f"Instance Variable ._{df_name}_df is an empty dataframe. Something may have went wrong with aggregating the raw file or when loading a preexisting aggregated df (like the raw data or imported file were originally empty)"
            )
        
        def safe_to_datetime(val): 
            try: 
                dt = pd.to_datetime(val) 
                return dt
            except (ValueError, pd.errors.OutOfBoundsDatetime): 
                print(f"Error converting: {val}") 
                return pd.NaT # or use np.nan if preferred
        
        # dataframe['start_time'] = pd.to_datetime(dataframe['start_time'], format='mixed', utc=True).dt.tz_convert(eastern)
        dataframe['start_time'] = dataframe['start_time'].apply(safe_to_datetime)
        dataframe['end_time'] = dataframe['end_time'].apply(safe_to_datetime)
        # dataframe['end_time'] = pd.to_datetime(dataframe['end_time'], format='mixed', utc=True).dt.tz_convert(eastern)
        dataframe['weighted_customers_out'] = pd.to_numeric(dataframe['weighted_customers_out'])
        dataframe['average_customers_out'] = pd.to_numeric(dataframe['average_customers_out'])
        dataframe['duration'] = pd.to_numeric(dataframe['duration'])

        if df_name == 'ground_truth':
            dataframe['duration_timestamp_diff'] = pd.to_numeric(dataframe['duration_timestamp_diff'])
            dataframe['duration_mean'] = pd.to_numeric(dataframe['duration_mean'])
            # dataframe['outage_point'] = dataframe['outage_point'].apply(lambda x: ast.literal_eval(x))
            # dataframe['outage_point'] = dataframe['outage_point'].apply(safe_literal_eval)
            
            dataframe['outage_point'] = dataframe['outage_point'].apply(transform_outage_point)



        setattr(self, variable_name, dataframe)

        return


    def create_groundtruth(self, force_agg = False):
        def _agg_vars(group):
            # the same as the compute_metric() method from aggregating county level data

            # Timestamp
            first_timestamp = group['timestamp'].min()
            last_timestamp = group['timestamp'].max()


            # Duration and start/end times
            duration_ts_diff = ((last_timestamp - first_timestamp).total_seconds()) / 60 
            start_time = group['outageStartTime'].min()
            duration_timestamp_start = ((last_timestamp - start_time).total_seconds()) / 60
            duration_approx = duration_timestamp_start

            end_time = start_time + timedelta(minutes=duration_approx)

            duration_max = max(duration_timestamp_start + 15, duration_ts_diff + 15) 
            duration_mean = (duration_timestamp_start + duration_max + duration_ts_diff) / 3


            ## Customers
            # HACK
            cust_restored = 0  # group['customersRestored'].max()

            # Customers Affected
                # Assumes time weight begins from current timestamp up to next timestamp
                # Left bounded?
            # group['duration_weight_lb'] = group['timestamp'].shift(-1) - group['timestamp']
            # group.loc[group.index[-1], 'duration_weight_lb'] = timedelta(minutes=15)

            group['duration_weight_lb'] = (
                group['timestamp'].shift(-1)
                .sub(group['timestamp'])
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_lb = (group['ImpactedCustomers'] * group['duration_weight_lb']).sum()
            cust_a_mean_lb = cust_affected_x_duration_lb / group['duration_weight_lb'].sum()

                # Assumes time weight begins from previous timestamp up to current timestamp
                # Right bounded?
            group['duration_weight_rb'] = (
                group['timestamp'].diff()
                .fillna(pd.Timedelta(minutes=15))
                .apply(lambda x: x.total_seconds() / 60)
            )
            cust_affected_x_duration_rb = (group['ImpactedCustomers'] * group['duration_weight_rb']).sum()
            cust_a_mean_rb = cust_affected_x_duration_rb / group['duration_weight_rb'].sum()

            cust_a_mean = round((cust_a_mean_lb + cust_a_mean_rb) / 2)

            cust_affected_x_duration_mean = (cust_affected_x_duration_rb + cust_affected_x_duration_lb) / 2


            # Location
            unique_coords = list(set(zip(group['long'], group['lat'])))
            lat = group['lat'].iloc[-1]
            longi = group['long'].iloc[-1]

            lat2 = group['lat'].unique()
            long2 = group['long'].unique()

            zipcode_map = self.geomap['zip_to_county_name']
            zipcode_list = group['zip'].unique()
            county_list = [
                self.geomap['zip_to_county_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA 
                for zipcode in zipcode_list
            ]
            county_fip_list = [
                self.geomap['zip_to_county_fips'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
                for zipcode in zipcode_list
            ]
            state_list = [
                self.geomap['zip_to_state_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
                for zipcode in zipcode_list
            ]
            # county_name = self.geomap['zip_to_county_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
            # county_fips = self.geomap['zip_to_county_fips'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA
            # state = self.geomap['zip_to_state_name'][zipcode] if (pd.notna(zipcode) and zipcode != '' and zipcode in zipcode_map) else pd.NA

            # utility_provider = group['EMC'].iloc[-1]
            # HACK
            utility_provider = 'ALL_UTIL'

            return pd.Series({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration_approx,
                'duration_timestamp_diff': duration_ts_diff,
                'duration_mean': duration_mean,
                'duration_max': duration_max,
                'average_customers_out': cust_a_mean,
                'customer_restored_max': cust_restored,
                'weighted_customers_out': cust_affected_x_duration_mean,
                'outage_point': unique_coords,
                'zipcode': zipcode_list,
                'county_name': county_list,
                'county_fips': county_fip_list,
                'state': state_list,
                'utility_provider': utility_provider
            })

        exists = super().create_groundtruth(force_agg=force_agg)    
        if not exists:
            # change to IncidentId instead of outageRecID
            aggregated = self._per_outage.groupby('IncidentId').apply(_agg_vars).reset_index()
            self._ground_truth_df = aggregated

            self._update_name_to_df_map('ground_truth', self._ground_truth_df)
        
        return