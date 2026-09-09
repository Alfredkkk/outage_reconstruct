import pandas as pd
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import pytz
from dateutil import tz
import json

import warnings

class ChangePointDetection:
    def __init__(self, growerdata, bluefiredata,
                 start_date=pd.Timestamp('2023-07-01 00:00:00').tz_localize('US/Eastern'),
                 end_date=pd.Timestamp('2023-08-01 00:00:00').tz_localize('US/Eastern')):
        self.best_frequency_diff = float('inf')
        self.best_fre_mod = 'l2'
        self.best_fre_pen = 0.1
        self.best_fre_change_points = {}
        self.mean_durations_seconds = float('inf')
        self.answer_avg_duration = float('inf')
        self.answer_frequency = float('inf')
        self.answer_avg_customer_affected_mean = float('inf')
        self.answer_total_customer_affected_mean_duration = float('inf')
        self.total_customer_affected_x_duration = float('inf')
        self.results_by_county = None
        self.change_points_by_county = None

        self.start_date = start_date
        self.end_date = end_date

        self.train_growerdata, self.train_bluefiredata = self.filter_data(growerdata, bluefiredata, start_date, end_date)
        self.test_growerdata = None
        self.test_bluefiredata = None

        self.best_model_hist = pd.DataFrame()

    def detect_outages(self, data, model, penalty):
        if data.empty:
            return pd.DataFrame(), []

        data = data.reset_index(drop=True)
        data['CustomersOut'] = pd.to_numeric(data['CustomersOut'], errors='coerce').fillna(0)
        data['CustomersTracked'] = pd.to_numeric(data['CustomersTracked'], errors='coerce').fillna(1)

        segments = []
        current_segment_start = 0
        last_non_zero = None
        prev_zero_TF = False

        for i, current_value in enumerate(data['CustomersOut']):
            if current_value == 0:
                if last_non_zero is not None and not prev_zero_TF:
                    segments.append((current_segment_start, i + 1))
                if not prev_zero_TF:
                    current_segment_start = i + 1
                last_non_zero = None
                prev_zero_TF = True
            else:
                if last_non_zero is None:
                    current_segment_start = i
                last_non_zero = current_value
                prev_zero_TF = False

        if last_non_zero is not None and current_segment_start < len(data):
            segments.append((current_segment_start, len(data)))

        all_segments = []
        outage_segments = []

        for start, end in segments:
            segment_data = data.iloc[start:end]
            segment_data['CustomersOut'] = segment_data['CustomersOut'].cummax()

            if end - start > 2:
                segment_data['OutageRatio'] = segment_data['CustomersOut'] / segment_data['CustomersTracked']
                points = segment_data['OutageRatio'].values
                algo = rpt.Pelt(model=model, min_size=1, jump=1).fit(points)

                try:
                    change_points = algo.predict(pen=penalty)
                    actual_change_points = [start + cp for cp in change_points if cp != 0]
                    all_segments.extend(actual_change_points)
                    start = actual_change_points[-1] if actual_change_points else start
                    if start < end:
                        all_segments.append(start)
                except Exception as e:
                    all_segments.append(start)
            else:
                all_segments.append(end)

        for start in all_segments:
            if start < len(data):
                end = all_segments[all_segments.index(start) + 1] if all_segments.index(start) + 1 < len(all_segments) else len(data)
                segment = data.iloc[start:end]
                if not segment.empty:
                    outage_segments.append({
                        'CountyFIPS': segment['CountyFIPS'].iloc[0],
                        'CityName': segment['CityName'].iloc[0],
                        'start_time': segment['RecordDateTime'].iloc[0],
                        'end_time': segment['RecordDateTime'].iloc[-1],
                        'average_customers_out': segment['CustomersOut'].mean(),
                        'duration': segment['RecordDateTime'].iloc[-1] - segment['RecordDateTime'].iloc[0],
                        'weighted_customers_out': segment['CustomersOut'].mean() * (segment['RecordDateTime'].iloc[-1] - segment['RecordDateTime'].iloc[0]).total_seconds() / 60,
                        'segment_label': f"Outage_{start}"
                    })

        return pd.DataFrame(outage_segments), all_segments

    def evaluate_model(self, data, mod, pen_val):
        grouped_data = data.reset_index().groupby(['CountyFIPS', 'CityName'])
        results_by_county = grouped_data.apply(lambda x: self.detect_outages(x, mod, pen_val)[0]).reset_index(drop=True)
        change_points_by_county = grouped_data.apply(lambda x: self.detect_outages(x, mod, pen_val)[1])
        return results_by_county, change_points_by_county

    def binary_search_best_penalty(self, data, mod, pen_min, pen_max, tolerance):
        max_iterations = 1000  # Define a maximum number of iterations
        no_improvement_count = 0
        max_no_improvement = 10
        iteration = 0
        model_best_frequency_diff = float('inf')

        while pen_max - pen_min > tolerance and iteration < max_iterations:
            pen_val = (pen_min + pen_max) / 2

            try:
                results_by_county, change_points_by_county = self.evaluate_model(data, mod, pen_val)
                frequency_diff = abs(len(results_by_county) - self.answer_frequency)
                print(frequency_diff, mod, pen_val)

                if frequency_diff < self.best_frequency_diff:
                    self.best_frequency_diff = frequency_diff
                    self.best_fre_mod = mod
                    self.best_fre_pen = pen_val
                    self.best_fre_change_points = change_points_by_county.to_dict()

                if frequency_diff < model_best_frequency_diff:
                    model_best_frequency_diff = frequency_diff
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= max_no_improvement:
                    print("No improvement for 10 iterations, stopping the search.")
                    break

                if len(results_by_county) > self.answer_frequency:
                    pen_min = pen_val
                elif len(results_by_county) < self.answer_frequency:
                    pen_max = pen_val
                else:
                    break
            except Exception as e:
                print(f"Error processing model {mod} with penalty {pen_val}: {e}")
                break

            iteration += 1

        if iteration >= max_iterations:
            print("Reached maximum iterations without convergence.")

        return pen_val

    def run(self, models=["l2", "rbf", "linear", "l1", "normal", "rank", "ar"], pen_min=0.000000001, pen_max=1, tolerance=0.0000001):
        self.answer_avg_duration = self.train_growerdata['duration'].mean() / 60
        self.answer_frequency = len(self.train_growerdata)
        self.answer_avg_customer_affected_mean = self.train_growerdata['customer_affected_mean'].mean()
        self.answer_total_customer_affected_mean_duration = self.train_growerdata['cust_affected_x_duration'].mean()
        self.total_customer_affected_x_duration = self.train_growerdata['cust_affected_x_duration'].sum()

        for mod in models:
            pen_val = self.binary_search_best_penalty(self.train_bluefiredata, mod, pen_min, pen_max, tolerance)
            new_row = {'mod': mod, 'pen': pen_val}
            new_row_df = pd.DataFrame([new_row])
            # if not ((self.best_model_hist == new_row_df.iloc[0]).all(1)).any():   # Checking if the row exists
            self.best_model_hist = pd.concat([self.best_model_hist, new_row_df], ignore_index=True)


    def filter_data(self, growerdata, bluefiredata, start_date, end_date):
        growerdata = growerdata[(growerdata['start_time'] >= start_date) & (growerdata['start_time'] <= end_date)]
        growerdata['duration'] = pd.to_timedelta(growerdata['end_time'] - growerdata['start_time'])

        bluefiredata = bluefiredata[(bluefiredata['RecordDateTime'] >= start_date) & (bluefiredata['RecordDateTime'] <= end_date)]
        bluefiredata['segment'] = 'Normal'
        return growerdata, bluefiredata

    def evaluate_best_model(self, growerdata, bluefiredata, start_date, end_date):
        self.test_growerdata, self.test_bluefiredata = self.filter_data(growerdata, bluefiredata, start_date, end_date)

        answer_avg_duration = self.test_growerdata['duration'].mean()
        answer_frequency = len(self.test_growerdata)
        answer_avg_customer_affected_mean = self.test_growerdata['customer_affected_mean'].mean()
        answer_total_customer_affected_mean_duration = self.test_growerdata['cust_affected_x_duration'].mean()
        total_customer_affected_x_duration = self.test_growerdata['cust_affected_x_duration'].sum()

        print("Original data statistics:")
        print("avg_duration: ", answer_avg_duration, "\nfrequency:", answer_frequency,
              "\navg_customer_affected_mean: ", answer_avg_customer_affected_mean,
              "\ntotal_customer_affected_mean_duration: ", answer_total_customer_affected_mean_duration,
              "\ntotal_customer_affected_x_duration: ", total_customer_affected_x_duration)

        print("\n\nBest model statistics:")
        results_by_county, change_points_by_county = self.evaluate_model(self.test_bluefiredata, self.best_fre_mod, self.best_fre_pen)
        print("Property for models: modelname-", self.best_fre_mod, " penalty value-", self.best_fre_pen,
              "\navg_duration: ", results_by_county['duration'].mean(), "\nfrequency:", len(results_by_county),
              "\navg_customer_affected_mean: ", results_by_county['average_customers_out'].mean(),
              "\ntotal_customer_affected_mean_duration: ", results_by_county['weighted_customers_out'].mean(),
              "\ntotal_customer_affected_x_duration: ", results_by_county['weighted_customers_out'].sum())
#         print(results_by_county.sort_values(by=['CountyFIPS','CityName','start_time'])[:60]) # for debugging
        self.best_fre_change_points = change_points_by_county.to_dict()

        return results_by_county['duration'].mean(), len(results_by_county), results_by_county['average_customers_out'].mean(), results_by_county['weighted_customers_out'].mean(), results_by_county['weighted_customers_out'].sum()

    def demonstrate_best_model(self):
        outage_counter = 0
        for (county, city), change_points in self.best_fre_change_points.items():
            county_city_data = self.test_bluefiredata[(self.test_bluefiredata['CountyFIPS'] == county) & (self.test_bluefiredata['CityName'] == city)]
            if change_points:
                county_city_indices = county_city_data.index.tolist()

                for start, end in zip([0] + change_points[:-1], change_points):
                    if end > start:
                        outage_counter += 1
                        idx_start = county_city_indices[start]
                        idx_end = county_city_indices[end - 1]
                        self.test_bluefiredata.loc[idx_start:idx_end, 'segment'] = f'Outage_{outage_counter}'
            else:
                if len(county_city_data) >= 1:
                    outage_counter += 1
                    self.test_bluefiredata.loc[county_city_data.index, 'segment'] = f'Outage_{outage_counter}'
        return self.test_bluefiredata