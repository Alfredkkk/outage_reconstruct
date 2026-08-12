import pandas as pd
import logging
import argparse
import numpy as np
from datetime import timedelta
from utils.ChangePointDetection import ChangePointDetection

from pathlib import Path
import concurrent.futures as cf

def do_algorithm(instance, threshold = 0.001):
    '''
    Input: None
    Output: per_county csv aggregated via Do methodology
    '''
    instance._per_county = instance._per_county.sort_values(by='timestamp')
    raw_df = (instance._per_county).copy()
    county_names = raw_df['county'].unique()

    def get_metric(group):
        county = group['county'].iloc[0]
        start_time = group['timestamp'].min()
        end_time = group['timestamp'].max()
        duration = ((end_time - start_time).total_seconds()) / 60 
        avg_customers_out = int(np.ceil(group['customersOutNow'].mean()))
        avg_customers_served = int(np.ceil(group['customersServed'].mean())) 
        avg_customer_out_x_duration = avg_customers_out * duration
        emc = group['utilityProvider'].iloc[0]

        return pd.Series([county, start_time, end_time, duration, avg_customers_out, avg_customer_out_x_duration, avg_customers_served, emc], index=['county','start_time', 'end_time', 'duration', 'average_customers_out', 'weighted_customers_out', 'customers_served', 'utility_provider'])

    final_df = pd.DataFrame()

    for county in county_names:
        county_df = raw_df[raw_df['county'] == county].copy()
        county_df['out'] = county_df['customersOutNow'] > county_df['customersServed'] * threshold
        county_df['outage_start'] = county_df['out'].eq(True) & (county_df['out'].shift(fill_value = 0).eq(False))
        county_df['outage_id'] = county_df['outage_start'].cumsum()
        minimum_out = county_df['customersServed'] * threshold
        county_df = county_df[(county_df['customersOutNow'] > minimum_out)] # since we have outage_id's marked already based on outage_start, we do not need the rows with False Threshold anymore

        county_df = county_df.groupby('outage_id').apply(get_metric, include_groups=False)

        # Adding to final dataframe
        final_df = pd.concat([final_df, county_df])

    # TODO: I've already done it, please check if this works
    final_df['params'] = f"threshold={threshold}"
    return final_df.reset_index(drop=True)

def ganz_algorithm(instance, threshold = 0.05):
    '''
    Input: None
    Output: per_county csv aggregated via Ganz methodology
    '''
    def loop_per_county(county_df):
        def get_metrics(group):
            county = group['county'].iloc[0]
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            duration = ((end_time - start_time).total_seconds()) / 60 
            avg_customers_out = int(np.ceil(group['customersOutNow'].mean()))
            avg_customers_served = int(np.ceil(group['customersServed'].mean())) 
            avg_customer_out_x_duration = avg_customers_out * duration
            emc = group['utilityProvider'].iloc[0]
            return pd.Series([county, start_time, end_time, duration, avg_customers_out, avg_customer_out_x_duration, avg_customers_served, emc], index=['county','start_time', 'end_time', 'duration', 'average_customers_out', 'weighted_customers_out', 'customers_served', 'utility_provider'])
                                                                                                                        
        def mark_local_max(county_df, outage_start_indices, outage_end_indices):
            county_df['max_outage'] = False
            if (len(outage_start_indices) != len(outage_end_indices)):
                error_message = f"""
                    Error: # of start indices != # of end indices
                    County where error occured: {county_df['county'].iloc[0]}
                    outage_start_indices: {outage_start_indices}
                    outage_end_indices: {outage_end_indices}
                """
                raise IndexError(error_message)

            start_before_end = all(start <= end
                for start, end in
                zip(outage_start_indices, outage_end_indices))
            
            if (not start_before_end):
                error_message = f"""
                    Index Error: The start indices are not before the end indices.
                    County where error occured: {county_df['name'].iloc[0]}
                    outage_start_indices: {outage_start_indices}
                    outage_end_indices: {outage_end_indices}
                """
                raise IndexError(error_message)

            for i in range(len(outage_start_indices)):
                start_index = outage_start_indices[i]
                end_index = outage_end_indices[i]

                temp_df = county_df.iloc[start_index:end_index+1]
                max_customers_out_index = temp_df['customersOutNow'].idxmax()
                county_df.at[max_customers_out_index, 'max_outage'] = True
            return county_df

        county_df = county_df.reset_index()
        county_df['above_thresh'] = county_df['customersOutNow'] > threshold * county_df['customersServed']

        county_df['outage_start'] = (county_df['above_thresh'] == True) & (county_df['above_thresh'].shift(fill_value = False) == False)
        county_df['outage_end'] = (county_df['above_thresh'].shift(periods=-1, fill_value = False) == False) & (county_df['above_thresh'] == True)
        outage_start_df = county_df[county_df['outage_start'] == True]
        outage_end_df = county_df[county_df['outage_end'] == True]
        outage_start_indices = outage_start_df.index.to_list()
        outage_end_indices = outage_end_df.index.to_list()

        county_df = mark_local_max(county_df, outage_start_indices, outage_end_indices)
        local_max_indices = (county_df[county_df['max_outage'] == True]).index.to_list()

        if (len(outage_start_indices) != len(local_max_indices) or len(local_max_indices) != len(outage_end_indices)):
            print("Error: # of outage_start indices != # of local_max indices or # of local_max indices != outage_end indices")
        else:
            county_final_df = []
            for i in range(len(outage_start_indices)):
                max_index = local_max_indices[i]
                end_index = outage_end_indices[i]

                temp_df = county_df.iloc[max_index:end_index+1]

                county_final_df.append(get_metrics(temp_df))
        return county_final_df
    
    final_df_v1_loop = []
    instance._per_county = instance._per_county.sort_values(by='timestamp')
    for county in instance._per_county['county'].unique():
        result = loop_per_county(instance._per_county[instance._per_county['county'] == county])
        final_df_v1_loop.extend(result)

    # TODO: I've already done it, please check if this works
    final_df = pd.DataFrame(final_df_v1_loop)
    final_df['params'] = f"threshold={threshold}"
    return final_df

def mixed_threshold_algorithm(instance, relative_threshold = 0.1, absolute_diff = 0):
    # determine if a row belongs to new outage
    def is_new(last_row, row):
        '''
        last_row = previous row of the original dataset
        row = current row of the original dataset
        '''
        if last_row is None:
            return True
        if (
            last_row["UtilityName"],
            # last_row["StateName"],
            last_row["CountyName"],
            # last_row["CityName"],
        ) != (row["UtilityName"], row["CountyName"]): # row["UtilityName"], row["StateName"], row["CountyName"], row["CityName"]
            return True
        if last_row["customersOutNow"] == 0:
            return True
        return False


    def show_tracking(tracking):
        r = ""
        for x in tracking:
            r += str(x["row_customers_out"])
            r += ", "
        r = r[:-2]
        return r


    def merger(outage_rows, threshold=relative_threshold, abs_diff=absolute_diff): # formerly threshold=0.1 and abs_diff = 0
        '''
        outage_rows = the [] containing rows from the original per_county df
        '''
        if len(outage_rows) == 0:
            return []
        results = []
        tracking = []
        prev_row = None
        for row in outage_rows:
            current = {} # current = the first row of the outage group (within the same county) to be grouped together into outages
                    # current becomes a row representing an outage instance
            # Basic information
            current["UtilityName"] = outage_rows[0]["UtilityName"]
            # current["StateName"] = outage_rows[0]["StateName"]
            current["CountyName"] = outage_rows[0]["CountyName"]
            # current["CityName"] = outage_rows[0]["CityName"]
            # current["CountyFIPS"] = outage_rows[0]["CountyFIPS"]
            current["customersServed"] = outage_rows[0]["customersServed"]
            current["timestamp"] = pd.to_datetime(row["timestamp"])

            # Must be new outage
            if tracking == []: # tracking holds all occuring outages
                
                current["start_time"] = current["timestamp"]
                current["end_time"] = current["timestamp"]
                current["row_customers_out"] = row["customersOutNow"] # the current # of customers out for an outage in tracking

                current["customer_affected_total"] = 0 
                current["average_customers_out"] = row["customersOutNow"]

                tracking.append(current)
                # logging.info(
                #     f"Case 0: Tracking is empty, create new outage ({show_tracking(tracking)})"
                # )
                prev_row = row
                continue

            last_outage = tracking[-1].copy()

            # Case A: increase of customers out is less than threshold, just update last outage
            if (
                row["customersOutNow"] - prev_row["customersOutNow"]
                < threshold * last_outage["row_customers_out"]
                or abs(row["customersOutNow"] - prev_row["customersOutNow"]) < abs_diff
            ) and row["customersOutNow"] >= prev_row["customersOutNow"]:
                # logging.debug(
                #     f"last_outage: {last_outage['average_customers_out']} will update with {row['customersOutNow'] - prev_row['customersOutNow']}"
                # )
                # logging.debug(
                #     f"Last_outage's row_customers_out: {last_outage['row_customers_out']}"
                # )
                tracking[-1]["end_time"] = current["timestamp"]

                tracking[-1]["row_customers_out"] = last_outage["row_customers_out"] + (
                    row["customersOutNow"] - prev_row["customersOutNow"]
                )

                tracking[-1]["customer_affected_total"] += (
                    current["timestamp"] - last_outage["end_time"]
                ).total_seconds() * last_outage["row_customers_out"]

                try:
                    tracking[-1]["average_customers_out"] = (
                        tracking[-1]["customer_affected_total"]
                        / (
                            tracking[-1]["end_time"] - tracking[-1]["start_time"]
                        ).total_seconds()
                    )
                except ZeroDivisionError as e:
                    # if 'end_time' == 'start_time', that means it is still early on in the outage and two consecutive rows have the same timestamp
                    # let 'average_customers_out' stay the same
                    pass
                # logging.info(
                #     f"Case A: increase of customers out is less than threshold, just update last outage, tracking: {show_tracking(tracking)}"
                # )
                prev_row = row
                continue

            # Case B: increase of customers out is more than threshold, create new outage
            elif (
                row["customersOutNow"] - prev_row["customersOutNow"]
                >= threshold * last_outage["row_customers_out"]
                and abs(row["customersOutNow"] - prev_row["customersOutNow"]) >= abs_diff
            ) and row["customersOutNow"] >= prev_row["customersOutNow"]:
                current["start_time"] = current["timestamp"]
                current["end_time"] = current["timestamp"]
                current["row_customers_out"] = (
                    row["customersOutNow"] - prev_row["customersOutNow"]
                )

                current["customer_affected_total"] = (
                    current["timestamp"] - current["start_time"]
                ).total_seconds() * current["row_customers_out"]

                current["average_customers_out"] = (
                    row["customersOutNow"] - prev_row["customersOutNow"]
                )
                tracking.append(current)
                # logging.info(
                #     f"Case B: increase of customers out is more than threshold, create new outage, tracking: {show_tracking(tracking)}"
                # )

                prev_row = row
                continue
            # Case C: decrease of customers out is less than threshold, just update last outage
            elif (
                prev_row["customersOutNow"] - row["customersOutNow"]
                < threshold * last_outage["row_customers_out"]
                or abs(row["customersOutNow"] - prev_row["customersOutNow"]) < abs_diff
            ) and row["customersOutNow"] <= prev_row["customersOutNow"]:
                tracking[-1]["end_time"] = current["timestamp"]
                tracking[-1]["row_customers_out"] = last_outage["row_customers_out"] - (
                    prev_row["customersOutNow"] - row["customersOutNow"]
                )

                tracking[-1]["customer_affected_total"] += (
                    current["timestamp"] - last_outage["end_time"]
                ).total_seconds() * last_outage["row_customers_out"]

                tracking[-1]["average_customers_out"] = (
                    tracking[-1]["customer_affected_total"]
                    / (
                        tracking[-1]["end_time"] - tracking[-1]["start_time"]
                    ).total_seconds()
                )
                # logging.info(
                #     f"Case C: decrease of customers out is less than threshold, just update last outage, tracking: {show_tracking(tracking)}"
                # )
                prev_row = row
                continue

            # Case D: decrease of customers out is more than threshold, record to first outage and add to results
            elif (
                prev_row["customersOutNow"] - row["customersOutNow"]
                >= threshold * last_outage["row_customers_out"]
                and abs(row["customersOutNow"] - prev_row["customersOutNow"]) >= abs_diff
            ) and row["customersOutNow"] <= prev_row["customersOutNow"]:

                decreasing_amount = prev_row["customersOutNow"] - row["customersOutNow"]

                indexes_to_pop = []

                closest_index = 0
                for outage in tracking:
                    if abs(decreasing_amount - outage["row_customers_out"]) < abs(
                        decreasing_amount - tracking[closest_index]["row_customers_out"]
                    ):
                        closest_index = tracking.index(outage)

                # Ideally, decreasing_amount should be closed to tracking[0]['average_customers_out']
                if abs(
                    decreasing_amount - tracking[closest_index]["row_customers_out"]
                ) >= max(10, 0.1 * decreasing_amount):
                    # logging.warning(
                    #     f"Problem: Decreasing amount {decreasing_amount} is not close to closest tracking {tracking[closest_index]['average_customers_out']}"
                    # )
                    tmp = ""
                    for x in outage_rows:
                        tmp += str(x["customersOutNow"]) + ","
                    tmp = tmp[:-1]
                    # logging.warning(f"Outage rows: {tmp}")
                    # logging.warning(f"We are tracking: {show_tracking(tracking)}")

                    # if decreasing amount <= closest_index, we should split this closest_index to 2 outages
                    if decreasing_amount <= tracking[closest_index]["row_customers_out"]:
                        # Create new outage
                        new_outage = tracking[closest_index].copy()
                        new_outage["row_customers_out"] = (
                            tracking[closest_index]["average_customers_out"]
                            - decreasing_amount
                        )
                        new_outage["average_customers_out"] = (
                            tracking[closest_index]["average_customers_out"]
                            - decreasing_amount
                        )
                        new_outage["customer_affected_total"] = (
                            new_outage["average_customers_out"]
                            * (
                                tracking[closest_index]["end_time"]
                                - tracking[closest_index]["start_time"]
                            ).total_seconds()
                        )

                        tracking.append(new_outage)

                        tracking[closest_index]["row_customers_out"] = decreasing_amount
                        tracking[closest_index][
                            "average_customers_out"
                        ] = decreasing_amount
                        tracking[closest_index]["customer_affected_total"] = (
                            tracking[closest_index]["average_customers_out"]
                            * (
                                tracking[closest_index]["end_time"]
                                - tracking[closest_index]["start_time"]
                            ).total_seconds()
                        )

                        indexes_to_pop.append(closest_index)

                        # logging.warning(
                        #     f"Case D-1: split to tracking: {show_tracking(tracking)}"
                        # )

                    else:
                        # logging.warning("Case D-2")
                        # indexes_to_pop.append(closest_index)
                        # sort tmp_l by descreasing average_customers_out
                        tracking = sorted(
                            tracking, key=lambda x: x["row_customers_out"], reverse=True
                        )

                        # Find the first outage that has row_customers_out < decreasing_amount
                        for outage in tracking:
                            if outage["row_customers_out"] < decreasing_amount:
                                closest_index = tracking.index(outage)
                                break

                        # Starting from closest_index, collect outages until the sum of row_customers_out >= decreasing_amount
                        sum_average_customers_out = 0
                        for i in range(closest_index, len(tracking)):
                            sum_average_customers_out += tracking[i]["row_customers_out"]
                            if sum_average_customers_out >= decreasing_amount:
                                sum_average_customers_out -= tracking[i][
                                    "row_customers_out"
                                ]
                                continue
                            indexes_to_pop.append(i)

                    # logging.warning("=====================================")
                else:
                    indexes_to_pop.append(closest_index)

                for index in indexes_to_pop:
                    tracking[index]["end_time"] = current["timestamp"]
                    # tracking[index]['row_customers_out'] = row['CustomersOut']

                    tracking[index]["customer_affected_total"] += (
                        current["timestamp"] - last_outage["end_time"]
                    ).total_seconds() * tracking[index]["row_customers_out"]

                    tracking[index]["average_customers_out"] = (
                        tracking[index]["customer_affected_total"]
                        / (
                            tracking[index]["end_time"] - tracking[index]["start_time"]
                        ).total_seconds()
                    )

                    tracking[index]["duration"] = (
                        tracking[index]["end_time"] - tracking[index]["start_time"]
                    )

                    tracking[index].pop("row_customers_out")
                    tracking[index].pop("customer_affected_total")
                    tracking[index].pop("timestamp")
                    # 'row_customers_out']

                    results.append(tracking[index])

                new_tracking = []
                for outage in tracking:
                    if tracking.index(outage) not in indexes_to_pop:
                        new_tracking.append(outage.copy())
                    else:
                        logging.debug(f"Pop outage: {outage['average_customers_out']}")

                tracking = sorted(new_tracking, key=lambda x: x["start_time"]).copy()

                # logging.info(
                #     f"Case D: decrease of customers out is more than threshold, record 'closest' outage and add to results, tracking: {show_tracking(tracking)}"
                # )

                prev_row = row
                continue
            # logging.error("Case E: Something wrong")

        # Add last tracking to results
        if len(tracking) > 0:
            # Close all outages in tracking
            for outage in tracking:

                outage["customer_affected_total"] += (
                    pd.to_datetime(prev_row["timestamp"]) - outage["end_time"]
                ).total_seconds() * outage["row_customers_out"]

                outage["end_time"] = pd.to_datetime(prev_row["timestamp"])
                outage["duration"] = outage["end_time"] - outage["start_time"]

                if (outage["end_time"] - outage["start_time"]).total_seconds() == 0:
                    # logging.error(f"{show_tracking(tracking)}")
                    # logging.error(f"outage: {outage}")
                    continue

                outage["average_customers_out"] = (
                    outage["customer_affected_total"]
                    / (outage["end_time"] - outage["start_time"]).total_seconds()
                )

                outage.pop("row_customers_out")
                outage.pop("customer_affected_total")
                outage.pop("timestamp")
                results.append(outage)

        return results


    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-l",
    #     "--loglevel",
    #     default="warning",
    #     help="Provide logging level. Example --loglevel debug, default=warning",
    # )

    # parser.add_argument(
    #     "-f",
    #     "--filename",
    #     default="test.csv",
    #     help="Provide filename. Example --filename test, default=test.csv",
    # )

    # parser.add_argument(
    #     "-t",
    #     "--threshold",
    #     default="0.0005",
    #     help="Provide threshold. Example --threshold 0.0005, default=0.0005",
    # )

    # parser.add_argument(
    #     "-a",
    #     "--abs_diff",
    #     default="0",
    #     help="Provide abs_diff. Example --abs_diff 0, default=0",
    # )


    # args = parser.parse_args()
    # logging.root.handlers = []
    # d = {
    #     "debug": logging.DEBUG,
    #     "info": logging.INFO,
    #     "warning": logging.WARNING,
    #     "error": logging.ERROR,
    #     "critical": logging.CRITICAL,
    # }

    # logging.basicConfig(
    #     level=d[args.loglevel.lower()],
    #     format="[%(levelname)s] %(message)s",
    #     handlers=[logging.StreamHandler()],
    # )

    # df = pd.read_csv("step0/" + args.filename + ".csv", encoding="utf-16")

    instance._per_county = instance._per_county.sort_values(by='timestamp')
    df = instance._per_county.copy()

    # show number of rows of df
    # logging.debug(df.shape[0])
    ## remove duplicate first, should not happen
    # idx = df.groupby(
    #     ["UtilityName", "StateName", "CountyName", "CityName", "RecordDateTime"]
    # )["customersOutNow"].idxmin()
    # df = df.loc[idx]
    # logging.debug(df.shape[0])


    # Convert each row into OutageRow object

    outage_rows = []  # belong to same outage # new list that holds the rows that should be counted together
    last_row = None
    result = []
    threshold = relative_threshold # 0.0005 # float(args.threshold)
    abs_diff = absolute_diff #0       # float(args.abs_diff)


    df.rename(columns={
        'utilityProvider': 'UtilityName',
        'county': 'CountyName'
    }, inplace = True)

    # for county in df['county'].unique():
    import time
    start_time = time.time()
    for index, row in df.iterrows():
        if is_new(last_row, row):
            if len(outage_rows) > 0:
                tmp = merger(outage_rows, threshold, abs_diff)
                if len(tmp) != 0:
                    result += tmp
            outage_rows = []
        outage_rows.append(row)
        last_row = row

    if len(outage_rows) > 0:
        tmp = merger(outage_rows, threshold, abs_diff)
        if len(tmp) != 0:
            result += tmp

    # Convert result to dataframe
    result_df = pd.DataFrame(result)
    result_df.rename(columns={
        'UtilityName': 'utility_provider',
        'CountyName': 'county',
        'customersServed': 'customers_served'
    }, inplace = True)

    # Filter out rows with "duration" == 0 seconds
    result_df['duration'] = result_df['duration'].apply(lambda x: x.total_seconds() / 60)
    result_df['weighted_customers_out'] = result_df['average_customers_out'] * result_df['duration']
    # TODO: I've already done it, please check if this works
    result_df['params'] = f"relative_threshold={relative_threshold} | absolute_diff={absolute_diff}"
    end_time = time.time()
    logging.info(f"Mixed_thres relative_threshold={relative_threshold} | absolute_diff={absolute_diff} Elapsed time: {(end_time - start_time):.2f} seconds")
    return result_df

def changepoints_algorithm(instance):
    # making sure the dataset columns are in the right format
    instance._ground_truth_df['start_time'] = pd.to_datetime(instance._ground_truth_df['start_time'], errors='coerce')
    instance._ground_truth_df['end_time'] = pd.to_datetime(instance._ground_truth_df['end_time'], errors='coerce')
    
    instance._per_county['customersOutNow'] = pd.to_numeric(instance._per_county['customersOutNow'])
    instance._per_county['customersServed'] = pd.to_numeric(instance._per_county['customersServed'])
    instance._per_county['timestamp'] = pd.to_datetime(instance._per_county['timestamp'], errors='coerce')
    
    train_per_outage = instance._ground_truth_df.copy() # should be test_per_outage since I am not training the model but naming for consistency
    # train_per_outage = train_per_outage.rename(columns={
    #     'duration_approx': 'duration'
    # })
    test_per_outage = instance._ground_truth_df.copy()
    test_per_outage = test_per_outage.rename(columns={
        'duration_approx': 'duration',
        'total_customer_outage_time': 'cust_affected_x_duration'
    })

    train_per_county = instance._per_county.copy() # should be test_per_county since I am not training the model but naming for consistency
    train_per_county = train_per_county.rename(columns={
        'customersOutNow': 'CustomersOut',
        'customersServed': 'CustomersTracked',
        'timestamp': 'RecordDateTime'
    })

    best_model_hist = {'rbf':8.395147e-01, 'normal':6.874084e-02,'rank':3.814707e-05,'l2':1.341115e-05,'l1':2.981232e-07}
    # best_model_hist = {'rbf': 0} # HACK
    best_model_hist = pd.DataFrame(list(best_model_hist.items()), columns=['mod', 'pen'])
    model_range = {'rbf': [1e-1, 1], 'normal': [1e-2, 1e-1], 'rank': [1e-5, 1e-4], 'l2': [1e-5, 1e-4], 'l1': [1e-7, 1e-6]}
    print("Right before creating Changepoint Instance")
    algo = ChangePointDetection(train_per_outage, train_per_county,start_date=instance._dataset_lower_bound, end_date=instance._dataset_upper_bound)

    merged_changepoints_df = pd.DataFrame()
    
    tasks = []
    # HACK
    # for _, row in best_model_hist.iterrows():
    #     mod = row['mod']
    #     for _pen in np.linspace(model_range[mod][0], model_range[mod][1], 10):
    #         algo.evaluate_model(algo.train_per_county, mod, _pen)
    with cf.ProcessPoolExecutor(max_workers=16) as executor:
        future_to_params = {}
        
        for _, row in best_model_hist.iterrows():
            mod = row['mod']
            for _pen in np.linspace(model_range[mod][0], model_range[mod][1], 10):
                _pen = np.round(_pen, 10)
                future = executor.submit(algo.evaluate_model, algo.train_per_county, mod, _pen)
                future_to_params[future] = (mod, _pen)
        
        for future in cf.as_completed(future_to_params):
            results_by_county, change_points_by_county = future.result()
            results_by_county['params'] = f"model={future_to_params[future][0]} | penalty={future_to_params[future][1]}"
            merged_changepoints_df = pd.concat([merged_changepoints_df, results_by_county], ignore_index=True)
            # merged_changepoints_df['params'] = f"model={future_to_params[future][0]} | penalty={future_to_params[future][1]}"
    merged_changepoints_df['utility_provider'] = instance._per_county['utilityProvider'].iloc[-1]
    
    return merged_changepoints_df