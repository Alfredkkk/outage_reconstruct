import yaml
import pandas as pd
from mvpipeline import GA1TX8, GA11TX12, GA3TX16, BaseMVPipeline, CAInvestor
import time
import os
import glob
from pathlib import Path
from datetime import timedelta
import rich_click as click
import logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s]%(filename)s:%(lineno)d-%(funcName)s:%(message)s',
    level=logging.INFO  # Set the desired logging level
)

erroneous_layout = []
model_rankings = pd.DataFrame()
valid_df_name_set = {'ground_truth', 'do', 'ganz', 'mixed_threshold', 'changepoints'}
constructor_map = {
    'ga_1': GA1TX8,
    'tx_8': GA1TX8,
    'ga_11': GA11TX12,
    'tx_12': GA11TX12,
    'ga_3': GA3TX16,
    'tx_16': GA3TX16,
    "ca_investor": CAInvestor,
    "ca_paloalto": GA1TX8,
}

export_folder = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(export_folder, exist_ok=True)
timeout_per_utility = 600 # in seconds

def export_files(pipeline, output_folder, replace = False):
    files_to_export = {
        "ground_truth": pipeline._ground_truth_df,
        "do": pipeline._do_df,
        "ganz": pipeline._ganz_df,
        "mixed_threshold": pipeline._mixed_threshold_df,
        "changepoints": pipeline._changepoints_df
    }

    os.makedirs(output_folder, exist_ok=True)
    transformed_data_folder = os.path.join(output_folder, "transformed_data")
    os.makedirs(transformed_data_folder, exist_ok=True)
    state_folder = os.path.join(transformed_data_folder, f"{pipeline.config['state']}")
    os.makedirs(state_folder, exist_ok=True)
    layout_folder = os.path.join(state_folder, f"layout_{pipeline.config['layout']}")
    os.makedirs(layout_folder, exist_ok=True)
    utility_folder = os.path.join(layout_folder, f"{pipeline.config['name']}")
    os.makedirs(utility_folder, exist_ok=True)

    lower_date = pipeline._dataset_lower_bound.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'
    upper_date = pipeline._dataset_upper_bound.strftime('%Y-%m-%d') 

    print(f"Exporting files for {pipeline.config['name']}")



    for file_type, df in files_to_export.items():
        if df is not None and not df.empty:
            df_file_name = f"{pipeline.config['name']}_{file_type}__{lower_date}_to_{upper_date}.csv"
            df_file_path = os.path.join(utility_folder, df_file_name)

            # check if exact file exists already
            if os.path.isfile(df_file_path):
                if not replace:
                    print(f"{df_file_path} already exists and not replacing. Not exporting...")
                else:
                    print(f"{df_file_path} already exists but replacing. Exporting...")
                    os.remove(df_file_path)
                    df.to_csv(df_file_path, index=False)
            
            else:
                # check if similar name file exists
                df_file_similar = glob.glob(os.path.join(utility_folder, f'*{pipeline.config["name"]}_{file_type}*'))

                if df_file_similar:
                    if replace:
                        print(f"Files similar to {pipeline.config['name']}_{file_type} found but replacing them")
                        for file in df_file_similar:
                            os.remove(file)
                    else:
                        print(f"Files similar to {pipeline.config['name']}_{file_type} found but NOT replacing them")
                
                print(f"Exporting {file_type}...")
                df.to_csv(df_file_path, index=False)


@click.command()
@click.option('--config_file', '-c', type=click.Path(exists=True), help='Path to the configuration file')
@click.option('--algo', '-a', 
              type=click.Choice(['mixed_threshold', 'changepoints', 'do', 'ganz']), help='Algorithm to run', multiple=True)
def main(config_file: str, algo: list[str]):
    global export_folder
    # breakpoint()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        base_file_path = os.path.expanduser(config['globals']['LOCAL_FILE_BASE_PATH'])       

    def test(pipeline):
        '''
        EDIT ME
        Function where you specify what about the pipeline you want to test (e.g. variables, functions)
        '''

        method_list = ['create_groundtruth', 'do_method', 'ganz_method', 'mixed_threshold_method', 'changepoints_method']
        pipeline._load_raw_data()
        pipeline.transform_datasets()
        # breakpoint()
        pipeline.create_groundtruth(force_agg = True)
        pipeline.export_file(df_name = 'ground_truth', replace = True)
        for _algo in algo:
            function_name = f"{_algo}_method"
            assert hasattr(pipeline, function_name), f"Function {function_name} not found in {pipeline.__class__.__name__}"
            getattr(pipeline, function_name)(force_agg=True)
            pipeline.export_file(df_name=_algo, replace=True)
                
        # pipeline.do_method(force_agg = True)
        # pipeline.export_file(df_name = 'do', replace = True)

        # pipeline.ganz_method(force_agg = True)
        # pipeline.export_file(df_name = 'ganz', replace = True)

        # pipeline.mixed_threshold_method(force_agg=True) #, relative_threshold=0.0005, absolute_difference = 0)
        
        # pipeline.export_file(df_name = 'mixed_threshold', replace = True)

        # pipeline._load_aggregated_df(df_name = 'ground_truth')
        # pipeline._transform_loaded_aggregated_df(df_name = 'ground_truth')
        # pipeline.changepoints_method(force_agg=True)
        # pipeline.export_file(df_name = 'changepoints', replace = True)

        # pipeline.run_methods(methods_to_run = method_list, force_agg = True)

        pipeline._comparison_table = pipeline.generate_comparison_table()



    def run_config():
        ''' 
        DO NOT EDIT ME
        '''
        global erroneous_layout
        erroneous_layout = []
        
        start_time = "2023-11-01 00:00:00"
        end_time = "2024-11-30 23:59:59"
        

        for provider in config['providers']:
            constructor_key = f"{provider['state']}_{provider['layout']}"
            if constructor_key in constructor_map:
                constructor = constructor_map[constructor_key]
                pipeline = constructor(provider, base_file_path, start_time = start_time, end_time = end_time)
                print()
                print(f"{provider['state']}_{provider['layout']} {provider['name']}")

                try:
                    # generate_ct(pipeline) #generate_ct(pipeline)
                    test(pipeline)
                except FileNotFoundError as e:
                    logging.error(f"file not found: {e}")
                    continue
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    raise          
                
                # export_files(pipeline, output_folder=export_folder, replace=True)
            
            else:
                raise ValueError(f"No matching class for key: {constructor_key}")
            

    def generate_ct(pipeline):
        comp_table = pipeline.generate_comparison_table()
        pipeline._comparison_table = comp_table
        pipeline.export_comparison_table(replace=True)
    
    def model_rankings():
        
        global model_rankings
        start_time = "2023-11-01 00:00:00"
        end_time = "2024-11-30 23:59:59"
        
        abridged_start_time = pd.Timestamp(start_time).strftime('%Y-%m-%d')
        abridged_end_time = pd.Timestamp(end_time).strftime('%Y-%m-%d')

        model_rankings = pd.DataFrame()
        merged_comparison_table = pd.DataFrame()

        for provider in config['providers']:
            constructor_key = f"{provider['state']}_{provider['layout']}"
            if constructor_key in constructor_map:
                constructor = constructor_map[constructor_key]
                pipeline = constructor(provider, base_file_path, start_time = start_time, end_time = end_time)
                print()
                print(f"{provider['state']}_{provider['layout']} {provider['name']}")

                try:
                    pipeline._load_comparison_table()
                    comp_table = pipeline._comparison_table

                    merged_comparison_table = pd.concat([merged_comparison_table, comp_table], ignore_index=True)

                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    # HACK
                    continue          
                
                # export_files(pipeline, output_folder=export_folder, replace=True)
            
            else:
                raise ValueError(f"No matching class for key: {constructor_key}")
            
        # breakpoint()
        # computing
        model_rankings = merged_comparison_table.groupby(['model_type', 'model_name']).agg({
            'frequency__percentage_difference': 'mean',
            'average_duration__percentage_difference': 'mean',
            'average_customers_out__percentage_difference': 'mean',
            'total_customer_outage_time__percentage_difference': 'mean'
        }).reset_index()

        model_rankings['rank_by_frequency_percentage'] = model_rankings['frequency__percentage_difference'].rank(method='min')
        model_rankings['rank_by_duration_percentage'] = model_rankings['average_duration__percentage_difference'].rank(method='min')
        model_rankings['rank_by_avg_customers_out_percentage'] = model_rankings['average_customers_out__percentage_difference'].rank(method='min')
        model_rankings['rank_by_customer_outage_time_percentage'] = model_rankings['total_customer_outage_time__percentage_difference'].rank(method='min')


        weights = {
            'rank_by_frequency_percentage': 0.5,
            'rank_by_duration_percentage': 0.2,
            'rank_by_avg_customers_out_percentage': 0.2,
            'rank_by_customer_outage_time_percentage': 0.1
        }
        
        model_rankings['weighted_sum_of_rank'] = (
            model_rankings['rank_by_frequency_percentage'] * weights['rank_by_frequency_percentage'] +
            model_rankings['rank_by_duration_percentage'] * weights['rank_by_duration_percentage'] +
            model_rankings['rank_by_avg_customers_out_percentage'] * weights['rank_by_avg_customers_out_percentage'] +
            model_rankings['rank_by_customer_outage_time_percentage'] * weights['rank_by_customer_outage_time_percentage']
        )

        model_rankings['rank_weighted_sum_of_rank'] = model_rankings['weighted_sum_of_rank'].rank(method='min')
        model_rankings.sort_values(by=['rank_weighted_sum_of_rank'], ascending=True, inplace=True)
        # model_rankings['rank_sum_of_rank'] = model_rankings['sum_of_rank'].rank(method='min')
        # model_rankings.sort_values(by=['rank_sum_of_rank'], 
        #                    ascending=True, inplace=True)


        model_rankings.rename(columns={
            'frequency__percentage_difference': 'mean__frequency_percentage_diff',
            'average_duration__percentage_difference': 'mean__average_duration_percentage_diff',
            'average_customers_out__percentage_difference': 'mean__average_customers_out_percentage_diff',
            'total_customer_outage_time__percentage_difference': 'mean__total_customer_outage_time_percentage_diff'
        }, inplace=True)

        columns_order = [
            'model_type', 'model_name',  # Model information first
            'rank_weighted_sum_of_rank', 'weighted_sum_of_rank', # Ranking information next
            'rank_by_frequency_percentage', 'rank_by_duration_percentage',
            'rank_by_avg_customers_out_percentage', 'rank_by_customer_outage_time_percentage',
            'mean__frequency_percentage_diff', 'mean__average_duration_percentage_diff',
            'mean__average_customers_out_percentage_diff', 'mean__total_customer_outage_time_percentage_diff'
        ]
        model_rankings = model_rankings[columns_order]

        model_rankings.reset_index(inplace=True, drop=True)

        merged_ct_path = Path(__file__).parent / 'output' / 'transformed_data' / f"merged comparison tables__{abridged_start_time}_to_{abridged_end_time}.csv"
        merged_comparison_table.to_csv(merged_ct_path)
        first_state_name = list(config['providers'])[0]['state']
        mr_path = Path(__file__).parent / 'output' / 'transformed_data' / f"model_rankings__{abridged_start_time}_to_{abridged_end_time}_{first_state_name}.csv"
        model_rankings.to_csv(mr_path)

    
    start_time = time.time()
    run_config()
    model_rankings()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")



if __name__ == "__main__":
    main()