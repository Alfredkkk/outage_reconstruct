import yaml
import pandas as pd
from mvpipeline import GA1TX8, GA11TX12, GA3TX16, CAInvestor
import time
import os
from pathlib import Path
import rich_click as click
import logging
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
    'ca_investor': CAInvestor
}

export_folder = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(export_folder, exist_ok=True)
timeout_per_utility = 600 # in seconds


def main():
    global export_folder

    with open('/root/autodl-tmp/mvpipeline/mvp_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        base_file_path = config['globals']['LOCAL_FILE_BASE_PATH']        

    def generate_ct(pipeline):
        comp_table = pipeline.generate_comparison_table()
        pipeline._comparison_table = comp_table
        pipeline.export_comparison_table(replace=True)


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
                    generate_ct(pipeline)
                except Exception as e:
                    logging.fatal(f"An error occurred: {e}")
                    raise          
                
                # export_files(pipeline, output_folder=export_folder, replace=True)
            
            else:
                raise ValueError(f"No matching class for key: {constructor_key}")
            

    start_time = time.time()
    run_config()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")



if __name__ == "__main__":
    main()