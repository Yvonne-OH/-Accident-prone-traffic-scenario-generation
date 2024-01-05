import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate


agent_threshold = 5
Mode_select=["train","val"]
current_directory = os.getcwd()

for mode in Mode_select:
    Data_Path=os.path.join(current_directory, "Data", "INTERACTION", "INTERACTION-Dataset-DR-multi-v1_2", mode)
    files = os.listdir(Data_Path)
    table_data = [(index + 1, file) for index, file in enumerate(files)]
    table_headers = ["Index", "File"]
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    
    for file_name in files:
        data = pd.read_csv((os.path.join(Data_Path,file_name)))
        
        #print(data.head())
        print(data.info())
        
        grouped_data = data.groupby('case_id')
        
        if "train" in file_name:
            output_directory = os.path.join(current_directory, "Data","Interation",file_name.rsplit('_', 1)[0],"train")
        elif "val" in file_name:
            output_directory = os.path.join(current_directory, "Data","Interation",file_name.rsplit('_', 1)[0],"val")
        else:
            raise ValueError("Wrong File---Please Check!")
        
        
        # Check if the directory exists
        if not os.path.exists(output_directory):
            # If it doesn't exist, create the directory
            os.makedirs(output_directory)
        else:
            # If it exists, delete all files in the directory
            files_in_output_directory = os.listdir(output_directory)
            for file_in_directory in files_in_output_directory:
                file_path = os.path.join(output_directory, file_in_directory)
                os.remove(file_path)
        
        grouped_data = data.groupby('case_id')
        
        for name, group in tqdm(grouped_data, desc="Processing Cases", total=len(grouped_data)):
            df_selected = group[['frame_id', 'track_id', 'x', 'y', 'agent_type']]
            df_selected.columns = ['frame_ID', 'agent_ID', 'pos_x', 'pos_y', 'agent_type']
        
            # Sort by 'frame_ID' in ascending order
            df_selected = df_selected.sort_values(by='frame_ID')
        
            # Reset the index
            df_selected = df_selected.reset_index(drop=True)
            # Ensure the data types of 'frame_ID' and 'agent_ID' columns are integers
            df_selected['frame_ID'] = df_selected['frame_ID'].astype(int)
            df_selected['agent_ID'] = df_selected['agent_ID'].astype(int)
        
            # Round 'pos_x' and 'pos_y' columns to three decimal places
            df_selected['pos_x'] = df_selected['pos_x'].round(3)
            df_selected['pos_y'] = df_selected['pos_y'].round(3)
        
            # Get unique agent_ID values for frame_id=1 and max frame_id
            agent_ids_frame_1 = df_selected.loc[df_selected['frame_ID'] == 1, 'agent_ID'].unique()
            agent_ids_max_frame = df_selected.loc[df_selected['frame_ID'] == df_selected['frame_ID'].max(), 'agent_ID'].unique()
        
            # Check if all agent_ID values for frame_id=1 are contained within agent_ID values for max frame_id
            if set(agent_ids_frame_1).issubset(set(agent_ids_max_frame)):
                # Check the number of unique agent_ID values
                unique_agent_ids = df_selected['agent_ID'].nunique()
        
                if unique_agent_ids >= agent_threshold:
                    # Convert each column to strings with left justification
                    df_str = df_selected.applymap(lambda x: str(x).ljust(15))
        
                    # Write the processed data frame to a text file without column names
                    output_filename = os.path.join(output_directory, f"{file_name[0:-4]}_case_{int(name)}.txt")
                    with open(output_filename, 'w', newline='') as file:
                        writer = csv.writer(file, delimiter='\t')
                        writer.writerows(df_str.values)        
                else:
                    pass
                    #print(f"Skipping case {name} due to fewer than {agent_threshold} unique agent_ID values.")
            else:
                pass
                #print(f"Skipping case {name} due to different agent_ID values for frame_id=1 and max frame_id.")
