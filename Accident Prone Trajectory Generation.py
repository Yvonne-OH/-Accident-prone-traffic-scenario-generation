import os
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from itertools import islice

#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:47:22 2024

@author: 39829
"""

import sys
#sys.argv = ["main.py", "--train","data/nba/rebound/train", "--test", "data/nba/rebound/test", "--ckpt", "log_rebound", "--config", "config/nba_rebound.py"]
#sys.argv = ["main.py", "--test","data/nba/rebound/test", "--ckpt", " models/nba/rebound", "--config", "config/nba_rebound.py"]
#sys.argv = ["main.py", "--test","data/univ/test", "--ckpt", " models/univ", "--config", "config/univ.py"]
sys.argv = ["main.py", "--test","data/Interation/DR_USA_Intersection_EP1", "--ckpt", " models\interaction\DR_USA_Intersection_EP1", "--config", "config/Interaction.py"]

import os, sys, time
import importlib
import torch
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=False)
parser.add_argument("--fpc-finetune", action="store_true", default=False)

"""
x: Input trajectory positions. Shape_x: (L1+1) x N x 6, where L1 is the length of the first trajectory, 
N is the number of trajectories, and 6 represents the trajectory information 
(e.g., x-coordinate, y-coordinate, velocity components).

y: Additional trajectory information (optional). Shape_y: L2 x N x 2 

neighbor: Neighbor information for each trajectory. Shape: (L1+L2+1) x N x Nn x 6, 
where L2 is the length of the second trajectory, 
Nn is the number of neighbors, and 6 represents neighbor information.
"""

#%%

def process_data_to_tensors(data, agent_threshold, ob_horizon, future_pre,device):
    
    N=ob_horizon+future_pre+2
    tensors_list = []
    num_items_to_process = 10
    processed_count=0

    grouped_data = data.groupby('case_id')
    

    for _, group in tqdm(grouped_data, desc="Processing Cases", total=num_items_to_process):
            df_selected = group[['frame_id', 'track_id', 'x', 'y', 'agent_type']]
            df_selected.columns = ['frame_ID', 'agent_ID', 'pos_x', 'pos_y', 'agent_type']
            df_selected = df_selected.sort_values(by='frame_ID')
            df_selected = df_selected.reset_index(drop=True)
            df_selected['frame_ID'] = df_selected['frame_ID'].astype(int)
            df_selected['agent_ID'] = df_selected['agent_ID'].astype(int)
            df_selected['pos_x'] = df_selected['pos_x'].round(5)
            df_selected['pos_y'] = df_selected['pos_y'].round(5)
    
            agent_ids_frame_1 = df_selected.loc[df_selected['frame_ID'] == 1, 'agent_ID'].unique()
            agent_ids_max_frame = df_selected.loc[df_selected['frame_ID'] == df_selected['frame_ID'].max(), 'agent_ID'].unique()
    
            if set(agent_ids_frame_1).issubset(set(agent_ids_max_frame)):
                unique_agent_ids = df_selected['agent_ID'].nunique()
                
    
                if unique_agent_ids >= agent_threshold:
                    windows = [df_selected[df_selected['frame_ID'].isin(range(start_frame, start_frame + ob_horizon + future_pre + 2))]
                               for start_frame in range(df_selected['frame_ID'].min(), df_selected['frame_ID'].max() - ob_horizon - future_pre + 1)]
                    #print(windows)
                    for scene_data in windows:            
                        Possible_ego_ids = scene_data['agent_ID'].value_counts() == ob_horizon + future_pre + 2
                        Possible_ego_ids_list = Possible_ego_ids[Possible_ego_ids].index.tolist()
                        
                        for Possible_ego in Possible_ego_ids_list:
                            ego_data=scene_data[scene_data['agent_ID'] == Possible_ego].reset_index(drop=True)
                            
                            ego_data['vx'] = ego_data['pos_x'].diff(-1).fillna(0)  # Negative diff for chronological order
                            ego_data['vy'] = ego_data['pos_y'].diff(-1).fillna(0)
                            
                            # Calculate ax and ay (differences in vx and vy)
                            ego_data['ax'] = ego_data['vx'].diff(-1).fillna(0)
                            ego_data['ay'] = ego_data['vy'].diff(-1).fillna(0)
                            
                            ego_data = ego_data.head(N-2).drop(columns=['agent_ID', 'frame_ID','agent_type'])         
                            hist_ego= np.array(ego_data.head(ob_horizon)).reshape(ob_horizon, 1, 6)                      
                            ground_truth_ego=np.array(ego_data.reset_index(drop=True).tail(future_pre)[['pos_x', 'pos_y']]).reshape(future_pre, 1, 2)
                           
                            neighbor=scene_data[scene_data['agent_ID'] != Possible_ego].reset_index(drop=True)
                            grouped = neighbor.groupby('agent_ID')
                            neighbor['vx'] = grouped['pos_x'].diff(-1)  # Fill NaN with 0 for first occurrence
                            neighbor['vy'] = grouped['pos_y'].diff(-1)
                            neighbor['ax'] = grouped['vx'].diff(-1)
                            neighbor['ay'] = grouped['vy'].diff(-1)
                            
                            neighbor =neighbor.sort_values(by="agent_ID").groupby('agent_ID').filter(lambda x: len(x) >= N)
                            a=neighbor
                                             
                            grouped = neighbor.groupby('frame_ID')
                            grouped_list = list(grouped)
                            # Remove the last two groups from the list
                            grouped_list = grouped_list[:-2]
                            
    
                            # Find the maximum size of any group
                            max_size = max(group_.shape[0] for _, group_ in grouped)
                            
                            # Pad each group with zeros if needed and convert to numpy array
                            padded_arrays = []
                            for _, group__ in grouped_list:
                                
                                group__ = group__.drop(columns=['agent_ID', 'frame_ID', 'agent_type'])
                                # Pad with zeros if the group is smaller than the max size
                                padded = np.pad(group__, ((0, max_size - group__.shape[0]), (0, 0)), mode='constant', constant_values=0)
                                padded_arrays.append(padded)
                                
                            neighbor=np.expand_dims(np.array(padded_arrays), axis=1)
                            
                           
                            neighbor=torch.from_numpy(neighbor).double().to(device)
                            x=torch.from_numpy(hist_ego).double().to(device)
                            y=torch.from_numpy(ground_truth_ego).double().to(device)
                            
                            tensors_list.append((x, y, neighbor))
                            
                            return_list=tensors_list
                            #print(len(return_list))
                        #print(len(return_list))
                    print("```````````````",len(return_list))
            if _>num_items_to_process:
                print("Break due to Num Limit")
                
                break                        
    return tensors_list,grouped_list
#%%
def Vehicle_Viz(ax, centers,angles,width, height,style="r-"):
    """

    :param ax: Matplotlib axis object.
    :param center: The coordinates (x, y) of the center of the vehicle.
    :param width: The width of the vehicle.
    :param height: The height of the vehicle.
    :param angle: Heading angle.
    """
   
    for center, angle in zip(centers, angles):
            # Calculate the half width and half height
            half_width = width / 2
            half_height = height / 2
    
            # Define the four corners of the rectangle based on the center, width, and height
            rectangle = np.array([
                [center[0] - half_width, center[1] - half_height],
                [center[0] + half_width, center[1] - half_height],
                [center[0] + half_width, center[1] + half_height],
                [center[0] - half_width, center[1] + half_height],
                [center[0] - half_width, center[1] - half_height]
            ])
    
            # Convert angle to radians
            theta = np.radians(angle)
    
            # Create rotation matrix
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
            # Rotate each corner of the rectangle around the center
            rotated_rectangle = np.dot(rectangle - center, rotation_matrix) + center
    
            # Draw the rotated rectangle
            ax.plot(rotated_rectangle[:, 0], rotated_rectangle[:, 1], style)

def calculate_headings(x, y):
    """
    Calculate the bearing angles between consecutive points and extend the last bearing.

    :param x: Array of x-coordinates.
    :param y: Array of y-coordinates.
    :return: Array of heading angles with the same length as x and y.
    """
    # Calculate the differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)

    # Calculate the bearing angles
    headings = np.arctan2(dy, dx)

    # Convert headings from radians to degrees
    headings = -np.degrees(headings)

    # Extend the last bearing
    headings = np.append(headings, headings[-1])

    return headings

def plot_trajectory(x,y,y_pred,neighbor,mode,EN_vehicle=True,Length=4,Width=1.8,Style='b--'):
    
    
    color_list = ['#F1D77E', '#d76364','#2878B5', '#9AC9DB', '#F8AC8C', '#C82423',
                      '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5",
                      '#F1D77E', '#d76364','#2878B5', '#9AC9DB', '#F8AC8C', '#C82423',
                      '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5",]
        
    
    fig, ax = plt.subplots()
    
        
    if mode=="Ego_Pred":
        # Drawing the trajectories
        plt.plot(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy(),
                 color='k', marker='o', markersize=6, markeredgecolor='black', markerfacecolor='k')
        
        plt.plot(y[:,0,0].cpu().detach().numpy(),y[:,0,1].cpu().detach().numpy(),
                 color='k', marker='*', markersize=10, markeredgecolor='black', markerfacecolor='g')
        
        if EN_vehicle:      
            headings=calculate_headings(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())          
            Vehicle_Viz(ax, list(zip(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())),headings,Length,Width,Style)
        
        # Loop for predictions
        for N in range(y_pred.cpu().detach().numpy().shape[0]):
            Pos_npred.append([y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy()])
            plt.plot(y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy(),
                     color=color_list[N], marker='o', markersize=6, markeredgecolor='black', markerfacecolor=color_list[N])
    
        plt.title('Trajectory Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
    if mode=="Scenario_Pred":
        # Drawing the trajectories
        
        plt.plot(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy(),
                 color='k', marker='o', markersize=6, markeredgecolor='black', markerfacecolor='k')
        
        plt.plot(y[:,0,0].cpu().detach().numpy(),y[:,0,1].cpu().detach().numpy(),
                 color='k', marker='*', markersize=1, markeredgecolor='black', markerfacecolor='g')
        
        if EN_vehicle:      
            headings=calculate_headings(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())          
            Vehicle_Viz(ax, list(zip(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())),headings,Length,Width,Style)
    
        # Loop for predictions
        for N in range(y_pred.cpu().detach().numpy().shape[0]):
            Pos_npred.append([y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy()])
            plt.plot(y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy(),
                     color=color_list[N], marker='o', markersize=1, markeredgecolor='black', markerfacecolor=color_list[N])
            
        neighbor_array= neighbor.cpu().detach().numpy().squeeze() 
      
        for i in range(neighbor_array.shape[1]):
            slice = neighbor_array[:, i, :]
            print(neighbor_array.shape[1])
            slice = slice[~(slice == 0).all(axis=1)]
            plt.plot(slice[:,0],slice[:,1], alpha=0.5)
            
            headings=calculate_headings(slice[:,0],slice[:,1])          
            Vehicle_Viz(ax, list(zip(slice[:,0],slice[:,1])),headings,Length,Width,Style)
           
    
            #print(f"Slice {i}:\n", slice)
        
        x_data = x[:, 0, 0].cpu().detach().numpy()
        y_data = x[:, 0, 1].cpu().detach().numpy()
        """
        x_range = (np.min(x_data) - 80 * (np.max(x_data) - np.min(x_data)), 
                   np.max(x_data) + 80 * (np.max(x_data) - np.min(x_data)))
        y_range = (np.min(y_data) - 80 * (np.max(y_data) - np.min(y_data)), 
                   np.max(y_data) + 80 * (np.max(y_data) - np.min(y_data)))
        
        plt.xlim(x_range)
        plt.ylim(y_range)
        """
    
        
        plt.title('Trajectory Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()

    return Pos_npred
#%%

if __name__ == "__main__":
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
        settings.device = "cpu"
    settings.device = torch.device(settings.device)
    
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=settings.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test:
        #print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
            
        #settings.test ---> File location
        
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
             


    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
   
    if settings.ckpt:
        
        ckpt = os.path.join(os.getcwd(),settings.ckpt.replace('/', '\\'),"ckpt-last").replace(" ", "")
        ckpt_best = os.path.join(os.getcwd(),settings.ckpt.replace('/', '\\'), "ckpt-best").replace(" ", "")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            start_epoch = state_dict["epoch"]
     
    #end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS
    #ade, fde = test(model, fpc)
    model.eval()
   
   
#%%
    # Define the file path
    agent_threshold = 5
    ob_horizon = 10
    future_pre = 25
       
    agent_threshold = 5
    Mode_select=["train","val"]
    current_directory = os.getcwd()
    

    mode="train"
    Data_Path=os.path.join(current_directory, "Data", "INTERACTION", "INTERACTION-Dataset-DR-multi-v1_2", mode)
    files = os.listdir(Data_Path)
    table_data = [(index + 1, file) for index, file in enumerate(files)]
    table_headers = ["Index", "File"]
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    
    file_name = files[10]
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
    
    tensor_data,a=process_data_to_tensors(data, agent_threshold, ob_horizon, future_pre,settings.device)
    #print(tensor_data)


#%%

    model.double()
    num=12
    x=tensor_data[num][0]
    y=tensor_data[num][1]
    neighbor=tensor_data[num][2]
    
    y_pred = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
    Pos_npred = []
    
    mode="Scenario_Pred"
    

    plot_trajectory(x,y,y_pred,neighbor,mode)


















