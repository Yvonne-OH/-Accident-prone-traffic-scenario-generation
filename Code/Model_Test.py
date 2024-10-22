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
def test(model, fpc=1):
    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
        0, len(test_dataset)
    ))
    tic = time.time()
    model.eval()
    ADE, FDE = [], []
    set_rng_state(init_rng_state, settings.device)
    batch = 0
    print("FPC:",int(fpc))
    fpc = int(fpc) if fpc else 1
 
    fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"

    with torch.no_grad():
        num_examples_to_output = 5
        for i, (x, y, neighbor) in enumerate(itertools.islice(test_data, num_examples_to_output)):
        #for x, y, neighbor in test_data:
            #print(x.shape,y.shape,neighbor.shape)
            batch += x.size(1)
            sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                batch, len(test_dataset), fpc_config, int(time.time()-tic)
            ))
            
            if config.PRED_SAMPLES > 0 and fpc > 1:
                # disable fpc testing during training
                y_ = []
                for _ in range(fpc):
                    """
                    generating multiple predictions (y_, config.PRED_SAMPLES) for 
                    a given input (x) and its associated neighbors (neighbor).
                    """
                    y_.append(model(x, neighbor, n_predictions=config.PRED_SAMPLES))
                    
                """
                y_ = torch.cat(y_, 0): Concatenates them along the specified dimension 
                (dimension 0 ) to create a single tensor y_ that contains all the predictions.   
                """
                y_ = torch.cat(y_, 0)
                
                """
                computing FPC (Final Prediction Confidence) for each predicted trajectory
                """                
                cand = []
                for i in range(y_.size(-2)):
                    cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=config.PRED_SAMPLES))
                    #print(cand)
                # n_samples x PRED_HORIZON x N x 2
                y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
            else:
                # n_samples x PRED_HORIZON x N x 2
                y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
            ade, fde = ADE_FDE(y_, y)
            if config.PRED_SAMPLES > 0:
                ade = torch.min(ade, dim=0)[0]
                fde = torch.min(fde, dim=0)[0]
            ADE.append(ade)
            FDE.append(fde)
    ADE = torch.cat(ADE)
    FDE = torch.cat(FDE)
    
    
    if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
        if not torch.is_tensor(config.WORLD_SCALE):
            config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
        ADE *= config.WORLD_SCALE
        FDE *= config.WORLD_SCALE
    ade = ADE.mean()
    fde = FDE.mean()
    sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
        ade, fde, fpc_config, 
        int(time.time()-tic))
    )
    print()
    return ade, fde

#%%

if __name__ == "__main__":
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        print(settings.test)
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

    for epoch in range(0,5):
        
        losses = None
        logger = True
        
        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        ade, fde = 10000, 10000
        perform_test = (train_data is None or epoch >= config.TEST_SINCE) and test_data is not None
        if perform_test:
            if not settings.no_fpc and not settings.fpc_finetune and losses is None and fpc_best > 1:
                fpc = fpc_best
            else:
                fpc = 1
            ade, fde = test(model, fpc)
            


#%%

    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
        0, len(test_dataset)
    ))
    tic = time.time()
    model.eval()
    ADE, FDE = [], []
    set_rng_state(init_rng_state, settings.device)
    batch = 0
    fpc = int(fpc) if fpc else 1
    fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"

    with torch.no_grad():
        
        num_examples_to_output = 5
        for i, (x, y, neighbor) in enumerate(itertools.islice(test_data, num_examples_to_output)):
        #for x, y, neighbor in test_data:
            print("____________________________")
            print(x.shape,y.shape,neighbor.shape)
            batch += x.size(1)
            sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                batch, len(test_dataset), fpc_config, int(time.time()-tic)
            ))
            
            if config.PRED_SAMPLES > 0 and fpc > 1:
                # disable fpc testing during training
                y_ = []
                for _ in range(fpc):
                    """
                    generating multiple predictions (y_, config.PRED_SAMPLES) for 
                    a given input (x) and its associated neighbors (neighbor).
                    """
                    y_.append(model(x, neighbor, n_predictions=config.PRED_SAMPLES))
                    
                """
                y_ = torch.cat(y_, 0): Concatenates them along the specified dimension 
                (dimension 0 ) to create a single tensor y_ that contains all the predictions.   
                """
                y_ = torch.cat(y_, 0) #FPC*n_predictions
                
                """
                computing FPC (Final Prediction Confidence) for each predicted trajectory
                """                
                cand = []
                for i in range(y_.size(-2)):
                    #
                    cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=config.PRED_SAMPLES))
                    #print(cand)
                    
                # n_samples x PRED_HORIZON x N x 2
                y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
            else:
                # n_samples x PRED_HORIZON x N x 2
                y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
            
            ade, fde = ADE_FDE(y_, y)
            if config.PRED_SAMPLES > 0:
                ade = torch.min(ade, dim=0)[0]
                fde = torch.min(fde, dim=0)[0]
            ADE.append(ade)
            FDE.append(fde)
    ADE = torch.cat(ADE)
    FDE = torch.cat(FDE)
    
    
    if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
        if not torch.is_tensor(config.WORLD_SCALE):
            config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
        ADE *= config.WORLD_SCALE
        FDE *= config.WORLD_SCALE
    ade = ADE.mean()
    fde = FDE.mean()
    sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
        ade, fde, fpc_config, 
        int(time.time()-tic))
    )
    print()
    
    
    
    print('Dataset size:', len(test_dataset))

    a=test_dataset.data[0][0][:,np.newaxis,:]
    b=test_dataset.data[0][1][:,np.newaxis,:]
    c=test_dataset.data[0][2][:,np.newaxis,:,:]
    
    num_examples_to_output = 10
    color_list = ['#F1D77E', '#d76364','#2878B5', '#9AC9DB', '#F8AC8C', '#C82423',
                  '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5",
                  '#F1D77E', '#d76364','#2878B5', '#9AC9DB', '#F8AC8C', '#C82423',
                  '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5",]
    
    for i, (x, y, neighbor) in enumerate(itertools.islice(test_data, num_examples_to_output)):
        
        #np.savetxt("neighbor.csv", neighbor[0,20,:,:].cpu().detach().numpy(), delimiter=',')

        y_=model(x, neighbor, n_predictions=config.PRED_SAMPLES)

        Pos_npred=[]
        
        print(y.cpu().detach().numpy().shape) 
        
        plt.plot(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy(),
                 color='k',
                 marker='o', markersize=6, markeredgecolor='black', markerfacecolor='k')
        
        
        plt.plot(y[:,0,0].cpu().detach().numpy(),y[:,0,1].cpu().detach().numpy(),
                 color='k',
                 marker='*', markersize=10, markeredgecolor='black', markerfacecolor='g')
        
        
        """
        for i in range(neighbor.cpu().detach().numpy().shape[2]):
            plt.plot(neighbor[:,:,i,0].cpu().detach().numpy(),neighbor[:,:,i,1].cpu().detach().numpy(),
                     color='g',
                     marker='.')"""
        
        
        for N in range(y_.cpu().detach().numpy().shape[0]):
            Pos_npred.append([y_[N,:,0,0].cpu().detach().numpy(),y_[N,:,0,1].cpu().detach().numpy()])
            

            plt.plot(y_[N,:,0,0].cpu().detach().numpy(),y_[N,:,0,1].cpu().detach().numpy(),
                     color='k',
                     marker='o', markersize=6, markeredgecolor='black', markerfacecolor=color_list[N])
            
            
        plt.title('Trajectory Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()
        


        
        
            












