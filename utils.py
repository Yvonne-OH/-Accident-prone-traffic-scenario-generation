import os
import torch
import random
import numpy as np 

def ADE_FDE(y_, y, batch_first=False):
    # average displacement error
    # final displacement error
    # y_, y: S x L x N x 2
    if torch.is_tensor(y):
        err = (y_ - y).norm(dim=-1)
    else:
        err = np.linalg.norm(np.subtract(y_, y), axis=-1)
    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
    return ade, fde

def kmeans(k, data, iters=None):
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]# Select the first k centroids

    if iters is None: iters = 100000
    for _ in range(iters):
    # while True:
        # Calculate Euclidean distances between each data point and each centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each data point to the nearest centroid
        closest = np.argmin(distances, axis=0)
        centroids_ = []
        for k in range(len(centroids)):
            cand = data[closest==k]
            if len(cand) > 0:
                centroids_.append(cand.mean(axis=0))
            else:
                # If a centroid has no assigned data points, 
                # randomly choose a data point as the new centroid
                centroids_.append(data[np.random.randint(len(data))])
        centroids_ = np.array(centroids_)
        # Check for convergence by measuring the change in centroids
        if np.linalg.norm(centroids_ - centroids) < 0.0001:
            break
        # Update centroids for the next iteration
        centroids = centroids_
    return centroids

def FPC(y, n_samples):
    """
    a simple algorithm for selecting goal points from a set of trajectories. 
    """
    # y: S x L x 2
    goal = y[...,-1,:2]  
    # Extracts the coordinates of the goal points at the last time step for each trajectory from the input array y, resulting in an array of shape (S, 2)
    goal_ = kmeans(n_samples, goal)  
    # Applies the kmeans clustering algorithm to cluster the goal points into n_samples clusters, obtaining an array of shape (n_samples, 2) representing cluster centers
    dist = np.linalg.norm(goal_[:, np.newaxis, :2] - goal[np.newaxis, :, :2], axis=-1)  
    # Computes the Euclidean distance from each trajectory's goal point to each cluster center, resulting in an array of shape (n_samples, S)
    chosen = np.argmin(dist, axis=1)  
    # For each cluster center, selects the index of the trajectory with the closest goal point, resulting in an array of shape (n_samples,) representing chosen trajectories
    return chosen

    
def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
