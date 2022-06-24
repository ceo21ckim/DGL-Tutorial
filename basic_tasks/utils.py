from settings import *
import dgl
import pandas as pd 
import torch 
import torch.nn.functional as F 

def load_zachery():
    nodes_data = pd.read_csv(NODE_DIR)
    edges_data = pd.read_csv(EDGE_DIR)
    
    src = edges_data['Src'].values 
    dst = edges_data['Dst'].values 
    
    g = dgl.graph((src, dst))
    club = nodes_data['Club'].tolist()
    
    club = torch.tensor([c == 'Officer' for c in club], dtype=torch.long)
    
    club_onehot = F.one_hot(club)
    g.ndata.update({'club':club, 'club_onehot' : club_onehot})
    return g 