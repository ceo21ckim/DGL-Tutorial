import argparse, time

import numpy as np 

import torch 
import torch.nn as nn  
import torch.optim as optim 
import torch.nn.functional as F 

import dgl 
import dgl.function as fn 

from dgl.data import (
    register_data_args, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)

from dgl.nn.pytorch.conv import SGConv 

def calc_accuracy(pred, true):
    pred = torch.argmax(pred, dim=1)
    correct = pred.eq(true).sum()
    acc = correct / len(true)
    return acc 

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask]
        labels = labels[mask]
        correct = calc_accuracy(logits, labels)
        return correct 


def main(args):
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    

    g = data[0]

    if args.gpu :
        cuda = True 
        g = g.int().to('cuda:0')
    else:
        cuda = False
    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels 
    n_edges = g.number_of_edges()


    # add self loop 
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    model = SGConv(in_feats, n_classes, k=2, cached=True, bias=args.bias)

    if cuda :
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3: 
            t0 = time.time()
        
        # forward 
        logits = model(g, features)
        loss = criterion(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_mask)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Accuracy {acc:.4f} |")

    print()
    acc = evaluate(model, g, features, labels, test_mask)
    print(f'Test Accuracy {acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--bias', action='store_true', default=False)
    parser.add_argument('--weight-decay', type=float, default=5e-6)
    args = parser.parse_args()
    print(args)

    main(args)

