from time import time
import torch.optim as optim 


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay 
        loss = loss + reg_loss 

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    allPos = dataset.allPos 
    start = time()
    
    pass 


################################################################################

import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description = 'Go LightGCN')
    parser.add_argument('--bpr_batch', type=int, default = 2048, help = 'the batch size for bpr loss training procedure')
    parser.add_argument('--embedding_size', type=int, default=64, help = 'the embedding size of LightGCN')
    parser.add_argument('--layer', type=int, default=3, help='the layer number of LightGCN')
    parser.add_argument('--lr', type=float, default=0.001, help = 'the learning rate')
    parser.add_argument('--decay', type=float, default=1e-4, help = 'the weight decay for l2 normalization')
    parser.add_argument('--dropout', type=int, default=0, help = 'using the dropout or not')
    parser.add_argument('--a_flod', type=int, default=100, help = 'the fold num used to split large adj matrix, like gowalla')
    parser.add_argument('--testbatch', type=int, default=100, help = 'the batch size of users for testing')
    parser.add_argument('--dataset', type=str, default='yelp2018', help = 'available datasets: [lastfm, gowalla, yelp2018, amazon-book]')
    parser.add_argument('--path', type=str, default="./checkpoints", help = 'path to save weights')
    parser.add_argument('--topks', nargs = '?', default="[20]", help = "@k test list")
    parser.add_argument('--tensorboard', type=int, default=1, help = "enable tensorboard")
    parser.add_argument('--comment', type=str, default='lgn')
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1004, help = 'random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    return parser.parse_args()

    