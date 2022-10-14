import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class BasicMF(nn.Module):
    def __init__(self, config, dataset):
        super(BasicMF, self).__init__()

        self.num_users = dataset.n_users 
        self.num_items = dataset.m_items 
        self.latent_dim = config['latent_dim_rec']

        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)
        self.sigmoid = nn.Sigmoid()
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.user_embedding(users)
        items_emb = self.item_embeding.weight 
        scores = torch.matmul(users_emb, items_emb.t())
        return self.sigmoid(scores)

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users 

        Return:
            (log-loss, l2-loss)
        """
        users_emb = self.user_embedding(users.long())
        pos_emb = self.item_embedding(pos.long())
        neg_emb = self.item_embedding(neg.long())

        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss 

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        scores = torch.sum(users_emb*items_emb, dim = 1) # torch.matmul(user_embed, item_embed)
        return self.sigmoid(scores)

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        self.config = config 
        self.dataset = dataset 
        
        self.num_users = self.dataset.n_users 
        self.num_items = self.dataset.m_items 
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        
        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)

        self.sigmoid = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        self._init_weight()

    def _init_weight(self):
        if self.config['pretrain'] == 0 :
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        else:
            self.user_embedding.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.item_embedding.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        
    def _dropout_x(self, x, keep_prob):
        size = x.size()
        idx = x.indices().t()
        values = x.values()
        random_idx = torch.rand(len(values)) + keep_prob 
        random_idx = random_idx.int().bool()
        idx = idx[random_idx]
        values = values[random_idx] / keep_prob 
        g = torch.sparse.FloatTensor(idx.t(), values, size)
        return g 
    
    def _dropout(self, keep_prob):
        graph = self._dropout_x(self.Graph, keep_prob)
        return graph 
    
    def computer(self):
        """
        propagate methods for LightGCN
        """
        user_emb = self.user_embedding.weight 
        item_emb = self.item_embedding.weight 
        all_emb = torch.cat([user_emb, item_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self._dropout(self.keep_prob)

            else:
                g_droped = self.Graph 
        
        else:
            g_droped = self.Graph 
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
        embs.append(all_emb)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items 
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating 
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.user_embedding(users_emb)
        pos_emb_ego = self.item_embedding(pos_emb)
        neg_emb_ego = self.item_embedding(neg_emb)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego 

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim = 1)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss 

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_product = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_product, dim=1)
        return gamma 