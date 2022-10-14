import os 
from os.path import join 
import sys
from typing import ItemsView 
import torch 
import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset, DataLoader 
from scipy.sparse import csr_matrix
import scipy.sparse as sp 

from time import time 


class Loader(Dataset):
    def __init__(self, path="../data/yelp2018"):
        self.mode_dict = {'train':0, 'test':1}
        self.mode = self.mode_dict['train']

        self.n_user = 0 
        self.m_item = 0 
        
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path 

        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0 
        self.testDataSize = 0 

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0 :
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)

                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        
        self.traindUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0 :
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        self.m_item += 1 
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None 
        
        # (users, items), bipartite graph 
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.usesr_D[self.users_D == 0.] = 1 
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._testDict = self.__build_test()

    @property
    def n_users(self):
        return self.n_user

    @property    
    def m_items(self):
        return self.m_item 
    
    @property 
    def trainDataSize(self):
        return self.traindataSize
    
    @property 
    def testDict(self):
        return self._testDict
    


    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds 
        for i_fold in range(self.folds):
            start = i_fold*fold_len 
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items 
            else:
                end = (i_fold + 1)*fold_len 
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to('cuda'))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        idx = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(idx, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_add_mat.npz')
                print('succesfully loaded...')
                norm_adj = pre_adj_mat 
            
            except:
                print('generating adjacency matrix')
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R 
                adj_mat[self.n_users:, :self.n_users] = R.T 
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to('cuda')

            return self.Graph 

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        
        return posItems 

    def __getitem__(self, index):
        user = self.traindUniqueUsers[index]
        return user 
    
    def __len__(self):
        return len(self.traindUniqueUsers)