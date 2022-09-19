# DGL

Deep Graph Library(DGL)을 공부 목적으로 정리하고 있습니다. 본 코드는 `Pytorch`기반으로 작성되어 있습니다. [DGL 공식 문서](https://docs.dgl.ai/index.html#getting-started), [KDD20](https://github.com/dglai/KDD20-Hands-on-Tutorial), [WWW20](https://github.com/dglai/WWW20-Hands-on-Tutorial) 그리고 [WSDM21](https://github.com/dglai/WSDM21-Hands-on-Tutorial)을 참고하였습니다. 공식 문서 내에 활용가능한 모델은 공식 문서 및 제안한 논문을 기반으로 작성했습니다. 

## Install DGL

### Docker setting
**1.clone this repository**
``` 
git clone https://github.com/ceo21ckim/DGL.git
cd DGL
```

**2.build Dockerfile**
```
docker build --tag [filename]:1.0 .
```
`Dockerfile`을 build해서 사용하고 싶은 경우 원하는 filename을 지정하면 됩니다. 현재 위치에 `Dockerfile`을 같이 두고, 위 명령어를 입력하는 경우 build할 수 있습니다. 
`Dockerfile`은 `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime`를 내려받아 사용하였습니다. `pytorch` version과 `cuda` version이 다른 경우에는 본인의 환경과 맞게 설정하시면 됩니다. 아래의 환경 세팅은 image name을 `dgl_tutorial:1.0`으로 설정했습니다. 


**3.execute**

```
# Docker version 2.0 or later.
docker run --itd --runtime=nvidia --name dgl_tuto -p 8888:8888 -v C:\Users\Name\:/workspace dgl_tutorial:1.0 /bin/bash
```

```
# Docker-ce 19.03 or later
docker run -itd --gpus all --name dgl_tuto -p 8888:8888 -v C:\Users\Name\:/workspace dgl_tutorial:1.0 /bin/bash
```

도커의 버전이 `19.03`이거나 이후 버전인 경우에는 `NVIDIA`를 따로 설치하지 않고 `--gpus all`만 사용하더라도 GPU 사용이 가능합니다. `pytorch`의 경우 작업 환경을 `/workspace`로 설정하기 때문에 동일하게 `/workspace`로 지정해주었습니다. port 연결을 통해 localhost에서도 접속이 가능합니다. `https://localhost:8888`


**4.use jupyter notebook**
```
docker exec -it dgl_tuto bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
`port`는 처음 실행할 때 연결한 `port`를 지정하시면 됩니다. 

## Basic Tasks

- [Introduction](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/1.Introduction)
- [Node Classification](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/2.Node_Classification)
- [Link Prediction](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/3.Link_Prediction)
- [GNN](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/0.Others)
- [Message Passing](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/0.Others)


## CS224W

**Lecture 1**

Why Graphs: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=1)

Application of Graph ML: [youtube](https://www.youtube.com/watch?v=aBHC6xzx9YI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=2), [blog](https://ok-lab.tistory.com/184)

Choice of Graph Representation: [youtube](https://www.youtube.com/watch?v=P-m1Qv6-8cI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=3), [blog](https://ok-lab.tistory.com/185)

**Lecture 2**

Traditional feature-based methods: Node-level features: [youtube](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4), [blog](https://ok-lab.tistory.com/186) 

Traditional feature-based methods: Link-level features: [youtube](https://www.youtube.com/watch?v=4dVwlE9jYxY&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=5), [blog](https://ok-lab.tistory.com/216#Link-Level_feature)

Traditional feature-based methods: Graph-level features: [youtube](https://www.youtube.com/watch?v=buzsHTa4Hgs&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=6), [blog](https://ok-lab.tistory.com/217)

**Lecture 3**

Node Embedding: [youtube](https://www.youtube.com/watch?v=Xv0wRy66Big&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=7), [blog](https://ok-lab.tistory.com/218?category=937496)

Random Walk Approaches for Node Embeddings: [youtube](https://www.youtube.com/watch?v=Xv0wRy66Big&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=8), [blog](https://ok-lab.tistory.com/218?category=937496)

Embedding Entre Graphs: [youtube](https://www.youtube.com/watch?v=eliMLfJeu7A&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=9), [blog](https://ok-lab.tistory.com/222)

**Lecture 4**

PageRank: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=10), [blog](https://ok-lab.tistory.com/223)

PageRank: How to Solve?: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=11), [blog](https://ok-lab.tistory.com/223)

Random Walk with Restarts: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=12)

Matrix Factorization and Node Embeddings: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=13)

**Lecture 5**

Message passing and Node Classification: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=14)

Relational and Iterative Classification: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=15)

Collective Classification: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=16)

**Lecture 6**

Introduction to Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=17)

Basics of Deep Learning: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=18)

Deep Learning for Graphs: [youtube](https://www.youtube.com/watch?v=QUO-HQ44EDc&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=19)

**Lecture 7**

A general Perspective on GNNs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=20)

A Single Layer of a GNN: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=21)

Stacking layers of a GNN: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=22)

**Lecture 8**

Graph Augmentation for GNNs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=23)

Training Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=24)

Setting up GNN Prediction Tasks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=25)

**Lecture 9**

How Expressive are Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=26)

Designing the Most Powerful GNNs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=27)

**Lecture 10** 

Heterogeneous & Knowledge Graph Embedding: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=28)

Knowledge Graph Completion: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=29)

Knowledge Graph completion Algorithms: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=30)

**Lecture 11**

Reasoning in Knowledge Graphs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=31)

Answering Predictive Queries: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=32)

Query2box: Reasoning over KGs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=33)

**Lecture 12**

Fast Neural Subgraph Matching & Counting: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=34)

Neural Subgraph Matching: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=35)

Finding Frequent Subgraphs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=36)

**Lecture 13**

Community Detection in Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=37)

Network Communities: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=38)

Louvain Algorithm: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=39)

Detecting Overlapping Communities: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=40)

**Lecture 14**

Generative Models for Graphs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=41)

Erdos Renyi Random Graphs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=42)

The Small World Model: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=43)

Kronecker Graph Model: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=44)

**Lecture 15**

Deep Generative Models for Graphs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=45)

Graph RNN: Generating Realistic Graphs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=46)

Scaling Up & Evaluating Graph Gen: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=47)

Application of Deep Graph Generation: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=48)

**Lecture 16**

Limitations of Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=49)

Position-Aware Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=50)

Identity-Aware Graph Neural Network: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=51)

Robustness of Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=52)

**Lecture 17**

Scaling up Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=53)

GraphSAGE Neighbor Sampling: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=54)

Cluster GNN: Scaling up GNNs: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=55)

Scaling up by Simplifying GNN: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=56)

**Lecture 18**

GNNs in Computational Biology: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=57)

**Lecture 19**

Pre-Training Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=58)

Hyperbolic Graph Embeddings: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=59)

Design Space of Graph Neural Networks: [youtube](https://www.youtube.com/watch?v=TU0ankRcHmo&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=60)


## Paper ##

**Embedding**
| Name | Paper Link |
| ---- |:--------:|
|DeepWalk: online learning of social representations (KDD'14) | [paper](https://arxiv.org/pdf/1403.6652.pdf)|
|LINE: Large-scale Information Network Embedding (WWW'15)| [paper](https://arxiv.org/pdf/1503.03578.pdf)|
|Convolutional Networks on Graphs for Learning Molecular Fingerprints (NeurIPS'15)| [paper](https://arxiv.org/pdf/1509.09292.pdf)|
|Gated Graph Sequence Neural Networks (ICLR'16)| [paper](https://arxiv.org/pdf/1511.05493.pdf)|
|Node2Vec: Scalable Feature Learning for Networks (KDD'16)| [paper](https://arxiv.org/pdf/1607.00653.pdf)|
|metapath2vec: Scalable Representation Learning for Heterogeneous Networks (KDD'17)| [paper](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)|
|struc2vec: Learning Node Representations from Structural Identity (KDD'17)| [paper](https://arxiv.org/pdf/1704.03165.pdf)|
|Inductive Representation Learning on Large Graphs (NeurIPS'17)| [paper](https://arxiv.org/pdf/1706.02216.pdf)|
|HARP: Hierarchical Representation Learning for Networks (AAAI'18)| [paper](https://arxiv.org/pdf/1706.07845.pdf)|
|Watch Your Step: Learning Node Embedding via Graph Attention (NeurIPS'18)| [paper](https://arxiv.org/pdf/1710.09599.pdf)|
|Anonymous Walk Embeddings (ICML'18)| [paper](https://arxiv.org/pdf/1805.11921.pdf)|


**Model**
| Name | Paper Link | Code Link | Blog Link  |
| ---- |:--------:|:--------:|:--------:|
| DGCN: Diffusion-convolutional neural networks (NeurIPS'16) | [paper](https://arxiv.org/pdf/1511.02136.pdf) |
| ChebNet: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS'16) | [paper](https://arxiv.org/pdf/1606.09375.pdf) | - | [blog](https://ok-lab.tistory.com/221?category=940094) |
| GCN: Semi-supervised Classification with Graph Convolutional Networks (ICLR'17) | [paper](https://arxiv.org/pdf/1609.02907.pdf) | [code](https://github.com/ceo21ckim/DGL-tutorial/blob/main/models/GraphConv/Implementation.ipynb) | [blog](https://ok-lab.tistory.com/205?category=940094) |
| MPNN: Neural Message Passing for Quantum Chemistry (PMLR'17) | [paper](https://arxiv.org/pdf/1704.01212.pdf) | - | - |
| GraphSAGE: Inductive Representation Learning on Large Graphs (NeurIPS'17) | [paper](https://arxiv.org/pdf/1706.02216.pdf) | - | - |
| GAT: Graph Attention Networks (ICLR'18) | [paper](https://arxiv.org/pdf/1710.10903.pdf) | - | [blog](https://ok-lab.tistory.com/225?category=940094) |
| GCMC: Graph Convolutional Matrix Completion (KDD'18) | [paper](https://arxiv.org/pdf/1706.02263.pdf) | [code](https://github.com/ceo21ckim/DGL-tutorial/blob/main/models/GCMC/RecSys(GCMC).ipynb) | - |
| SEAL: Link Prediction Based on Graph Neural Networks (NeurIPS'18) | [paper](https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf) | - | - |
| LGCN: Large-Scale Learnable Graph Convolutional Networks (KDD'18) | [paper](https://arxiv.org/pdf/1808.03965.pdf) | - | - |
| SGCN: Simplifying Graph Convolutional Networks (ICML'19)| [paper](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf) | - | - |
| GraphRec: Graph Neural Network for social Networks (WWW'19) | [paper](https://arxiv.org/pdf/1902.07243.pdf) | - | [blog](https://ok-lab.tistory.com/226) |
| SGC: Simplifying Graph Convolutional Networks (PMLR'19) | [paper](https://arxiv.org/pdf/1902.07153.pdf) | [code](https://github.com/ceo21ckim/DGL-tutorial/blob/main/models/SGC/models.py) | [blog](https://ok-lab.tistory.com/224) |
| GNNExplainer: Generating Explanations for Graph Neural Networks (NeurIPS'19) | [paper](https://arxiv.org/pdf/1903.03894.pdf) | - | - |
| KGCN: Knowledge Graph Convolutional Networks for Recommender (WWW'19) | [paper](https://arxiv.org/pdf/1904.12575.pdf) | - | - |
| TokenGT: Pure Transformers are Powerful Graph Learners | [paper](https://arxiv.org/pdf/2207.02505.pdf) | - | - |

**Survey**
| Name | Paper Link |
| ---- |:--------:|
| Graph Embedding Techniques, Applications, and Performance: A Survey (2017) | [paper](https://arxiv.org/pdf/1705.02801.pdf) |
| A Survey on Influence Maximization in a Social Network (2018) | [paper](https://arxiv.org/pdf/1808.05502.pdf) |
| Random Graph Modeling: A Survey of the Concepts (2019) | [paper](https://dl.acm.org/doi/pdf/10.1145/3369782) |
| A Comprehensive Survey on Graph Neural Networks (2019) | [paper](https://arxiv.org/pdf/1901.00596.pdf) |


**Others**

| Name | Paper Link |
| ---- |:--------:|
|A New Models for Learning in Graph Domains (IJCNN'05)| [paper](https://www.researchgate.net/profile/Franco-Scarselli/publication/4202380_A_new_model_for_earning_in_raph_domains/links/0c9605188cd580504f000000/A-new-model-for-earning-in-raph-domains.pdf)|
|Disinformation on the Web: Impact, Characteristics, and Detection of Wikipedia Hoaxes (WWW'16)| [paper](https://dl.acm.org/doi/pdf/10.1145/2872427.2883085)|
|Model Degradation Hinders Deep Graph Neural Networks (KDD'22) | [paper] (https://arxiv.org/pdf/2206.04361.pdf) |

***Recommender Systems***
| Name | Paper Link | Code Link | Blog Link  |
| ---- |:--------:|:--------:|:--------:|
| Representing and Recommending Shopping Baskets with Complementarity, Compatibility and Loyalty (CIKM'18) | [paper](https://dl.acm.org/doi/pdf/10.1145/3269206.3271786) | - | - |
| NGCF: Neural Graph Collaborative Filtering (SIGIR'19) | [paper](https://arxiv.org/pdf/1905.08108.pdf)| [blog](https://ok-lab.tistory.com/204?category=940094) | - |
| Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters (PRML'20) | [paper](http://proceedings.mlr.press/v119/yu20e/yu20e.pdf)| - | - |
| LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR'20) | [paper](https://arxiv.org/pdf/2002.02126.pdf) | [blog](https://ok-lab.tistory.com/200?category=940094) | - | - |
| MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems (KDD'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467408) | - | - |
| UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation (CIKM'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3459637.3482291) | - | - |
| Self-supervised Learning for Large-scale Item Recommendations (CIKM'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3459637.3481952) | - | - |
| SGL: Self-supervised Graph Learning for Recommendation (SIGIR'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462862) | - | - |
| Are Graph Augmentations Necessary?: Simple Graph Contrastive Learning for Recommendation (SIGIR'22) | [paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531937) | - | - |
