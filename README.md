# DGL 튜토리얼

Deep Graph Library(DGL)을 공부 목적으로 정리하고 있습니다. DGL-tutorial을 통해 그래프를 처음 접하시는 분들도 쉽게 작성할 수 있도록 도움을 주고자 `Pytorch`기반으로 작성하였습니다. [DGL 공식 문서](https://docs.dgl.ai/index.html#getting-started), [KDD20](https://github.com/dglai/KDD20-Hands-on-Tutorial), [WWW20](https://github.com/dglai/WWW20-Hands-on-Tutorial) 그리고 [WSDM21](https://github.com/dglai/WSDM21-Hands-on-Tutorial)을 참고하였습니다. 공식 문서 내에 활용가능한 모델은 공식 문서 및 제안한 논문을 기반으로 작성했습니다.



# Outline
1. [Install DGL](#Install-DGL)
2. [Basic Tasks](#Basic-Tasks)
3. [CS224W](https://github.com/ceo21ckim/DGL-tutorial/tree/main/CS224W)
4. [Conference Paper](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Conference_Paper)
5. [Conference Slide](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Conference_Slides)
6. [Models](#Models)

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
docker run -itd --runtime=nvidia --name dgl_tuto -p 8888:8888 -v C:\Users\Name\:/workspace dgl_tutorial:1.0 /bin/bash
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
- [Graph Classification](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/4.Graph_Classification)
- [Message Passing](https://github.com/ceo21ckim/DGL-tutorial/tree/main/Basic_Tasks/0.Others)



## Models

| Name | Title | Paper Link | Code Link |
| :----: | :---- |:--------:|:--------:|
| DGCN | Diffusion-convolutional neural networks (NeurIPS'16) | [paper](https://arxiv.org/pdf/1511.02136.pdf) |
|ChebNet | Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS'16) | [paper](https://arxiv.org/pdf/1606.09375.pdf) | - |
| GCN | Semi-supervised Classification with Graph Convolutional Networks (ICLR'17) | [paper](https://arxiv.org/pdf/1609.02907.pdf)              | - |
| VGAE | Variational Graph Auto-Encoders | [paper](https://arxiv.org/pdf/1611.07308.pdf) | - |
| R-GCN | Modeling Relational Data with Graph Convolutional Networks (ESWC'17) | [paper](https://arxiv.org/pdf/1703.06103.pdf) | - |
| MPNN | Neural Message Passing for Quantum Chemistry (PMLR'17) | [paper](https://arxiv.org/pdf/1704.01212.pdf) | - |
| GraphSAGE | Inductive Representation Learning on Large Graphs (NeurIPS'17) | [paper](https://arxiv.org/pdf/1706.02216.pdf) | - |
| GAT | Graph Attention Networks (ICLR'18) | [paper](https://arxiv.org/pdf/1710.10903.pdf) | - |
| GCMC | Graph Convolutional Matrix Completion (KDD'18) | [paper](https://arxiv.org/pdf/1706.02263.pdf) | - |
| SEAL | Link Prediction Based on Graph Neural Networks (NeurIPS'18) | [paper](https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf) | - |
| JK-Net | Representation Learning on Graphs with Jumping Knowledge Networks | [paper](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf)| - |
| LGCN | Large-Scale Learnable Graph Convolutional Networks (KDD'18) | [paper](https://arxiv.org/pdf/1808.03965.pdf) | - |
| GraphRec | Graph Neural Network for social Networks (WWW'19) | [paper](https://arxiv.org/pdf/1902.07243.pdf) | - |
| SGC | Simplifying Graph Convolutional Networks (PMLR'19) | [paper](https://arxiv.org/pdf/1902.07153.pdf) | - |
| HAN | Heterogenous Graph Attention Networks (WWW'19) | [paper](https://dl.acm.org/doi/pdf/10.1145/3308558.3313562) | - | 
| APPNP | Predict then Propagate: Graph Neural Networks meet Personalized PageRank (ICLR'19)| [paper](https://arxiv.org/pdf/1810.05997.pdf) | - |
| DGC | Diffusion Improves Graph Learning (NeurIPS'19) | [paper](https://arxiv.org/pdf/1911.05485.pdf) | - |
| GNNExplainer | GNNExplainer: Generating Explanations for Graph Neural Networks (NeurIPS'19) | [paper](https://arxiv.org/pdf/1903.03894.pdf) | - |
| DeepGCNs | Can GCNs Go as Deep as CNNs? (ICCV'19) | [paper](https://arxiv.org/pdf/1904.03751.pdf) | - |
| KGCN | Knowledge Graph Convolutional Networks for Recommender (WWW'19) | [paper](https://arxiv.org/pdf/1904.12575.pdf) | - |
| NDLS | Node Dependent Local Smoothing for Scalable Graph Learning (NeurIPS'21) | [paper](https://proceedings.neurips.cc/paper/2021/file/a9eb812238f753132652ae09963a05e9-Paper.pdf) | - |
| GemNet | Universal Directional Graph Neural Networks for Molecules (NeurIPS'21) | [paper](https://arxiv.org/pdf/2106.08903.pdf) | - |
| AIR | Model Degradation Hinders Deep Graph Neural Networks (KDD'22) | [paper](https://arxiv.org/pdf/2206.04361.pdf) | - |
| TokenGT | Pure Transformers are Powerful Graph Learners (NeurIPS'22) | [paper](https://arxiv.org/pdf/2207.02505.pdf) | - |
| PatchGT | PatchGT: Transformer over Non-trainable Clusters for Learning Graph Representations (LoG'22) | [paper](https://arxiv.org/pdf/2211.14425.pdf) | [code](https://github.com/tufts-ml/PatchGT) |
| Graphair | Learning Fair Graph Representations via Automated Data Augmentations (ICLR'23) | [paper](https://openreview.net/pdf?id=1_OGWcP1s9w) | - |

***Recommender Systems***
| Name | Paper Link |
| :---- |:--------:|
| Representing and Recommending Shopping Baskets with Complementarity, Compatibility and Loyalty (CIKM'18) | [paper](https://dl.acm.org/doi/pdf/10.1145/3269206.3271786) |
| NGCF: Neural Graph Collaborative Filtering (SIGIR'19) | [paper](https://arxiv.org/pdf/1905.08108.pdf)|
| Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters (ICML'20) | [paper](http://proceedings.mlr.press/v119/yu20e/yu20e.pdf)|
| LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR'20) | [paper](https://arxiv.org/pdf/2002.02126.pdf) |
| MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems (KDD'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467408) |
| UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation (CIKM'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3459637.3482291) |
| Self-supervised Learning for Large-scale Item Recommendations (CIKM'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3459637.3481952) |
| BUIR: Bootstrapping User and Item Representations for One-Class Collaborative Filtering (SIGIR’21)| [paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462935)|
| SGL: Self-supervised Graph Learning for Recommendation (SIGIR'21) | [paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462862) |
| Are Graph Augmentations Necessary?: Simple Graph Contrastive Learning for Recommendation (SIGIR'22) | [paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531937) |


**Others**

| Name | Paper Link |
| :---- |:--------:|
|A New Models for Learning in Graph Domains (IJCNN'05)| [paper](https://www.researchgate.net/profile/Franco-Scarselli/publication/4202380_A_new_model_for_earning_in_raph_domains/links/0c9605188cd580504f000000/A-new-model-for-earning-in-raph-domains.pdf)|
|Disinformation on the Web: Impact, Characteristics, and Detection of Wikipedia Hoaxes (WWW'16) | [paper](https://dl.acm.org/doi/pdf/10.1145/2872427.2883085)|
|Deeper insights into graph convolutional networks for semi-supervised learning (AAAI'18) | [paper](https://dl.acm.org/doi/pdf/10.5555/3504035.3504468) | 
| Directional Message Passing for Molecular Graphs (ICLR'20) | [paper](https://arxiv.org/pdf/2003.03123.pdf) |
|Model Degradation Hinders Deep Graph Neural Networks (KDD'22) | [paper](https://arxiv.org/pdf/2206.04361.pdf) |

***Survey***
| Name | Paper Link |
| :---- |:--------:|
| Graph Embedding Techniques, Applications, and Performance: A Survey (2017) | [paper](https://arxiv.org/pdf/1705.02801.pdf) |
| A Survey on Influence Maximization in a Social Network (2018) | [paper](https://arxiv.org/pdf/1808.05502.pdf) |
| Random Graph Modeling: A Survey of the Concepts (2019) | [paper](https://dl.acm.org/doi/pdf/10.1145/3369782) |
| A Comprehensive Survey on Graph Neural Networks (2019) | [paper](https://arxiv.org/pdf/1901.00596.pdf) |
| Graph Neural Networks in Recommender Systems: A Survey (ACM, 2022) | [paper](https://arxiv.org/abs/2011.02260) |

***Embedding***
| Name | Paper Link |
| :--------------------------------- |:-:|
|DeepWalk: online learning of social representations (KDD'14) | [paper](https://arxiv.org/pdf/1403.6652.pdf)|
|LINE: Large-scale Information Network Embedding (WWW'15)| [paper](https://arxiv.org/pdf/1503.03578.pdf)|
|Convolutional Networks on Graphs for Learning Molecular Fingerprints (NeurIPS'15)| [paper](https://arxiv.org/pdf/1509.09292.pdf)|
|Gated Graph Sequence Neural Networks (ICLR'16)| [paper](https://arxiv.org/pdf/1511.05493.pdf)|
|Node2Vec: Scalable Feature Learning for Networks (KDD'16)| [paper](https://arxiv.org/pdf/1607.00653.pdf)|
|metapath2vec: Scalable Representation Learning for Heterogeneous Networks (KDD'17)| [paper](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)|
|struc2vec: Learning Node Representations from Structural Identity (KDD'17)| [paper](https://arxiv.org/pdf/1704.03165.pdf)|
|HARP: Hierarchical Representation Learning for Networks (AAAI'18)| [paper](https://arxiv.org/pdf/1706.07845.pdf)|
|Watch Your Step: Learning Node Embedding via Graph Attention (NeurIPS'18)| [paper](https://arxiv.org/pdf/1710.09599.pdf)|
|Anonymous Walk Embeddings (ICML'18)| [paper](https://arxiv.org/pdf/1805.11921.pdf)|

