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


## CS224W

Application of Graph ML: [youtube](https://www.youtube.com/watch?v=aBHC6xzx9YI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=2), [blog](https://ok-lab.tistory.com/184)

Choice of Graph Representation: [youtube](https://www.youtube.com/watch?v=P-m1Qv6-8cI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=3), [blog](https://ok-lab.tistory.com/185)

Traditional feature-based methods: Node-level features: [youtube](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4), [blog](https://ok-lab.tistory.com/186) 

Traditional feature-based methods: Link-level features: [youtube](https://www.youtube.com/watch?v=4dVwlE9jYxY&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=5), [blog](https://ok-lab.tistory.com/216#Link-Level_feature)

Traditional feature-based methods: Graph-level features: [youtube](https://www.youtube.com/watch?v=buzsHTa4Hgs&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=6), [blog](https://ok-lab.tistory.com/217)

Random Walk Approaches for Node Embeddings : [youtube](https://www.youtube.com/watch?v=Xv0wRy66Big&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=8), [blog](https://ok-lab.tistory.com/218?category=937496)

## Paper ##

DeepWalk: online learning of social representations (KDD'14) [paper](https://arxiv.org/pdf/1403.6652.pdf)

LINE: Large-scale Information Network Embedding (WWW'15) [paper](https://arxiv.org/pdf/1503.03578.pdf)

Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeuIPS'16) [paper](https://arxiv.org/pdf/1606.09375.pdf)

Node2Vec: Scalable Feature Learning for Networks (KDD'16) [paper](https://arxiv.org/pdf/1607.00653.pdf)

Semi-supervised Classification with Graph Convolutional Networks (ICLR'17), [paper](https://arxiv.org/pdf/1609.02907.pdf), [code](https://github.com/ceo21ckim/DGL-tutorial/blob/main/models/GraphConv/Implementation.ipynb)

Neural Message Passing for Quantum Chemistry (PMLR'17) [paper](https://arxiv.org/pdf/1704.01212.pdf)

struc2vec: Learning Node Representations from Structural Identity (KDD'17) [paper](https://arxiv.org/pdf/1704.03165.pdf)

Inductive Representation Learning on Large Graphs (NeuIPS'17) [paper](https://arxiv.org/pdf/1706.02216.pdf)

HARP: Hierarchical Representation Learning for Networks (AAAI'18) [paper](https://arxiv.org/pdf/1706.07845.pdf)

Graph Convolutional Matrix Completion (KDD'18) [paper](https://arxiv.org/pdf/1706.02263.pdf), [code](https://github.com/ceo21ckim/DGL-tutorial/blob/main/models/GCMC/RecSys(GCMC).ipynb)

metapath2vec: Scalable Representation Learning for Heterogeneous Networks (KDD'17) [paper](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)

Watch Your Step: Learning Node Embedding via Graph Attention (NeurIPS'18) [paper](https://arxiv.org/pdf/1710.09599.pdf)

Simplifying Graph Convolutional Networks (PMLR'19) [paper](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)


**Survey**

Graph Embedding Techniques, Applications, and Performance: A Survey [paper](https://arxiv.org/pdf/1705.02801.pdf)
