# DGL

Deep Graph Library(DGL)을 공부 목적으로 정리하고 있습니다. 본 코드는 `Pytorch`기반으로 작성되어 있습니다. [DGL 공식 문서](https://docs.dgl.ai/index.html#getting-started)와 [KDD20](https://github.com/dglai/KDD20-Hands-on-Tutorial), [WW20](https://github.com/dglai/WWW20-Hands-on-Tutorial)을 참고하였습니다. 공식 문서 내에 활용가능한 모델은 공식 문서 및 제안한 논문을 기반으로 작성했습니다. 

## Paper ##

GCN : Semi-supervised Classification with Graph Convolutional Networks (ICLR'17), [paper](https://arxiv.org/pdf/1609.02907.pdf) [code](https://github.com/ceo21ckim/DGL/blob/main/GraphConvolution/Implementation.ipynb)

GraphSAGE : Inductive Representation Learning on Large Graphs (NeuIPS'17) [paper](https://arxiv.org/pdf/1706.02216.pdf)

## Install DGL

***CPU***
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install -c dglteam dgl
```
`DGL`은 `python3.6` 버전 이상에서 사용이 가능합니다. `Pytorch 1.9.0+`, `Apache MXNet 1.6+`, `TensorFlow 2.3+`


## Docker Setting 
**1. 도커 이미지 불러오기**

***GPU***
```
docker pull dgllib/dgl-ci-gpu:cu11
```

***CPU***
```
docker pull dgllib/dgl-ci-cpu
```

**2. 컨테이너 실행**

***GPU***
```
# Docker version 2.0 or later.
docker run --itd --runtime=nvidia --name dgl_tutorial -p 8888:8888 -v C:\Users\Name\:/workspace dgl-ci-gpu:cu11 /bin/bash
```

```
# Docker-ce 19.03 or later
docker run -itd --gpus all --name dgl_tutorial -p 8888:8888 -v C:\Users\Name\:/workspace dgl-ci-gpu:cu11 /bin/bash
```

***CPU***
```
docker run -itd --name dgl_tutorial -p=8888:8888 -v C:\Users\Name\:/workspace dgl-ci-cpu /bin/bash
```

도커의 버전이 `19.03`이거나 이후 버전인 경우에는 `NVIDIA`를 따로 설치하지 않고 `--gpus all`만 사용하더라도 GPU 사용이 가능합니다. `pytorch`의 경우 작업 환경을 `/workspace`로 설정하기 때문에 동일하게 `/workspace`로 지정해주었습니다. port 연결을 통해 localhost에서도 접속이 가능합니다. `https://localhost:8888`


**3. 주피터 사용하기**
```
docker exec -it dgl_tutorial bash

pip install --upgrade pip

sudo pip install notebook 

sudo apt-get -y install ipython ipython-notebook

sudo -H pip install jupyter
```

