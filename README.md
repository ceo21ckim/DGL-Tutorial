# DGL


## Docker Setting 
**1. 도커 이미지 불러오기**
```
docker pull dgllib/dgl-ci-gpu
```

**2. 컨테이너 실행**
Docker version 2.0 or later.
```
docker run --itd --runtime=nvidia --name dgl_tutorial -p 8888:8888 -v C:\Users\Name\:/workspace dgl-ci-gpu /bin/bash
```

```
# Docker-ce 19.03 or later
docker run -itd --gpus all --name dgl_tutorial -p 8888:8888 -v C:\Users\Name\:/workspace dgl-ci-gpu /bin/bash
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

`sudo -H pip install jupyter`에서 오류가 발생하는 경우에는 `sudo -H pip install --upgrade pip`를 먼저 수행한 후 진행하면 설치가 됩니다. 내려받은 `docker image`는 `Ubuntu 16.04` 이기에 위와 같은 방법으로 진행하였습니다.
