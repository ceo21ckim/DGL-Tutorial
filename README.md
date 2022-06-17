# DGL


## Docker Setting 
1. 도커 이미지 불러오기
```
docker pull nilsine11202/dgl-tutorial:1.0
```

2. 컨테이너 실행
```
docker run --itd --runtime=nvidia --name dgl_tutorial -p 8885:8885 -v C:\Users\Name\:/workspace nilsine11202/dgl-1.0 /bin/bash
```
