FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && conda install -c dglteam dgl-cuda11.3

RUN conda install pyg -c pyg && pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html \
    && pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html \
    && pip install torch-geometric

RUN pip install jupyter

WORKDIR /workspace


CMD ["bash"]

