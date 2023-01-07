FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

WORKDIR /mnt/76B0F5A1B0F5684F/Docker

RUN apt update && apt install -y sudo fish
RUN pip install --upgrade pip
RUN pip install keras==2.1.1
RUN pip install pandas
RUN pip install opencv-python
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install pydot
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install fasttext
RUN pip install --upgrade gensim
RUN pip install numpy --upgrade

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
#RUN useradd -ms /bin/bash -g root -G sudo -p  dockeruser dockeruser
#RUN echo dockeruser:dockeruser | chpasswd
#USER dockeruser

WORKDIR  /
EXPOSE 8888

CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--allow-root"]
