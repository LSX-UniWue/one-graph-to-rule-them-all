FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
LABEL org.opencontainers.image.authors="ingo.scholtes@uni-wuerzburg.de"
LABEL "com.example.vendor"="University of Würzburg, ML4Nets"
COPY *.ipynb /notebooks/demos/

WORKDIR /home/user

ENV DEBIAN_FRONTEND=noninteractive
RUN touch `date` && apt-get update

RUN apt install python3 python3-pip python3-dev python3-venv zsh byobu htop vim git wget lsof fuse parallel rsync curl cmake g++ -y

RUN pip3 install --upgrade pip

## config
ARG USER=zehe
ARG UID=1052


## setup (you don't have to touch this)

RUN adduser ${USER} --uid ${UID} --home /home/ls6/zehe/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/ls6/${USER}
RUN chown -R ${USER} /home/ls6/${USER}

RUN mkdir -p /pip
RUN chown -R ${USER} /pip

RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_sm

RUN apt-get update && apt install git --assume-yes

USER ${UID}
RUN python3 -m venv --system-site-packages /pip

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip3 install git+https://github.com/pathpy/pathpy.git@master
RUN pip3 install zstandard
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install seaborn
RUN pip3 install jedi==0.17.2
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip3 install ogb
RUN /opt/conda/bin/conda install jupyter -y --quiet
EXPOSE 8888
CMD cd /notebooks && /opt/conda/bin/jupyter notebook --ip='*' --port=8888 --no-browser --allow-root