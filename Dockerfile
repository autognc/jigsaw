FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6

COPY /jigsaw /jigsaw
COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml

WORKDIR /jigsaw

CMD ["/bin/bash", "-c", "source activate jigsaw && ./cli.py"]
