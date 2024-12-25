FROM nvcr.io/nvidia/pytorch:23.09-py3
RUN pip3 install open3d

RUN apt-get update && apt-get install -y \
    libx11-6 \
    libx11-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxext6 \
    libxi6 \
    && apt-get clean

RUN  apt-get install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module