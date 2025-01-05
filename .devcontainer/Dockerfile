FROM nvcr.io/nvidia/pytorch:23.09-py3
# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN pip3 install open3d

RUN apt-get update && apt-get install -y \
    libx11-6 \
    libx11-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxext6 \
    libxi6 \
    && apt-get clean

RUN  apt-get install -y \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    freeglut3 \
    freeglut3-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    python3-tk


# Preconfigure timezone and install python3-tk
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    apt-get update && apt-get install -y \
    python3-tk \
    tzdata \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN pip3 install trimesh "pyglet<2"