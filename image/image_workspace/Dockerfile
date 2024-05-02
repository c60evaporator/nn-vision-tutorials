FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Copy requirements.txt
COPY ./requirements.txt .

RUN apt update \
 && apt upgrade -y
# Setup python environment
RUN apt install -y \
    curl \
    python3 \
    python3-dev \
    python3-distutils \
    && curl https://bootstrap.pypa.io/pip/get-pip.py --output get-pip.py \
    && python3 get-pip.py \
    # Install PyTorch
    && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 \
    # Install other packages using requirements.txt
    && pip install -r requirements.txt

# Install common dependencies
RUN apt install -y git nano
# Create repository directory
RUN mkdir /repos

###### Object Detection ######
# Install YOLOX
WORKDIR /repos
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git
WORKDIR /repos/YOLOX
RUN pip install -r requirements.txt
RUN pip install -v -e .

# Install YOLOv5
WORKDIR /repos
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /repos/yolov5
RUN pip install -r requirements.txt
RUN apt install -y libgl1 \
    && ln -fs /usr/share/zoneinfo/US/Pacific /etc/localtime \
    && DEBIAN_FRONTEND=noninteractive apt install tzdata -y \
    && DEBIAN_FRONTEND=noninteractive apt install keyboard-configuration -y \
    && apt install -y libglib2.0-0

# Install DETR
RUN mkdir /repos/DETR
WORKDIR /repos/DETR
RUN git clone https://github.com/facebookresearch/detr.git
RUN pip install cython

###### Semantic Segmentation ######
# Install segmentation_models.pytorch (https://github.com/qubvel/segmentation_models.pytorch)
RUN pip install segmentation-models-pytorch

# Install Segment Anything
RUN pip install git+https://github.com/facebookresearch/segment-anything.git


# Register working directory
WORKDIR /scripts
