#!/bin/bash
set -e

PROJECT_DIR=/home/xilinx/zcu104_croprow_ncnn
NCNN_DIR=/home/xilinx/build_ai/ncnn/build/install

cd $PROJECT_DIR/build

g++ -O2 ../src/croprow_seg_image.cpp -o croprow_seg_image \
  -I$NCNN_DIR/include/ncnn \
  -I/usr/include/opencv4 \
  -L$NCNN_DIR/lib \
  -lncnn \
  -lopencv_core \
  -lopencv_imgproc \
  -lopencv_imgcodecs \
  -lopencv_videoio \
  -fopenmp \
  -lgomp \
  -pthread \
  -ldl \
  -latomic

echo "Build image app done"
