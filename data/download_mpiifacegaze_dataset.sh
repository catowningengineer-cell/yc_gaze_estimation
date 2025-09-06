#!/usr/bin/env bash

set -Ceu

# 切换到脚本所在目录下的 data 文件夹
mkdir -p ../data
cd ../data

# 下载并解压数据集
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip
unzip MPIIFaceGaze_normalized.zip

# 重命名（修正拼写错误）
mv MPIIFaceGaze_normalizad MPIIFaceGaze_normalized

# 可选：删除压缩包节省空间
rm MPIIFaceGaze_normalized.zip
