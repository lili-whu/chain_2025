#!/usr/bin/env bash

clear

# 0. 前置准备：删除后台miner监听进程
#pid=$(lsof -t -i :5000)
#
#if [ -z "$pid" ]; then
#    echo "No process is using port 5000."
#else
#    # 终止进程
#    kill -9 $pid
#    echo "Process $pid using port 5000 has been terminated."
#fi

# 0.1 准备minst分割数据
#unbuffer  python3 data/federated_data_extractor.py > ./output/dataset_output.txt 2>&1 &

# 0.2 清空上一次运行得到的所有数据
TARGET_DIR="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src\clients"
rm -rf "${TARGET_DIR:?}"/*
TARGET_DIR="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src\blocks"
rm -rf "${TARGET_DIR:?}"/*
TARGET_DIR="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src\output"
rm -rf "${TARGET_DIR:?}"/*

