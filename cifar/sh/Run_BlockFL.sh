#!/usr/bin/env bash

clear

# 0. 前置准备：删除后台miner监听进程
pid=$(lsof -t -i :5000)

if [ -z "$pid" ]; then
    echo "No process is using port 5000."
else
    # 终止进程
    kill -9 $pid
    echo "Process $pid using port 5000 has been terminated."
fi

# 0.1 准备minst分割数据

# 0.2 清空上一次运行得到的所有数据
TARGET_DIR="/Users/lili/Desktop/毕业论文2025-1-19/BlockchainForFederatedLearning-master/src/clients/"
rm -rf "${TARGET_DIR:?}"/*
TARGET_DIR="/Users/lili/Desktop/毕业论文2025-1-19/BlockchainForFederatedLearning-master/src/blocks/"
rm -rf "${TARGET_DIR:?}"/*
TARGET_DIR="/Users/lili/Desktop/毕业论文2025-1-19/BlockchainForFederatedLearning-master/src/output/"
rm -rf "${TARGET_DIR:?}"/*

# 客户端数量
NUM_CLIENTS=10

#1. 运行区块链挖矿程序，完成创世区块创建及端口监听
echo "Start federated learning on n clients:"
# 后台运行 miner.py 并将输出写入到 miner_output.txt
python3 miner.py -g 1 -l $NUM_CLIENTS > ./output/miner_output.txt 2>&1 &

sleep 5


echo "Creating datasets for n clients:"

sleep 3

# 每个客户端运行的轮次
NUM_RUNS=10

for j in `seq 1 $NUM_RUNS`;
do
    for i in `seq 0 $((NUM_CLIENTS - 1))`;
    do
        echo "Run $j for client $i"
        # 后台运行 client.py 并将每个客户端的输出追加写入 client_i_output.txt
        python3 client.py -d "data/federated_data_$i.d" -e 1 -tr 1 >> "./output/client_${i}_output.txt" 2>&1 &
    done
    sleep 100
done

sleep 100
