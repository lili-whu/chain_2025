#!/usr/bin/env bash

##############################################
#     1. 定义实验场景与聚合方式
##############################################
# 这里列出你要跑的4种数据分布
EXPERIMENTS=("centralized" "federated_normal" "federated_20malicious" "federated_50malicious")
# 两种聚合方式
AGGREGATORS=("FedAvg" "AccWeight")

##############################################
#     2. 开始循环跑 4*2 = 8种组合
##############################################
for EXP_NAME in "${EXPERIMENTS[@]}"; do
  for AGG in "${AGGREGATORS[@]}"; do
    
    echo -e "\n\n===== 开始实验: ${EXP_NAME}, 聚合方式: ${AGG} =====\n"

    # （1）根据 EXP_NAME 决定客户端数量
    # centralized 场景只有 1个client，其余场景均是10个client
    if [ "$EXP_NAME" == "centralized" ]; then
      CLIENT_COUNT=1
    else
      CLIENT_COUNT=10
    fi

    # (2) 清空上一次运行得到的所有数据
    TARGET_DIR="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src\clients"
    rm -rf "${TARGET_DIR:?}"/*
    TARGET_DIR="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src\blocks"
    rm -rf "${TARGET_DIR:?}"/*
    TARGET_DIR="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src\output"
    rm -rf "${TARGET_DIR:?}"/*

    # （3）启动矿工：-g 1 表示创建创世区块，-l 设置 update_limit，可与 CLIENT_COUNT 相同
    echo ">> 启动 miner.py, aggregator=${AGG}, update_limit=${CLIENT_COUNT}"
    D:/anaconda/envs/BlockchainForFederatedLearning/python.exe miner.py \
      -g 1 \
      -l ${CLIENT_COUNT} \
      --aggregator ${AGG} \
      > "output/miner_${EXP_NAME}_${AGG}.txt" 2>&1 &

    MINER_PID=$!
    echo "Miner PID = ${MINER_PID}"
    sleep 5  # 等待矿工准备就绪

    # （4）启动相应客户端，每个客户端使用自己的 .pkl 数据
    echo ">> 启动 ${CLIENT_COUNT} 个 client.py..."
    for (( i=0; i<${CLIENT_COUNT}; i++ ))
    do
       D:/anaconda/envs/BlockchainForFederatedLearning/python.exe client.py \
         -d "./experiments/${EXP_NAME}/client_${i}.pkl" \
         -e 1 \
         >> "output/client_${i}_${EXP_NAME}_${AGG}.txt" 2>&1 &
    done

    # 视情况等待若干秒（保证客户端都能训练 & 提交更新）
    echo ">> 等待 20~30 秒，客户端训练与提交更新"
    sleep 30

    # （5）结束矿工进程
    kill -9 ${MINER_PID}
    echo ">> 已结束 Miner 进程，实验: ${EXP_NAME}, 聚合: ${AGG} 完成"
    
  done
done

echo -e "\n===== 全部 8 组实验执行完毕！请查看 output/ 文件夹下日志 ====="
