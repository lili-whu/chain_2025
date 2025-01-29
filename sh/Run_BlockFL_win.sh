#!/usr/bin/env bash

chcp 65001  # Windows下切换到UTF-8，避免中文乱码


##############################################
# 1. 定义实验场景 与 聚合方式
##############################################
#EXPERIMENTS=("centralized" "federated_normal" "federated_20malicious" "federated_50malicious")
#AGGREGATORS=("FedAvg" "AccWeight")

EXPERIMENTS=("federated_20malicious" "federated_50malicious")
AGGREGATORS=("FedAvg" "AccWeight")

##############################################
# 2. 其他参数
##############################################


FED_ROUNDS=10

# 每轮客户端本地训练的 epoch
LOCAL_EPOCH=1

# 等待时间: 用于等矿工挖矿
SLEEP_TIME=360  # 每轮提交后等待180秒看是否打包完成

# Python解释器路径
PYEXE="D:/anaconda/envs/BlockchainForFederatedLearning/python.exe"

# 路径
BASE_PATH="C:\Users\xiaoming\Desktop\BlockchainForFederatedLearning-master\src"

rm -rf "${BASE_PATH}/output"/*
##############################################
# 3. 循环 4(分布)*2(聚合)=8种组合
##############################################
for EXP_NAME in "${EXPERIMENTS[@]}"; do
  for AGG in "${AGGREGATORS[@]}"; do
    
    echo -e "\n\n===== 开始实验: ${EXP_NAME}, 聚合方式: ${AGG} =====\n"

    # 先根据分布决定客户端数量
    if [ "$EXP_NAME" == "centralized" ]; then
      CLIENT_COUNT=1
    else
      CLIENT_COUNT=10
    fi

    # 先清理上次实验的区块(可选)
    rm -rf "${BASE_PATH}/blocks"/*.block
    rm -rf "${BASE_PATH}/clients"/*

    # 启动矿工
    echo ">> 启动矿工 aggregator=${AGG}, update_limit=${CLIENT_COUNT}"
    "${PYEXE}" miner.py \
      -g 1 \
      -l ${CLIENT_COUNT} \
      --aggregator ${AGG} \
      > "${BASE_PATH}/output/miner_${EXP_NAME}_${AGG}.txt" 2>&1 &

    MINER_PID=$!
    echo "Miner PID=${MINER_PID}"
    sleep 180  # 稍等矿工就绪

    # 多轮联邦训练
    echo ">> 准备进行 FED_ROUNDS=${FED_ROUNDS} 轮"
    for (( r=1; r<=${FED_ROUNDS}; r++ ))
    do
      echo "---- Round $r ----"
      # 每个客户端提交1次更新
      for (( i=0; i<${CLIENT_COUNT}; i++ ))
      do
         "${PYEXE}" client.py \
           -d "./experiments/${EXP_NAME}/client_${i}.pkl" \
           -e ${LOCAL_EPOCH} \
           >> "${BASE_PATH}/output/client_${i}_${EXP_NAME}_${AGG}.txt" 2>&1
         # 注意这里不再后台 & 了，而是顺序执行；也可并发执行，然后 sleep
      done

      # 等待矿工完成本轮区块
      echo ">> 等待 ${SLEEP_TIME} 秒，保证矿工完成本轮打包"
      sleep ${SLEEP_TIME}

    done

    # R 轮结束后，结束矿工
    kill -9 ${MINER_PID}
    echo ">> 已结束 Miner，实验: ${EXP_NAME}, 聚合: ${AGG} 完成"

  done
done

echo -e "\n===== 全部 8 组实验执行完毕！请查看 ${BASE_PATH}/output/ 下日志 ====="
