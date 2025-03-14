#!/usr/bin/env bash

D:/anaconda/python.exe ./kill.py
##############################################
# 1. 定义要跑的实验场景 与 聚合方式
##############################################
# 这里只跑 federated_20malicious 和 federated_50malicious
EXPERIMENTS=("federated_20malicious_v2")
# 只跑 FedAvg 和 AccWeight
AGGREGATORS=("AccWeight")

##############################################
# 2. 其他参数
##############################################
# 当收集到多少客户端更新后打包区块 (与客户端数一致)
CLIENT_COUNT=10

# 全局要跑多少轮 (FED_ROUNDS)
FED_ROUNDS=5

# 每个客户端每次启动接受几次联邦学习 应当为1
LOCAL_EPOCH=1

# 每轮结束后等待 (秒)
SLEEP_TIME=50  # 每轮提交后等待50秒看是否打包完成

# Python解释器
PYEXE="D:/anaconda/envs/BlockchainForFederatedLearning/python.exe"

# 工程 src 目录 (请按实际路径修改)
BASE_PATH="D:\BlockchainForFederatedLearning-master\cifar"
D:/anaconda/python.exe ./kill.py
##############################################
# 3. 清理 output 文件夹 (可选)
##############################################
rm -rf "${BASE_PATH}/output"/*

for EXP_NAME in "${EXPERIMENTS[@]}"; do
  for AGG in "${AGGREGATORS[@]}"; do

    echo -e "\n\n===== 开始实验: ${EXP_NAME}, 聚合方式: ${AGG} =====\n"

    # 清理区块、客户端输出
    rm -rf "${BASE_PATH}/blocks"/*.block
    rm -rf "${BASE_PATH}/clients"/*

    # 启动矿工 -g 1 表示创建创世区块，-l ${CLIENT_COUNT} 表示每个区块包含多少更新
    echo ">> 启动 miner.py, aggregator=${AGG}, update_limit=${CLIENT_COUNT}"
    "${PYEXE}" miner.py \
      -g 1 \
      -l ${CLIENT_COUNT} \
      --aggregator ${AGG} \
      > "${BASE_PATH}/output/miner_${EXP_NAME}_${AGG}.txt" 2>&1 &

    MINER_PID=$!
    echo "Miner PID=${MINER_PID} 稍等矿工就绪"
    sleep ${SLEEP_TIME} # 稍等矿工就绪

    # 多轮联邦训练
    echo ">> 准备进行 FED_ROUNDS=${FED_ROUNDS} 轮"
    for (( r=1; r<=${FED_ROUNDS}; r++ ))
    do
      echo "---- Global Round $r ----"
      # 让10个客户端按顺序各提交一次更新
      for (( i=0; i<${CLIENT_COUNT}; i++ ))
      do
         echo "---- Client $i training ----"
         "${PYEXE}" client.py \
           -d "./experiments/${EXP_NAME}/client_${i}.pkl" \
           -e ${LOCAL_EPOCH} \
           -gr 1 \
           -le 10 \
           >> "${BASE_PATH}/output/client_${i}_${EXP_NAME}_${AGG}.txt" 2>&1
      done

      # 等待矿工完成本轮打包
      echo ">> 等待 ${SLEEP_TIME} 秒，保证矿工完成本轮区块打包"
      sleep ${SLEEP_TIME}

    done
    D:/anaconda/python.exe ./kill.py
    # 结束矿工
    kill -9 ${MINER_PID}
    echo ">> 已结束 Miner，实验: ${EXP_NAME}, 聚合: ${AGG} 完成"

  done
done

echo -e "\n===== 全部实验执行完毕！请查看 ${BASE_PATH}/output/ 下日志 ====="
