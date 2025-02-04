# miner.py
# - Blockchain for Federated Learning -
#   Mining script

import os
import time
import pickle
import requests
import glob

from flask import Flask, jsonify, request
from uuid import uuid4
from blockchain import Blockchain, Block  # 这里的blockchain.py里包含了Blockchain类
from threading import Thread, Event
from federatedlearner import *
import codecs
from logging.config import dictConfig
from argparse import ArgumentParser
from app import app
############################################################
#                Flask 应用和全局状态
############################################################

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})



status = {
    's': "receiving",          # 当前状态：receiving or mining
    'id': str(uuid4()).replace('-', ''),  # 矿工节点 ID
    'blockchain': None,        # Blockchain 对象
    'address': ""              # 当前节点的地址 "host:port"
}

STOP_EVENT = Event()

def get_cifar_test_online():
    """
    自动从网上下载 CIFAR-10（若本地已缓存，则直接读取），并返回
    一个字典,如 {'test_images': ..., 'test_labels': ...}
    """
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    # test_x: (10000,32,32,3), test_y: (10000,1)
    test_x = test_x.astype('float32') / 255.0
    # One-hot
    test_y = tf.keras.utils.to_categorical(test_y, 10)

    return {
        "test_images": test_x,
        "test_labels": test_y
    }


############################################################
#                    工作量证明线程
############################################################

class PoWThread(Thread):
    """
    异步挖矿线程：启动后执行 proof_of_work()，结束后回调 on_end_mining()
    """
    def __init__(self, stop_event, blockchain, node_identifier):
        super().__init__()
        self.stop_event = stop_event
        self.blockchain = blockchain
        self.node_identifier = node_identifier
        self.response = None

    def run(self):
        block, stopped, accuracy_history = self.blockchain.proof_of_work(self.stop_event)
        app.logger.info("聚合轮次: " + str(len(accuracy_history)) + "轮")
        app.logger.info(", ".join(str(x) for x in accuracy_history))
        self.response = {
            'message': "End mining",
            'stopped': stopped,
            'block': str(block)
        }
        on_end_mining(stopped)


############################################################
#                      工具函数
############################################################

def delete_prev_blocks():
    """
    删除之前存储在本地的所有区块文件，通常在启动时调用，以清理旧数据
    """
    files = glob.glob('blocks/*.block')
    for f in files:
        os.remove(f)


def make_base():
    """
    用一部分训练数据训练初始模型，作为创世区块的模型
    也可以直接用随机初始化/空模型，这里只是示例
    """
    reset()

    dataset = get_cifar_test_online()

    worker = NNWorker(None, None,
                        dataset['test_images'], dataset['test_labels'],
                        0, "validation", steps=0)
    worker.build_base()
    model = dict()
    model['model'] = worker.get_model()
    model['accuracy'] = worker.evaluate()
    worker.close()
    return model


def mine():
    """
    启动挖矿线程
    """
    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT, status["blockchain"], status["id"])
    status['s'] = "mining"
    thread.start()


def on_end_mining(stopped):
    """
    挖矿结束时的回调
    """
    if status['s'] == "receiving":
        return
    if stopped:
        status["blockchain"].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status["blockchain"].nodes:
        requests.get('http://{node}/stopmining'.format(node=node))


############################################################
#                   Flask 路由接口
############################################################

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    """
    客户端提交训练完成后的本地模型更新
    """
    time.sleep(1)
    if status['s'] != "receiving":
        return 'Miner not receiving', 400

    values = request.get_json()
    required = ['client', 'baseindex', 'update', 'datasize', 'computing_time']
    if not all(k in values for k in required):
        return 'Missing values', 400

    if values['client'] in status['blockchain'].current_updates:
        return 'Model already stored', 400

    index = status['blockchain'].new_update(
        values['client'],
        values['baseindex'],
        dict(pickle.loads(codecs.decode(values['update'].encode(), "base64"))),
        values['datasize'],
        values['computing_time']
    )

    # 将收到的更新再转发给其他节点，以达成同步
    for node in status["blockchain"].nodes:
        requests.post('http://{node}/transactions/new'.format(node=node),
                      json=request.get_json())

    # 当累计到一定数量的 updates 或达到时间限制，就启动挖矿
    if (status['s'] == 'receiving' and (
            len(status["blockchain"].current_updates) >= status['blockchain'].last_block['update_limit']
            or time.time() - status['blockchain'].last_block['timestamp'] > status['blockchain'].last_block['time_limit'])):
        mine()

    response = {'message': f"Update will be added to block {index}"}
    return jsonify(response), 201


@app.route('/status', methods=['GET'])
def get_status():
    """
    查看当前矿工状态
    """
    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].last_block['index']
    }
    return jsonify(response), 200


@app.route('/chain', methods=['GET'])
def full_chain():
    """
    返回完整的区块链和区块数量
    """
    response = {
        'chain': status['blockchain'].hashchain,
        'length': len(status['blockchain'].hashchain)
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    """
    注册新节点到区块链网络中，并将该节点信息广播给其他已注册节点
    """
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Enter valid nodes in the list ", 400

    for node in nodes:
        if node != status['address'] and node not in status['blockchain'].nodes:
            status['blockchain'].register_node(node)
            # 将这个新节点也广播给其他节点
            for miner in status['blockchain'].nodes:
                if miner != node:
                    requests.post(f'http://{miner}/nodes/register',
                                  json={'nodes': [node]})
    response = {
        'message': "New nodes have been added",
        'total_nodes': list(status['blockchain'].nodes)
    }
    return jsonify(response), 201


@app.route('/block', methods=['POST'])
def get_block():
    """
    获取指定区块
    """
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock and status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile(f"./blocks/federated_model{hblock['index']}.block"):
        with open(f"./blocks/federated_model{hblock['index']}.block", "rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
                             json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open(f"./blocks/federated_model{hblock['index']}.block", "wb") as f:
                    pickle.dump(block, f)

    valid = False
    if block and status['blockchain'].hash(str(block)) == hblock['hash']:
        valid = True
    response = {
        'block': str(block),
        'valid': valid
    }
    return jsonify(response), 200


@app.route('/model', methods=['POST'])
def get_model():
    """
    从指定区块中获取模型参数
    """
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock and status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile(f"./blocks/federated_model{hblock['index']}.block"):
        with open(f"./blocks/federated_model{hblock['index']}.block", "rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
                             json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open(f"./blocks/federated_model{hblock['index']}.block", "wb") as f:
                    pickle.dump(block, f)

    valid = False
    if block:
        model = block.basemodel
        # 校验model哈希
        if status['blockchain'].hash(codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode()) == hblock['model_hash']:
            valid = True
        response = {
            'model': codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode(),
            'valid': valid
        }
        return jsonify(response), 200
    else:
        return jsonify({'model': None, 'valid': False}), 200


@app.route('/nodes/resolve', methods=["GET"])
def consensus():
    replaced = status['blockchain'].resolve_conflicts(STOP_EVENT)
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': status['blockchain'].hashchain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': status['blockchain'].hashchain
        }
    return jsonify(response), 200


@app.route('/stopmining', methods=['GET'])
def stop_mining():
    status['blockchain'].resolve_conflicts(STOP_EVENT)
    response = {
        'mex': "stopped!"
    }
    return jsonify(response), 200


############################################################
#                   主函数入口
############################################################

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    parser.add_argument('-i', '--host', default='127.0.0.1', help='IP address of this miner')
    parser.add_argument('-g', '--genesis', default=0, type=int, help='instantiate genesis block (0 or 1)')
    parser.add_argument('-l', '--ulimit', default=10, type=int, help='number of updates stored in one block')
    parser.add_argument('-ma', '--maddress', help='other miner IP:port')
    # 新增 aggregator 参数
    parser.add_argument('--aggregator', default='FedAvg', help='aggregation method: FedAvg or AccWeight')

    args = parser.parse_args()
    address = f"{args.host}:{args.port}"
    status['address'] = address

    # 删除旧区块文件
    delete_prev_blocks()

    if args.genesis == 1:
        # 创世区块：使用 make_base() 生成初始模型
        base_model = make_base()
        print("base model accuracy:", base_model['accuracy'])
        status['blockchain'] = Blockchain(
            miner_id=address,
            base_model=base_model,
            gen=True,
            update_limit=args.ulimit,
            aggregator=args.aggregator  # 把聚合方式传入
        )
    else:
        # 非创世：加入已有矿工节点
        status['blockchain'] = Blockchain(
            miner_id=address,
            base_model=None,
            gen=False,
            update_limit=args.ulimit,
            aggregator=args.aggregator
        )
        if args.maddress:
            status['blockchain'].register_node(args.maddress)
            requests.post(f'http://{args.maddress}/nodes/register', json={'nodes': [address]})
            status['blockchain'].resolve_conflicts(STOP_EVENT)
        else:
            # 如果既没有genesis，也没指定其他矿工地址，则可能报错
            print("Warning: no genesis, no maddress, chain might be empty...")

    app.run(host=args.host, port=args.port)
