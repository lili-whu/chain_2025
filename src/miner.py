"""
 - Blockchain for Federated Learning -
           Mining script 
"""

import hashlib
import json
import math
import time
from flask import Flask,jsonify,request
from uuid import uuid4
import requests
import random
import pickle
from blockchain import *
from threading import Thread, Event
from federatedlearner import *
import numpy as np
import codecs
import os
import glob
from logging.config import dictConfig


def make_base():
# 通过一部分训练数据训练初始模型，作为创世区块
    ''' 
    Function to do the base level training on the first set of client data 
    for the genesis block
    '''
    reset()
    dataset = None
    with open("data/federated_data_0.d",'rb') as f:
        dataset = pickle.load(f)
    worker = NNWorker(dataset["train_images"],
        dataset["train_labels"],
        dataset["test_images"],
        dataset["test_labels"],
        0,
        "base0")
    worker.build_base()
    model = dict()
    model['model'] = worker.get_model()
    model['accuracy'] = worker.evaluate()
    worker.close()
    return model


# 异步执行工作量证明（PoW）的线程类， 在调用 run() 方法时，开始挖矿，生成一个区块，并在挖矿结束后调用 on_end_mining() 函数。
class PoWThread(Thread):
    def __init__(self, stop_event,blockchain,node_identifier):
        self.stop_event = stop_event
        Thread.__init__(self)
        self.blockchain = blockchain
        self.node_identifier = node_identifier
        self.response = None

    def run(self):
        block,stopped = self.blockchain.proof_of_work(self.stop_event)
        self.response = {
            'message':"End mining",
            'stopped': stopped,
            'block': str(block)
        }
        on_end_mining(stopped)
        app.logger.info("mining completed")


STOP_EVENT = Event()
count = 0
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
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

app = Flask(__name__)
status = {
    's':"receiving",
    'id':str(uuid4()).replace('-',''),
    # 区块链Node
    'blockchain': None,
    'address' : ""
    }

# mine() 函数用于启动挖矿线程，设置系统状态为 mining，表示当前正在挖矿。
def mine():
    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT,status["blockchain"],status["id"])
    status['s'] = "mining"
    thread.start()


# 挖矿结束时会调用这个函数。如果挖矿成功并停止，它会调用 resolve_conflicts 进行冲突解决，并通知所有其他节点停止挖矿。
def on_end_mining(stopped):
    if status['s'] == "receiving":
        return
    if stopped:
        status["blockchain"].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status["blockchain"].nodes:
        requests.get('http://{node}/stopmining'.format(node=node))

@app.route('/transactions/new',methods=['POST'])
def new_transaction():
    app.logger.info("New Transaction Received")
    time.sleep(1)
    if status['s'] != "receiving":
        return 'Miner not receiving', 400
    values = request.get_json()
    # 客户端，
    required = ['client','baseindex','update','datasize','computing_time']
    # 校验参数是否齐全
    if not all(k in values for k in required):
        return 'Missing values', 400
    if values['client'] in status['blockchain'].current_updates:
        return 'Model already stored', 400
    
    index = status['blockchain'].new_update(values['client'],
        values['baseindex'],
        dict(pickle.loads(codecs.decode(values['update'].encode(), "base64"))),
        values['datasize'],
        values['computing_time'])
    
    # 通知其他节点
    for node in status["blockchain"].nodes:
        requests.post('http://{node}/transactions/new'.format(node=node),
            json=request.get_json())
    

    # 接受模型的条件

    # 接收状态且时间允许，运行mining
    if (status['s']=='receiving' and (
        len(status["blockchain"].current_updates)>=status['blockchain'].last_block['update_limit']
        or time.time()-status['blockchain'].last_block['timestamp']>status['blockchain'].last_block['time_limit'])):
        app.logger.info("start mining")
        mine()
    
    response = {'message': "Update will be added to block {index}".format(index=index)}
    return jsonify(response),201

@app.route('/status',methods=['GET'])
def get_status():
    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].last_block['index']
        }
    return jsonify(response),200

# 返回完整的区块链和区块数量
@app.route('/chain',methods=['GET'])
def full_chain():
    response = {
        'chain': status['blockchain'].hashchain,
        'length':len(status['blockchain'].hashchain)
    }
    return jsonify(response),200

# 注册新节点到区块链网络中，并将该节点信息广播给其他已注册节点
@app.route('/nodes/register',methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Enter valid nodes in the list ", 400
    for node in nodes:
        if node!=status['address'] and not node in status['blockchain'].nodes:
            status['blockchain'].register_node(node)
            for miner in status['blockchain'].nodes:
                if miner!=node:
                    print("node",node,"miner",miner)
                    requests.post('http://{miner}/nodes/register'.format(miner=miner),
                        json={'nodes': [node]})
    response = {
        'message':"New nodes have been added",
        'total_nodes':list(status['blockchain'].nodes)
    }
    return jsonify(response),201

'''
    解释
    功能：这个端点用于获取指定的区块。
    流程：
    接受请求数据 hblock，其中包含了区块的基本信息，如索引 index 和哈希 hash。
    首先检查当前区块是否就是目标区块，如果是则直接返回。
    如果当前区块不是目标区块，则检查本地存储的区块文件是否包含该区块。如果有，读取并返回。
    如果本地也没有该区块，向其他节点发送请求以获取该区块并保存到本地文件系统。
    验证获取的区块的哈希值是否匹配。
    最后返回区块数据和其验证状态（有效或无效）。
'''
@app.route('/block',methods=['POST'])
def get_block():
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block","rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
            json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block","wb") as f:
                    pickle.dump(block,f)
    valid = False
    # 校验区块合法性
    if Blockchain.hash(str(block))==hblock['hash']:
        valid = True
    response = {
        'block': str(block),
        'valid': valid
    }
    return jsonify(response),200

# 获取某个特定区块中的联邦学习模型
@app.route('/model',methods=['POST'])
def get_model():
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block","rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
            json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block","wb") as f:
                    pickle.dump(block,f)
    valid = False
    model = block.basemodel
    if Blockchain.hash(codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode())==hblock['model_hash']:
        valid = True
    response = {
        'model': codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode(),
        'valid': valid
    }
    return jsonify(response),200

'''
    解释
    功能：这个端点用于处理共识机制。
    流程：
    调用 resolve_conflicts 函数来解决冲突。
    如果本节点的链被其他节点的链取代，则返回新的链。
    如果本节点的链仍然是权威链，则返回当前链。
'''
@app.route('/nodes/resolve',methods=["GET"])
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

# 用于停止当前节点的挖矿操作，并进行冲突解决
@app.route('/stopmining',methods=['GET'])
def stop_mining():
    status['blockchain'].resolve_conflicts(STOP_EVENT)
    response = {
        'mex':"stopped!"
    }
    return jsonify(response),200

# 删除之前存储在本地的所有区块文件，通常在启动时调用，以清理旧数据
def delete_prev_blocks():
    files = glob.glob('blocks/*.block')
    for f in files:
        os.remove(f)


# 生成创世区块，运行监听程序
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    parser.add_argument('-i', '--host', default='127.0.0.1', help='IP address of this miner')
    parser.add_argument('-g', '--genesis', default=0, type=int, help='instantiate genesis block')
    parser.add_argument('-l', '--ulimit', default=10, type=int, help='number of updates stored in one block')
    parser.add_argument('-ma', '--maddress', help='other miner IP:port')
    args = parser.parse_args()
    address = "{host}:{port}".format(host=args.host,port=args.port)
    status['address'] = address
    if args.genesis==0 and args.maddress==None:
        raise ValueError("Must set genesis=1 or specify maddress")
    delete_prev_blocks()
    if args.genesis==1:
        model = make_base()
        print("base model accuracy:",model['accuracy'])
        status['blockchain'] = Blockchain(address,model,True,args.ulimit)
    else:
        status['blockchain'] = Blockchain(address)
        status['blockchain'].register_node(args.maddress)
        requests.post('http://{node}/nodes/register'.format(node=args.maddress),
            json={'nodes': [address]})
        status['blockchain'].resolve_conflicts(STOP_EVENT)
    app.run(host=args.host,port=args.port)
