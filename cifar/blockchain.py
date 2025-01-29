# blockchain.py
import hashlib
import json
import time
from urllib.parse import urlparse
import requests
import random
import codecs
import pickle
import logging
import numpy as np
import tensorflow as tf
from app import app
import cifar_data_extractor as dataext
from federatedlearner import NNWorker, reset
logger = logging.getLogger(__name__)
accuracy_list = []

##########################################################
#               聚合函数：只保留2种核心方式
##########################################################

def compute_upd_2(weights, base, updates, lrate):
    """
    纯平均聚合 (FedAvg)。忽略外部传入weights，强行平分
    """
    # 强制平均
    n = len(weights)
    for client in weights:
        weights[client] = 1.0 / n
    app.logging.info(", ".join(str(x) for x in weights.values()))
    # 基于 base 的 param dict:
    upd = {}
    for k in base.keys():
        if k=="size":
            upd["size"] = 0
            continue
        upd[k] = np.zeros_like(base[k])

    for client, up in updates.items():
        model_dict = up.update
        for k in model_dict.keys():
            if k=="size":
                continue
            upd[k] += weights[client] * model_dict[k]

    upd["size"] = 0
    return upd



def compute_upd_3(weights, base, updates, lrate):
    """
    AccWeight模式聚合
    """
    app.logging.info(", ".join(str(x) for x in weights.values()))
    upd = {}
    for k in base.keys():
        if k=="size":
            upd["size"]=0
            continue
        upd[k] = np.zeros_like(base[k])

    for client, up in updates.items():
        model_dict = up.update
        for k in model_dict.keys():
            if k=="size":
                continue
            upd[k] += weights[client] * model_dict[k]

    upd["size"]=0
    return upd



def compute_global_model(base_block, updates, lrate, aggregator="FedAvg"):
    """
    根据 aggregator 参数决定要用哪个聚合函数。
    1. 先对每个局部模型进行准确率评估
    2. 计算 Acc + ΔAcc 的加权
    3. 按 aggregator 选择 compute_upd_2 (FedAvg) 或 compute_upd_3 (AccWeight)
    """

    base = base_block.basemodel
    accuracy_local_all = dict()

    # 1. 评估每个局部模型在测试集上的准确率
    # 假设 dataext.load_data("data/cifar_test.pkl") 返回 {'test_images':..., 'test_labels':...}
    # 1. 在线获取 CIFAR-10 测试集
    dataset = get_cifar_test_online()

    for client, update in updates.items():
        worker = NNWorker(None, None,
                          dataset['test_images'], dataset['test_labels'],
                          0, "validation", steps=0)
        worker.build(update.update)  # 这里用 CNN 构建
        evaluateAccuracy = worker.evaluate()
        worker.close()
        accuracy_local_all[client] = evaluateAccuracy

    # 2. 计算 (Acc + ΔAcc)
    alpha = 0.1
    beta = 0.9
    weights = {}
    total_weight = 0.0
    for client, update in updates.items():
        acc_i_t = accuracy_local_all[client]
        acc_change_i_t = acc_i_t - base_block.accuracy
        # MQI = alpha * Acc + beta * (Acc - Acc_global_prev)
        weight = alpha * acc_i_t + beta * acc_change_i_t
        weights[client] = weight
        # 避免除0
        if weights[client] <= 0:
            weights[client] = 1e-8
        total_weight += weight
        
    
    

    for client in updates.keys():
        weights[client] = weights[client] / total_weight

    # 3. 根据 aggregator 决定采用哪种聚合
    if aggregator == "AccWeight":
        upd = compute_upd_3(weights, base, updates, lrate)
    else:
        # 默认为 FedAvg
        upd = compute_upd_2(weights, base, updates, lrate)



    # 4. 计算新的全局模型精度
    tf.reset_default_graph()
    worker = NNWorker(None, None,
                      dataset['test_images'], dataset['test_labels'],
                      0, "validation", steps=0)
    worker.build(upd)
    accuracy = worker.evaluate()
    worker.close()

    return accuracy, upd


##########################################################
#    Update & Block 类：基本不变，只去掉不必要的函数
##########################################################

def find_len(text, strk):
    return text.find(strk), len(strk)

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

class Update:
    def __init__(self, client, baseindex, update, datasize, computing_time, timestamp=time.time()):
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time

    @staticmethod
    def from_string(metadata):
        i, l = find_len(metadata, "'timestamp':")
        i2, l2 = find_len(metadata, "'baseindex':")
        i3, l3 = find_len(metadata, "'update': ")
        i4, l4 = find_len(metadata, "'client':")
        i5, l5 = find_len(metadata, "'datasize':")
        i6, l6 = find_len(metadata, "'computing_time':")

        baseindex = int(metadata[i2 + l2:i3].replace(",", '').replace(" ", ""))
        update = dict(pickle.loads(codecs.decode(metadata[i3 + l3:i4 - 1].encode(), "base64")))
        timestamp = float(metadata[i + l:i2].replace(",", '').replace(" ", ""))
        client = metadata[i4 + l4:i5].replace(",", '').replace(" ", "")
        datasize = int(metadata[i5 + l5:i6].replace(",", '').replace(" ", ""))
        computing_time = float(metadata[i6 + l6:].replace(",", '').replace(" ", ""))

        return Update(client, baseindex, update, datasize, computing_time, timestamp)

    def __str__(self):
        return "'timestamp': {timestamp},\
            'baseindex': {baseindex},\
            'update': {update},\
            'client': {client},\
            'datasize': {datasize},\
            'computing_time': {computing_time}".format(
            timestamp=self.timestamp,
            baseindex=self.baseindex,
            update=codecs.encode(pickle.dumps(sorted(self.update.items())), "base64").decode(),
            client=self.client,
            datasize=self.datasize,
            computing_time=self.computing_time
        )


class Block:
    def __init__(self, miner, index, basemodel, accuracy, updates, timestamp=time.time()):
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(metadata):
        i, l = find_len(metadata, "'timestamp':")
        i2, l2 = find_len(metadata, "'basemodel': ")
        i3, l3 = find_len(metadata, "'index':")
        i4, l4 = find_len(metadata, "'miner':")
        i5, l5 = find_len(metadata, "'accuracy':")
        i6, l6 = find_len(metadata, "'updates':")
        i9, l9 = find_len(metadata, "'updates_size':")

        index = int(metadata[i3 + l3:i4].replace(",", '').replace(" ", ""))
        miner = metadata[i4 + l4:i].replace(",", '').replace(" ", "")
        timestamp = float(metadata[i + l:i2].replace(",", '').replace(" ", ""))

        basemodel = dict(pickle.loads(codecs.decode(metadata[i2 + l2:i5 - 1].encode(), "base64")))
        accuracy = float(metadata[i5 + l5:i6].replace(",", '').replace(" ", ""))

        su = metadata[i6 + l6:i9]
        su = su[:su.rfind("]") + 1]
        updates = dict()
        for x in json.loads(su):
            isep, lsep = find_len(x, "@|!|@")
            updates[x[:isep]] = Update.from_string(x[isep + lsep:])

        updates_size = int(metadata[i9 + l9:].replace(",", '').replace(" ", ""))
        return Block(miner, index, basemodel, accuracy, updates, timestamp)

    def __str__(self):
        return "'index': {index},\
            'miner': {miner},\
            'timestamp': {timestamp},\
            'basemodel': {basemodel},\
            'accuracy': {accuracy},\
            'updates': {updates},\
            'updates_size': {updates_size}".format(
            index=self.index,
            miner=self.miner,
            basemodel=codecs.encode(pickle.dumps(sorted(self.basemodel.items())), "base64").decode(),
            accuracy=self.accuracy,
            timestamp=self.timestamp,
            updates=str([str(x[0]) + "@|!|@" + str(x[1]) for x in sorted(self.updates.items())]),
            updates_size=str(len(self.updates))
        )


##########################################################
#                Blockchain 核心逻辑
##########################################################

class Blockchain(object):
    def __init__(self, miner_id, base_model=None, gen=False,
                 update_limit=10, time_limit=1800,
                 aggregator="FedAvg"):
        """
        :param aggregator: "FedAvg" or "AccWeight"
        """
        super(Blockchain, self).__init__()
        self.miner_id = miner_id
        self.curblock = None
        self.hashchain = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit
        self.aggregator = aggregator  # 新增：决定使用何种聚合方式
        self.accuracy_history = []

        if gen:
            genesis, hgenesis, _ = self.make_block(base_model=base_model, previous_hash=1)
            self.store_block(genesis, hgenesis)
        self.nodes = set()

    def register_node(self, address):
        if address[:4] != "http":
            address = "http://" + address
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def make_block(self, previous_hash=None, base_model=None):
        accuracy = 0
        basemodel = None
        time_limit = self.time_limit
        update_limit = self.update_limit

        if len(self.hashchain) > 0:
            update_limit = self.last_block['update_limit']
            time_limit = self.last_block['time_limit']

        if previous_hash is None:
            previous_hash = self.hash(str(sorted(self.last_block.items())))

        if base_model is not None:
            accuracy = base_model['accuracy']
            basemodel = base_model['model']
        elif len(self.current_updates) > 0:
            # -----------------------
            # 关键：调用 compute_global_model 时传入 self.aggregator
            # -----------------------
            base_block = self.curblock
            accuracy, basemodel = compute_global_model(base_block,
                                                       self.current_updates,
                                                       lrate=1,
                                                       aggregator=self.aggregator)

        index = len(self.hashchain) + 1
        block = Block(
            miner=self.miner_id,
            index=index,
            basemodel=basemodel,
            accuracy=accuracy,
            updates=self.current_updates
        )
        # 将accuracy存到accuracy_history
        self.accuracy_history.append(accuracy)

        hashblock = {
            'index': index,
            'hash': self.hash(str(block)),
            'proof': random.randint(0, 100000000),
            'previous_hash': previous_hash,
            'miner': self.miner_id,
            'accuracy': str(accuracy),
            'timestamp': time.time(),
            'time_limit': time_limit,
            'update_limit': update_limit,
            'model_hash': self.hash(codecs.encode(pickle.dumps(sorted(block.basemodel.items())), "base64").decode())
        }
        return block, hashblock, self.accuracy_history

    def store_block(self, block, hashblock):
        if self.curblock:
            with open("blocks/federated_model" + str(self.curblock.index) + ".block", "wb") as f:
                pickle.dump(self.curblock, f)
        self.curblock = block
        self.hashchain.append(hashblock)
        self.current_updates = dict()
        return hashblock

    def new_update(self, client, baseindex, update, datasize, computing_time):
        self.current_updates[client] = Update(
            client=client,
            baseindex=baseindex,
            update=update,
            datasize=datasize,
            computing_time=computing_time
        )
        return self.last_block['index'] + 1

    @staticmethod
    def hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def last_block(self):
        return self.hashchain[-1]

    def proof_of_work(self, stop_event):
        block, hblock, accuracy_history = self.make_block()
        stopped = False
        while self.valid_proof(str(sorted(hblock.items()))) is False:
            if stop_event.is_set():
                stopped = True
                break
            hblock['proof'] += 1
        if not stopped:
            self.store_block(block, hblock)
        return hblock, stopped, accuracy_history

    @staticmethod
    def valid_proof(block_data):
        guess_hash = hashlib.sha256(block_data.encode()).hexdigest()
        # 挖矿难度(这里只检查前两位是否'00')
        k = "00"
        return guess_hash[:len(k)] == k

    def valid_chain(self, hchain):
        last_block = hchain[0]
        curren_index = 1
        while curren_index < len(hchain):
            hblock = hchain[curren_index]
            if hblock['previous_hash'] != self.hash(str(sorted(last_block.items()))):
                print("prev_hash differ at block:", curren_index)
                return False
            if not self.valid_proof(str(sorted(hblock.items()))):
                print("invalid proof at block:", curren_index)
                return False
            last_block = hblock
            curren_index += 1
        return True

    def resolve_conflicts(self, stop_event):
        neighbours = self.nodes
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)

        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node

        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            resp = requests.post('http://{node}/block'.format(node=bnode),
                                 json={'hblock': hblock})
            self.current_updates = dict()
            if resp.status_code == 200:
                if resp.json()['valid']:
                    self.curblock = Block.from_string(resp.json()['block'])
            return True
        return False
